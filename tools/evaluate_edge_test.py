#!/usr/bin/env python3
"""Evaluate trained jiant edge-probing runs on labeled test caches."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.edge_test_gold_labels import build_gold_matrix_from_test_cache, load_test_lines

import jiant.proj.main.runscript as main_runscript
import jiant.proj.main.components.container_setup as container_setup
import jiant.proj.main.components.evaluate as main_evaluate
import jiant.proj.main.components.task_sampler as task_sampler
import jiant.shared.caching as shared_caching
import jiant.shared.initialization as initialization


def write_metrics(results_dict, metrics_aggregator, output_path, verbose=True):
    results_to_write = {
        "aggregated": task_sampler.compute_aggregate_major_metrics_from_results_dict(
            metrics_aggregator=metrics_aggregator,
            results_dict=results_dict,
        ),
    }
    for task_name, task_results in results_dict.items():
        task_results_to_write = {}
        if "loss" in task_results:
            task_results_to_write["loss"] = task_results["loss"]
        if "metrics" in task_results:
            task_results_to_write["metrics"] = task_results["metrics"].to_dict()
        results_to_write[task_name] = task_results_to_write

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_to_write, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(json.dumps(results_to_write, indent=2))
    return results_to_write


def ensure_real_test_labels(
    task, test_cache, test_labels_dir, test_jsonl: Path, chunk_size: int, force: bool = False
):
    """Write test_labels cache from **true** gold in ``test.jsonl`` (order aligned by ``guid``).

    jiant's ``get_test_examples()`` uses placeholder labels for ``test`` split, so the test
    **feature** cache carries dummy ``label_ids``. Evaluation must supply gold via a parallel
    ``test_labels`` cache built from the JSONL targets.
    """
    data_args_path = test_labels_dir / "data_args.p"
    if data_args_path.exists() and not force:
        return

    if not test_jsonl.is_file():
        raise FileNotFoundError(f"Missing test JSONL for gold labels: {test_jsonl}")

    if test_labels_dir.exists():
        shutil.rmtree(test_labels_dir)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    test_lines = load_test_lines(test_jsonl)
    gold = build_gold_matrix_from_test_cache(task, test_cache, test_lines)
    shared_caching.chunk_and_save(
        data=gold,
        chunk_size=chunk_size,
        data_args={
            "phase": "test_labels",
            "task": task.name,
            "chunk_size": chunk_size,
        },
        output_dir=str(test_labels_dir),
    )


def append_summary_tsv(summary_path: Path, task: str, run_name: str, metrics_path: Path, metrics: dict):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not summary_path.exists()
    task_data = metrics[task]
    minor = task_data["metrics"]["minor"]
    row = [
        task,
        run_name,
        f'{task_data["loss"]:.8f}',
        f'{minor.get("acc", float("nan")):.8f}',
        f'{minor.get("f1_micro", float("nan")):.8f}',
        f'{task_data["metrics"]["major"]:.8f}',
        str(metrics_path),
    ]
    with summary_path.open("a", encoding="utf-8") as f:
        if new_file:
            f.write("task\trun_name\tloss\tacc\tf1_micro\tmajor\ttest_metrics_path\n")
        f.write("\t".join(row) + "\n")


def build_eval_config(task, base_runconfig_path, exp_dir, model, work_dir, test_labels_dir):
    config = json.loads(base_runconfig_path.read_text(encoding="utf-8"))
    task_cache = config["task_cache_config_dict"][task]
    task_cache["val"] = str(exp_dir / "cache" / model / task / "test")
    task_cache["val_labels"] = str(test_labels_dir)

    config["global_train_config"]["max_steps"] = 0
    config["global_train_config"]["warmup_steps"] = 0
    config["task_run_config"]["train_task_list"] = []
    config["task_run_config"]["train_val_task_list"] = []
    config["task_run_config"]["val_task_list"] = [task]
    config["task_run_config"]["test_task_list"] = []

    config_path = work_dir / f"{task}.test_eval_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def evaluate_one(args):
    exp_dir = Path(args.exp_dir)
    run_dir = exp_dir / "runs" / args.model / args.task / args.run_name
    best_model_path = run_dir / args.checkpoint_name
    if not best_model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_model_path}")

    base_runconfig_path = (
        exp_dir / "runconfigs" / args.model / args.task / f"{args.run_name}.json"
    )
    if not base_runconfig_path.exists():
        raise FileNotFoundError(f"Missing runconfig: {base_runconfig_path}")

    test_cache_dir = exp_dir / "cache" / args.model / args.task / "test"
    if not (test_cache_dir / "data_args.p").exists():
        raise FileNotFoundError(f"Missing test cache: {test_cache_dir}")

    model_config_root = exp_dir / "models" / args.model
    model_config = json.loads((model_config_root / "config.json").read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir = exp_dir / "cache" / args.model / args.task / "test_labels"

    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    task_container_probe = container_setup.create_jiant_task_container_from_json(
        jiant_task_container_config_path=str(base_runconfig_path),
        verbose=False,
    )
    task = task_container_probe.task_dict[args.task]
    test_cache = shared_caching.ChunkedFilesDataCache(str(test_cache_dir))
    test_jsonl = Path(args.test_jsonl) if args.test_jsonl else Path(task.test_path)
    ensure_real_test_labels(
        task=task,
        test_cache=test_cache,
        test_labels_dir=test_labels_dir,
        test_jsonl=test_jsonl,
        chunk_size=args.label_chunk_size,
        force=not args.reuse_test_labels,
    )

    eval_config_path = build_eval_config(
        task=args.task,
        base_runconfig_path=base_runconfig_path,
        exp_dir=exp_dir,
        model=args.model,
        work_dir=work_dir,
        test_labels_dir=test_labels_dir,
    )

    run_args = main_runscript.RunConfiguration.from_dict(
        {
            "jiant_task_container_config_path": str(eval_config_path),
            "output_dir": str(output_dir),
            "hf_pretrained_model_name_or_path": model_config["hf_pretrained_model_name_or_path"],
            "model_path": str(best_model_path),
            "model_config_path": model_config["model_config_path"],
            "model_load_mode": "all",
            "do_train": False,
            "do_val": True,
            "do_save": False,
            "do_save_last": False,
            "do_save_best": False,
            "write_val_preds": args.write_preds,
            "write_test_preds": False,
            "eval_every_steps": 0,
            "save_every_steps": 0,
            "save_checkpoint_every_steps": 0,
            "no_improvements_for_n_evals": 0,
            "keep_checkpoint_when_done": False,
            "force_overwrite": True,
            "learning_rate": 1e-5,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "optimizer_type": "adam",
            "no_cuda": args.no_cuda,
            "fp16": False,
            "fp16_opt_level": "O1",
            "local_rank": -1,
            "server_ip": "",
            "server_port": "",
            "seed": args.seed,
        }
    )

    quick_init_out = initialization.quick_init(args=run_args, verbose=True)
    with quick_init_out.log_writer.log_context():
        jiant_task_container = container_setup.create_jiant_task_container_from_json(
            jiant_task_container_config_path=str(eval_config_path),
            verbose=True,
        )
        runner = main_runscript.setup_runner(
            args=run_args,
            jiant_task_container=jiant_task_container,
            quick_init_out=quick_init_out,
            verbose=True,
        )
        results_dict = runner.run_val(
            task_name_list=[args.task],
            return_preds=args.write_preds,
            verbose=True,
        )
        metrics = write_metrics(
            results_dict=results_dict,
            metrics_aggregator=jiant_task_container.metrics_aggregator,
            output_path=output_dir / "test_metrics.json",
            verbose=True,
        )
        if args.write_preds:
            main_evaluate.write_preds(
                eval_results_dict=results_dict,
                path=str(output_dir / "test_preds.p"),
            )

    torch.save({"task": args.task, "run_name": args.run_name}, output_dir / "test_eval.done")

    if args.summary_tsv:
        append_summary_tsv(
            summary_path=Path(args.summary_tsv),
            task=args.task,
            run_name=args.run_name,
            metrics_path=output_dir / "test_metrics.json",
            metrics=metrics,
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--exp-dir", default="/cluster/home/zhangzx/my_project/base_exp/exp_edge")
    parser.add_argument("--model", default="bert-large-uncased")
    parser.add_argument("--checkpoint-name", default="best_model.p")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label-chunk-size", type=int, default=10000)
    parser.add_argument(
        "--reuse-test-labels",
        action="store_true",
        help="If set, reuse existing test_labels cache instead of rebuilding from test.jsonl.",
    )
    parser.add_argument(
        "--test-jsonl",
        default=None,
        help="Override path to test.jsonl for gold labels (default: task test_path from config).",
    )
    parser.add_argument(
        "--summary-tsv",
        default=None,
        help="Append one result row to this TSV (creates file with header if missing).",
    )
    parser.add_argument("--write-preds", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    metrics = evaluate_one(args)
    task_metrics = metrics[args.task]["metrics"]
    minor = task_metrics["minor"]
    print(
        "TEST_RESULT\t{task}\t{run}\t{loss:.8f}\t{acc:.8f}\t{f1:.8f}\t{major:.8f}".format(
            task=args.task,
            run=args.run_name,
            loss=metrics[args.task]["loss"],
            acc=minor.get("acc", float("nan")),
            f1=minor.get("f1_micro", float("nan")),
            major=task_metrics["major"],
        )
    )


if __name__ == "__main__":
    main()
