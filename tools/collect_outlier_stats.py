#!/usr/bin/env python3
"""Collect weight and activation outlier statistics for BERT-large baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import BertConfig, BertModel


LINEAR_MODULES = {
    "attention.self.query": "attn_q",
    "attention.self.key": "attn_k",
    "attention.self.value": "attn_v",
    "attention.output.dense": "attn_out",
    "intermediate.dense": "ffn_in",
    "output.dense": "ffn_out",
}

ACTIVATION_THRESHOLDS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
WEIGHT_SIGMA_THRESHOLDS = [3.0, 4.0, 5.0]


RUNS = [
    {
        "task": "ner",
        "requested_run_name": "formal_e3_ms256_retry1",
        "run_name": "formal_e3_ms256_tner",
        "note": "NER formal e3 is named formal_e3_ms256_tner in this project.",
    },
    {
        "task": "dep",
        "requested_run_name": "formal_e3_ms256_retry1",
        "run_name": "formal_e3_ms256_retry1",
        "note": "",
    },
    {
        "task": "semeval",
        "requested_run_name": "formal_e10_ms256",
        "run_name": "formal_e10_ms256",
        "note": "",
    },
]


class RunningAbsStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.abs_sum = 0.0
        self.abs_sq_sum = 0.0
        self.max_abs = 0.0
        self.gt_counts = {threshold: 0 for threshold in ACTIVATION_THRESHOLDS}

    @torch.no_grad()
    def update(self, tensor: torch.Tensor) -> None:
        x = tensor.detach().float().reshape(-1)
        if x.numel() == 0:
            return
        ax = x.abs()
        self.count += int(x.numel())
        self.sum += float(x.sum().item())
        self.sq_sum += float((x * x).sum().item())
        self.abs_sum += float(ax.sum().item())
        self.abs_sq_sum += float((ax * ax).sum().item())
        self.max_abs = max(self.max_abs, float(ax.max().item()))
        for threshold in ACTIVATION_THRESHOLDS:
            self.gt_counts[threshold] += int((ax > threshold).sum().item())

    def finalize(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "abs_mean": 0.0,
                "abs_std": 0.0,
                "max_abs": 0.0,
            }
        mean = self.sum / self.count
        var = max(self.sq_sum / self.count - mean * mean, 0.0)
        abs_mean = self.abs_sum / self.count
        abs_var = max(self.abs_sq_sum / self.count - abs_mean * abs_mean, 0.0)
        result = {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(var),
            "abs_mean": abs_mean,
            "abs_std": math.sqrt(abs_var),
            "max_abs": self.max_abs,
            "max_abs_over_abs_mean": self.max_abs / abs_mean if abs_mean else 0.0,
        }
        for threshold in ACTIVATION_THRESHOLDS:
            result[f"ratio_abs_gt_{format_threshold(threshold)}"] = (
                self.gt_counts[threshold] / self.count
            )
        return result


def format_threshold(value: float) -> str:
    return str(value).replace(".", "p")


def quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().reshape(-1), q).item())


def top_fraction_share(abs_values: torch.Tensor, fraction: float) -> float:
    flat = abs_values.float().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    k = max(1, int(math.ceil(flat.numel() * fraction)))
    total = float(flat.sum().item())
    if total == 0.0:
        return 0.0
    return float(torch.topk(flat, k).values.sum().item()) / total


def parse_weight_key(key: str) -> Tuple[int, str, str] | None:
    prefix = "encoder.encoder.layer."
    if not key.startswith(prefix) or not key.endswith(".weight"):
        return None
    rest = key[len(prefix) : -len(".weight")]
    layer_text, module_name = rest.split(".", 1)
    if module_name not in LINEAR_MODULES:
        return None
    return int(layer_text), module_name, LINEAR_MODULES[module_name]


def collect_weight_stats(task: str, run_name: str, checkpoint_path: Path) -> List[Dict[str, object]]:
    state = torch.load(checkpoint_path, map_location="cpu")
    rows: List[Dict[str, object]] = []
    for key, tensor in state.items():
        parsed = parse_weight_key(key)
        if parsed is None:
            continue
        layer, module_name, module_short = parsed
        w = tensor.detach().float()
        aw = w.abs()
        std = float(w.std(unbiased=False).item())
        mean = float(w.mean().item())
        abs_mean = float(aw.mean().item())
        row: Dict[str, object] = {
            "task": task,
            "run_name": run_name,
            "layer": layer,
            "module": module_short,
            "module_name": module_name,
            "shape": "x".join(str(x) for x in w.shape),
            "numel": int(w.numel()),
            "mean": mean,
            "std": std,
            "abs_mean": abs_mean,
            "max_abs": float(aw.max().item()),
            "p99_abs": quantile(aw, 0.99),
            "p999_abs": quantile(aw, 0.999),
            "p9999_abs": quantile(aw, 0.9999),
            "top_0p1pct_abs_share": top_fraction_share(aw, 0.001),
            "top_1pct_abs_share": top_fraction_share(aw, 0.01),
            "max_abs_over_abs_mean": float(aw.max().item()) / abs_mean if abs_mean else 0.0,
            "p999_abs_over_p99_abs": 0.0,
        }
        if row["p99_abs"]:
            row["p999_abs_over_p99_abs"] = float(row["p999_abs"]) / float(row["p99_abs"])
        for sigma in WEIGHT_SIGMA_THRESHOLDS:
            name = format_threshold(sigma)
            row[f"ratio_abs_gt_{name}sigma"] = float((aw > sigma * std).sum().item()) / w.numel() if std else 0.0

        out_channel_max = aw.reshape(w.shape[0], -1).max(dim=1).values
        in_channel_max = aw.transpose(0, 1).reshape(w.shape[1], -1).max(dim=1).values
        row.update(
            {
                "top_out_channel": int(out_channel_max.argmax().item()),
                "top_out_channel_max_abs": float(out_channel_max.max().item()),
                "median_out_channel_max_abs": float(out_channel_max.median().item()),
                "top_in_channel": int(in_channel_max.argmax().item()),
                "top_in_channel_max_abs": float(in_channel_max.max().item()),
                "median_in_channel_max_abs": float(in_channel_max.median().item()),
            }
        )
        med_out = float(row["median_out_channel_max_abs"])
        med_in = float(row["median_in_channel_max_abs"])
        row["top_out_channel_over_median"] = float(row["top_out_channel_max_abs"]) / med_out if med_out else 0.0
        row["top_in_channel_over_median"] = float(row["top_in_channel_max_abs"]) / med_in if med_in else 0.0
        rows.append(row)
    return sorted(rows, key=lambda x: (int(x["layer"]), str(x["module"])))


def build_bert_from_checkpoint(project_root: Path, checkpoint_path: Path, device: torch.device) -> BertModel:
    config_path = project_root / "base_exp" / "exp_edge" / "models" / "bert-large-uncased" / "model" / "config.json"
    config = BertConfig.from_json_file(str(config_path))
    model = BertModel(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    bert_state = {}
    for key, value in state.items():
        if key.startswith("encoder."):
            bert_state[key[len("encoder.") :]] = value
    missing, unexpected = model.load_state_dict(bert_state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("cls.")]
    if unexpected:
        raise RuntimeError(f"Unexpected BERT keys while loading {checkpoint_path}: {unexpected[:8]}")
    non_head_missing = [key for key in missing if not key.startswith("pooler.")]
    if non_head_missing:
        raise RuntimeError(f"Missing BERT keys while loading {checkpoint_path}: {non_head_missing[:8]}")
    model.to(device)
    model.eval()
    return model


def iter_cached_batches(project_root: Path, task: str, max_samples: int, batch_size: int) -> Iterable[Dict[str, torch.Tensor]]:
    cache_dir = project_root / "base_exp" / "exp_edge" / "cache" / "bert-large-uncased" / task / "val"
    chunk_paths = sorted(cache_dir.glob("data_*.chunk"))
    emitted = 0
    batch = []
    for chunk_path in chunk_paths:
        chunk = torch.load(chunk_path, map_location="cpu")
        for item in chunk:
            row = item["data_row"]
            batch.append(row)
            if len(batch) == batch_size:
                yield batch_to_tensors(batch)
                emitted += len(batch)
                batch = []
                if emitted >= max_samples:
                    return
    if batch and emitted < max_samples:
        yield batch_to_tensors(batch)


def batch_to_tensors(rows) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor([row.input_ids for row in rows], dtype=torch.long)
    attention_mask = torch.tensor([row.input_mask for row in rows], dtype=torch.long)
    token_type_ids = torch.tensor([row.segment_ids for row in rows], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def collect_activation_stats(
    project_root: Path,
    task: str,
    run_name: str,
    checkpoint_path: Path,
    max_samples: int,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    model = build_bert_from_checkpoint(project_root, checkpoint_path, device)
    stats: Dict[Tuple[int, str, str], RunningAbsStats] = {}
    hooks = []

    for layer_idx, layer in enumerate(model.encoder.layer):
        modules = {
            "attn_q_input": layer.attention.self.query,
            "attn_k_input": layer.attention.self.key,
            "attn_v_input": layer.attention.self.value,
            "attn_out_input": layer.attention.output.dense,
            "ffn_in_input": layer.intermediate.dense,
            "ffn_out_input": layer.output.dense,
        }
        for name, module in modules.items():
            key = (layer_idx, name, "linear_input")
            stats[key] = RunningAbsStats()

            def make_pre_hook(stat_key):
                def hook(_module, inputs):
                    stats[stat_key].update(inputs[0])

                return hook

            hooks.append(module.register_forward_pre_hook(make_pre_hook(key)))

        gelu_key = (layer_idx, "gelu_input", "intermediate_dense_output")
        stats[gelu_key] = RunningAbsStats()

        def make_forward_hook(stat_key):
            def hook(_module, _inputs, output):
                stats[stat_key].update(output)

            return hook

        hooks.append(layer.intermediate.dense.register_forward_hook(make_forward_hook(gelu_key)))

    samples = 0
    with torch.no_grad():
        for batch in iter_cached_batches(project_root, task, max_samples, batch_size):
            batch = {key: value.to(device) for key, value in batch.items()}
            samples += int(batch["input_ids"].shape[0])
            model(**batch)

    for hook in hooks:
        hook.remove()

    rows: List[Dict[str, object]] = []
    for (layer_idx, module, capture_point), stat in sorted(stats.items()):
        row: Dict[str, object] = {
            "task": task,
            "run_name": run_name,
            "layer": layer_idx,
            "module": module,
            "capture_point": capture_point,
            "sample_count": samples,
        }
        row.update(stat.finalize())
        rows.append(row)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def summarize_layers(weight_rows: List[Dict[str, object]], activation_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in weight_rows:
        key = (str(row["task"]), str(row["run_name"]), int(row["layer"]))
        item = summary.setdefault(
            key,
            {
                "task": key[0],
                "run_name": key[1],
                "layer": key[2],
                "weight_module_count": 0,
                "weight_max_abs_max": 0.0,
                "weight_p999_abs_mean": 0.0,
                "weight_top_0p1pct_abs_share_mean": 0.0,
                "activation_module_count": 0,
                "activation_max_abs_max": 0.0,
                "activation_ratio_abs_gt_6p0_mean": 0.0,
                "activation_ratio_abs_gt_8p0_mean": 0.0,
                "activation_max_abs_over_abs_mean_mean": 0.0,
            },
        )
        item["weight_module_count"] += 1
        item["weight_max_abs_max"] = max(float(item["weight_max_abs_max"]), float(row["max_abs"]))
        item["weight_p999_abs_mean"] += float(row["p999_abs"])
        item["weight_top_0p1pct_abs_share_mean"] += float(row["top_0p1pct_abs_share"])
    for row in activation_rows:
        key = (str(row["task"]), str(row["run_name"]), int(row["layer"]))
        item = summary.setdefault(
            key,
            {
                "task": key[0],
                "run_name": key[1],
                "layer": key[2],
                "weight_module_count": 0,
                "weight_max_abs_max": 0.0,
                "weight_p999_abs_mean": 0.0,
                "weight_top_0p1pct_abs_share_mean": 0.0,
                "activation_module_count": 0,
                "activation_max_abs_max": 0.0,
                "activation_ratio_abs_gt_6p0_mean": 0.0,
                "activation_ratio_abs_gt_8p0_mean": 0.0,
                "activation_max_abs_over_abs_mean_mean": 0.0,
            },
        )
        item["activation_module_count"] += 1
        item["activation_max_abs_max"] = max(float(item["activation_max_abs_max"]), float(row["max_abs"]))
        item["activation_ratio_abs_gt_6p0_mean"] += float(row.get("ratio_abs_gt_6p0", 0.0))
        item["activation_ratio_abs_gt_8p0_mean"] += float(row.get("ratio_abs_gt_8p0", 0.0))
        item["activation_max_abs_over_abs_mean_mean"] += float(row.get("max_abs_over_abs_mean", 0.0))

    rows = []
    for item in summary.values():
        if item["weight_module_count"]:
            item["weight_p999_abs_mean"] /= int(item["weight_module_count"])
            item["weight_top_0p1pct_abs_share_mean"] /= int(item["weight_module_count"])
        if item["activation_module_count"]:
            item["activation_ratio_abs_gt_6p0_mean"] /= int(item["activation_module_count"])
            item["activation_ratio_abs_gt_8p0_mean"] /= int(item["activation_module_count"])
            item["activation_max_abs_over_abs_mean_mean"] /= int(item["activation_module_count"])
        rows.append(item)
    return sorted(rows, key=lambda x: (str(x["task"]), int(x["layer"])))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def make_markdown(metadata: Dict[str, object], layer_rows: List[Dict[str, object]], activation_rows: List[Dict[str, object]]) -> str:
    lines = [
        "# Outlier statistics: NER / DEP / SemEval baselines",
        "",
        "This file is generated by `tools/collect_outlier_stats.py`.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Layer summary",
        "",
    ]

    for task in sorted({str(row["task"]) for row in layer_rows}):
        task_rows = [row for row in layer_rows if row["task"] == task]
        top_act = sorted(task_rows, key=lambda x: float(x.get("activation_ratio_abs_gt_6p0_mean", 0.0)), reverse=True)[:8]
        lines.append(f"### {task}: top layers by mean activation ratio |x| > 6")
        lines.append("")
        lines.append(
            markdown_table(
                ["layer", "act_gt6_mean", "act_gt8_mean", "act_max_abs", "weight_p999_mean", "weight_top0.1_share"],
                [
                    [
                        row["layer"],
                        f"{float(row.get('activation_ratio_abs_gt_6p0_mean', 0.0)):.6g}",
                        f"{float(row.get('activation_ratio_abs_gt_8p0_mean', 0.0)):.6g}",
                        f"{float(row.get('activation_max_abs_max', 0.0)):.4g}",
                        f"{float(row.get('weight_p999_abs_mean', 0.0)):.4g}",
                        f"{float(row.get('weight_top_0p1pct_abs_share_mean', 0.0)):.4g}",
                    ]
                    for row in top_act
                ],
            )
        )
        lines.append("")

    if activation_rows:
        top_modules = sorted(activation_rows, key=lambda x: float(x.get("ratio_abs_gt_6p0", 0.0)), reverse=True)[:20]
        lines.extend(["## Top activation module outliers", ""])
        lines.append(
            markdown_table(
                ["task", "layer", "module", "capture", "gt6", "gt8", "max_abs", "max/abs_mean"],
                [
                    [
                        row["task"],
                        row["layer"],
                        row["module"],
                        row["capture_point"],
                        f"{float(row.get('ratio_abs_gt_6p0', 0.0)):.6g}",
                        f"{float(row.get('ratio_abs_gt_8p0', 0.0)):.6g}",
                        f"{float(row.get('max_abs', 0.0)):.4g}",
                        f"{float(row.get('max_abs_over_abs_mean', 0.0)):.4g}",
                    ]
                    for row in top_modules
                ],
            )
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/cluster/home/zhangzx/my_project")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-activations", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "base_exp" / "exp_edge" / "analysis" / "outlier_stats" / "ner_dep_semeval_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    weight_rows: List[Dict[str, object]] = []
    activation_rows: List[Dict[str, object]] = []
    run_metadata = []

    for run in RUNS:
        task = run["task"]
        run_name = run["run_name"]
        checkpoint_path = project_root / "base_exp" / "exp_edge" / "runs" / "bert-large-uncased" / task / run_name / "best_model.p"
        val_metrics_path = checkpoint_path.parent / "val_metrics.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        run_metadata.append(
            {
                **run,
                "checkpoint_path": str(checkpoint_path),
                "val_metrics_path": str(val_metrics_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            }
        )
        weight_rows.extend(collect_weight_stats(task, run_name, checkpoint_path))
        if not args.skip_activations:
            activation_rows.extend(
                collect_activation_stats(
                    project_root=project_root,
                    task=task,
                    run_name=run_name,
                    checkpoint_path=checkpoint_path,
                    max_samples=args.max_val_samples,
                    batch_size=args.batch_size,
                    device=device,
                )
            )

    layer_rows = summarize_layers(weight_rows, activation_rows)
    metadata = {
        "runs": run_metadata,
        "max_val_samples": args.max_val_samples,
        "batch_size": args.batch_size,
        "device": str(device),
        "activation_thresholds_abs": ACTIVATION_THRESHOLDS,
        "weight_sigma_thresholds": WEIGHT_SIGMA_THRESHOLDS,
        "linear_modules": LINEAR_MODULES,
        "notes": [
            "Activation stats use cached validation inputs and BERT encoder weights from task fine-tuned best_model.p.",
            "Activation captures include linear inputs for Q/K/V, attention output, FFN input/output, plus intermediate dense output as GELU input.",
            "NER requested name formal_e3_ms256_retry1 is mapped to existing project run formal_e3_ms256_tner.",
        ],
    }

    write_csv(output_dir / "weight_module_stats.csv", weight_rows)
    write_csv(output_dir / "activation_module_stats.csv", activation_rows)
    write_csv(output_dir / "layer_summary.csv", layer_rows)
    (output_dir / "outlier_stats.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "layer_summary": layer_rows,
                "weight_module_stats": weight_rows,
                "activation_module_stats": activation_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "README_outlier_stats.md").write_text(make_markdown(metadata, layer_rows, activation_rows), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
