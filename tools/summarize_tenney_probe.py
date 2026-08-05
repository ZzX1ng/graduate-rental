#!/usr/bin/env python3
"""Summarize Tenney-style scalar-mix and cumulative probing runs."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch


DEFAULT_TASK = "semeval"
DEFAULT_MODEL = "bert-large-uncased"
DEFAULT_FULL_RUN = "tenney_scalar_mix_ms256_semeval_lr1e4_e10_noearly"
DEFAULT_CUMULATIVE_PATTERN = (
    "tenney_cumulative_l{layer:02d}_ms256_semeval_lr1e4_e10_noearly"
)


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_metrics(runs_root, task, run_name):
    path = runs_root / task / run_name / "val_metrics.json"
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = read_json(path)
    task_metrics = data.get(task, {}).get("metrics", {})
    minor = task_metrics.get("minor", {})
    return {
        "run_name": run_name,
        "metrics_path": str(path),
        "major": float(task_metrics.get("major", data.get("aggregated"))),
        "aggregated": float(data.get("aggregated", task_metrics.get("major"))),
        "f1_micro": float(minor.get("f1_micro", 0.0)),
        "acc": float(minor.get("acc", 0.0)),
        "loss": float(data.get(task, {}).get("loss", 0.0)),
    }


def find_single_key(state, suffix):
    keys = [key for key in state if key.endswith(suffix)]
    if len(keys) != 1:
        raise KeyError("Expected one key ending with %s, got %s" % (suffix, keys))
    return keys[0]


def read_scalar_mix(runs_root, task, run_name):
    path = runs_root / task / run_name / "best_model.p"
    if not path.exists():
        raise FileNotFoundError(str(path))
    state = torch.load(str(path), map_location="cpu")
    logits_key = find_single_key(state, "tenney_scalar_logits")
    gamma_key = find_single_key(state, "tenney_scalar_gamma")
    logits = state[logits_key].float()
    weights = torch.softmax(logits, dim=0)
    return {
        "checkpoint_path": str(path),
        "logits_key": logits_key,
        "gamma_key": gamma_key,
        "gamma": float(state[gamma_key].float().reshape(-1)[0].item()),
        "logits": [float(x) for x in logits.tolist()],
        "weights": [float(x) for x in weights.tolist()],
        "center_of_gravity": sum(i * float(w) for i, w in enumerate(weights.tolist())),
        "entropy": -sum(float(w) * math.log(float(w) + 1e-12) for w in weights.tolist()),
    }


def normalized(values):
    denom = sum(values)
    if abs(denom) < 1e-12:
        return [0.0 for _ in values]
    return [float(v) / denom for v in values]


def positive_normalized(values):
    positives = [max(0.0, float(v)) for v in values]
    return normalized(positives)


def expected_layer(deltas):
    denom = sum(deltas)
    if abs(denom) < 1e-12:
        return None
    return sum(layer * delta for layer, delta in enumerate(deltas)) / denom


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/cluster/home/zhangzx/my_project")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--full-run", default=DEFAULT_FULL_RUN)
    parser.add_argument("--cumulative-pattern", default=DEFAULT_CUMULATIVE_PATTERN)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    runs_root = project_root / "base_exp" / "exp_edge" / "runs" / args.model
    output_dir = Path(args.output_dir) if args.output_dir else (
        project_root
        / "base_exp"
        / "exp_edge"
        / "analysis"
        / "tenney_probe"
        / "semeval_e10_lr1e4_noearly"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    full_metrics = read_metrics(runs_root, args.task, args.full_run)
    scalar_mix = read_scalar_mix(runs_root, args.task, args.full_run)

    cumulative = []
    for layer in range(25):
        run_name = args.cumulative_pattern.format(layer=layer)
        metrics = read_metrics(runs_root, args.task, run_name)
        metrics["layer"] = layer
        cumulative.append(metrics)

    cumulative_scores = [row["major"] for row in cumulative]
    differential = []
    for layer, score in enumerate(cumulative_scores):
        if layer == 0:
            differential.append(score)
        else:
            differential.append(score - cumulative_scores[layer - 1])
    contextual_differential = [0.0] + [
        cumulative_scores[layer] - cumulative_scores[layer - 1] for layer in range(1, 25)
    ]
    norm_differential_all = normalized(differential)
    norm_differential_contextual = normalized(contextual_differential)
    norm_differential_positive = positive_normalized(contextual_differential)

    figure2_rows = []
    for layer in range(25):
        figure2_rows.append(
            {
                "task": args.task,
                "layer": layer,
                "scalar_mix_weight": scalar_mix["weights"][layer],
                "scalar_mix_logit": scalar_mix["logits"][layer],
                "cumulative_major": cumulative_scores[layer],
                "cumulative_f1_micro": cumulative[layer]["f1_micro"],
                "cumulative_acc": cumulative[layer]["acc"],
                "differential_from_previous": differential[layer],
                "contextual_differential_from_previous": contextual_differential[layer],
                "normalized_differential_all": norm_differential_all[layer],
                "normalized_differential_contextual": norm_differential_contextual[layer],
                "positive_normalized_differential": norm_differential_positive[layer],
                "cumulative_run_name": cumulative[layer]["run_name"],
            }
        )

    figure1_summary = {
        "task": args.task,
        "full_scalar_mix_run": args.full_run,
        "p0_cumulative_major": cumulative_scores[0],
        "pL_cumulative_major": cumulative_scores[-1],
        "full_scalar_mix_major": full_metrics["major"],
        "full_scalar_mix_f1_micro": full_metrics["f1_micro"],
        "full_scalar_mix_acc": full_metrics["acc"],
        "scalar_mix_center_of_gravity": scalar_mix["center_of_gravity"],
        "scalar_mix_entropy": scalar_mix["entropy"],
        "scalar_mix_gamma": scalar_mix["gamma"],
        "cumulative_expected_layer_contextual_raw": expected_layer(contextual_differential),
        "cumulative_expected_layer_contextual_positive": expected_layer(
            [max(0.0, x) for x in contextual_differential]
        ),
        "best_cumulative_layer": max(range(25), key=lambda layer: cumulative_scores[layer]),
        "best_cumulative_major": max(cumulative_scores),
    }

    output = {
        "metadata": {
            "task": args.task,
            "model": args.model,
            "project_root": str(project_root),
            "full_run": args.full_run,
            "cumulative_pattern": args.cumulative_pattern,
            "notes": [
                "Figure 1 fields include P0, full scalar-mix score, scalar mixing center of gravity, and cumulative expected layer.",
                "Figure 2 fields include full scalar-mix weights and cumulative differential scores for layers 0-24.",
                "Layer 0 denotes embedding output; layers 1-24 denote BERT-large Transformer layer outputs.",
            ],
        },
        "figure1_summary": figure1_summary,
        "scalar_mix": scalar_mix,
        "full_metrics": full_metrics,
        "figure2_layerwise": figure2_rows,
    }

    (output_dir / "tenney_figure_data.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "tenney_figure1_summary.csv", [figure1_summary])
    write_csv(output_dir / "tenney_figure2_layerwise.csv", figure2_rows)

    top_scalar = sorted(figure2_rows, key=lambda row: row["scalar_mix_weight"], reverse=True)[:8]
    top_diff = sorted(
        figure2_rows, key=lambda row: row["positive_normalized_differential"], reverse=True
    )[:8]
    lines = [
        "# Tenney-style %s probing summary" % args.task,
        "",
        "Generated by `tools/summarize_tenney_probe.py`.",
        "",
        "## Figure 1 fields",
        "",
        markdown_table(
            [
                "task",
                "P0 cumulative",
                "PL cumulative",
                "full scalar major",
                "scalar COG",
                "scalar entropy",
                "cumulative expected",
            ],
            [
                [
                    figure1_summary["task"],
                    "%.6f" % figure1_summary["p0_cumulative_major"],
                    "%.6f" % figure1_summary["pL_cumulative_major"],
                    "%.6f" % figure1_summary["full_scalar_mix_major"],
                    "%.4f" % figure1_summary["scalar_mix_center_of_gravity"],
                    "%.4f" % figure1_summary["scalar_mix_entropy"],
                    "%.4f" % figure1_summary["cumulative_expected_layer_contextual_positive"],
                ]
            ],
        ),
        "",
        "## Top scalar-mix weights",
        "",
        markdown_table(
            ["layer", "weight", "cumulative_major", "positive_norm_delta"],
            [
                [
                    row["layer"],
                    "%.6f" % row["scalar_mix_weight"],
                    "%.6f" % row["cumulative_major"],
                    "%.6f" % row["positive_normalized_differential"],
                ]
                for row in top_scalar
            ],
        ),
        "",
        "## Top cumulative differential layers",
        "",
        markdown_table(
            ["layer", "positive_norm_delta", "raw_delta", "cumulative_major", "scalar_weight"],
            [
                [
                    row["layer"],
                    "%.6f" % row["positive_normalized_differential"],
                    "%.6f" % row["contextual_differential_from_previous"],
                    "%.6f" % row["cumulative_major"],
                    "%.6f" % row["scalar_mix_weight"],
                ]
                for row in top_diff
            ],
        ),
        "",
    ]
    (output_dir / "README_tenney_probe.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote Tenney-style probing summary to %s" % output_dir)


if __name__ == "__main__":
    main()
