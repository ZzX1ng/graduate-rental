#!/usr/bin/env python3
"""Summarize representative-layer perturbations and Fisher rank correlation."""

import argparse
import json
from pathlib import Path


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def pearson(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = (
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def metric(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["semeval"]["metrics"]
    return metrics["major"], metrics["minor"]["f1_micro"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fisher-stats", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    fisher = json.loads(args.fisher_stats.read_text(encoding="utf-8"))
    by_layer = {row["layer"]: row for row in fisher["layers"]}
    baseline_major, baseline_f1 = metric(
        args.run_root
        / "ptq_fisher_stage1_control_w32a32_seed20260806_semeval_e10"
        / "val_metrics.json"
    )
    rows = []
    for layer in (0, 3, 7, 11, 15, 19, 23):
        for side in ("weight", "activation"):
            run_name = f"ptq_fisher_stage1_{side}_layer{layer:02d}_seed20260806_semeval_e10"
            major, f1 = metric(args.run_root / run_name / "val_metrics.json")
            rows.append(
                {
                    "layer": layer,
                    "side": side,
                    "fisher_score": by_layer[layer][f"{side}_score"],
                    "major": major,
                    "f1_micro": f1,
                    "major_drop": baseline_major - major,
                    "f1_drop": baseline_f1 - f1,
                    "run_name": run_name,
                }
            )
    correlations = {}
    for side in ("weight", "activation"):
        selected = [row for row in rows if row["side"] == side]
        correlations[side] = {
            "spearman_fisher_vs_major_drop": pearson(
                ranks([row["fisher_score"] for row in selected]),
                ranks([row["major_drop"] for row in selected]),
            ),
            "spearman_fisher_vs_f1_drop": pearson(
                ranks([row["fisher_score"] for row in selected]),
                ranks([row["f1_drop"] for row in selected]),
            ),
        }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with args.output_prefix.with_suffix(".tsv").open("w", encoding="utf-8") as writer:
        writer.write("\t".join(columns) + "\n")
        for row in rows:
            writer.write("\t".join(str(row[column]) for column in columns) + "\n")
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "baseline_major": baseline_major,
                "baseline_f1_micro": baseline_f1,
                "rows": rows,
                "correlations": correlations,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
