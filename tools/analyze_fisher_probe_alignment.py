#!/usr/bin/env python3
"""Compare fixed-probing and empirical-Fisher layer rankings."""

import argparse
import csv
import json
import math
from pathlib import Path


def moving_average(values):
    return [
        sum(values[max(0, index - 1) : min(len(values), index + 2)])
        / len(values[max(0, index - 1) : min(len(values), index + 2)])
        for index in range(len(values))
    ]


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    for rank, index in enumerate(order, 1):
        result[index] = float(rank)
    return result


def pearson(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probing-csv", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe_rows = list(csv.DictReader(args.probing_csv.open(encoding="utf-8")))
    rows = []
    for task in ("semeval", "ner", "dep"):
        probing = {
            int(row["layer"]): float(row["major"])
            for row in probe_rows
            if row["task"] == task
        }
        probing = moving_average([probing[layer] for layer in range(1, 25)])
        fisher = json.loads(
            (
                args.analysis_dir
                / f"{task}_fisher_w4a4_seed20260806_cal16.json"
            ).read_text(encoding="utf-8")
        )["layers"]
        weight = [row["weight_score"] for row in fisher]
        activation = [row["activation_score"] for row in fisher]
        rows.append(
            {
                "task": task,
                "spearman_probe_fisher_weight": pearson(ranks(probing), ranks(weight)),
                "spearman_probe_fisher_activation": pearson(
                    ranks(probing), ranks(activation)
                ),
                "probe_top_layers": sorted(
                    range(24), key=lambda index: probing[index], reverse=True
                )[:5],
                "fisher_weight_top_layers": sorted(
                    range(24), key=lambda index: weight[index], reverse=True
                )[:5],
                "fisher_activation_top_layers": sorted(
                    range(24), key=lambda index: activation[index], reverse=True
                )[:5],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
