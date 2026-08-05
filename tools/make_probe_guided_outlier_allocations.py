#!/usr/bin/env python3
"""Build BOP-matched layer-wise outlier allocations from fixed probing scores."""

import argparse
import csv
import json
import math
import random
from pathlib import Path


def moving_average_three(values):
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - 1)
        end = min(len(values), index + 2)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def normalize_mean(values):
    mean = sum(values) / len(values)
    return [value / mean for value in values]


def exp_shape(values, strength=1.0):
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    centered = [2.0 * (value - low) / (high - low) - 1.0 for value in values]
    return normalize_mean([math.exp(strength * value) for value in centered])


def random_shape(seed, layer_count):
    generator = random.Random(seed)
    return normalize_mean([math.exp(generator.gauss(0.0, 0.65)) for _ in range(layer_count)])


def match_bop(shape, base_weight_ratio, base_activation_ratio):
    """Scale both ratio vectors so mean(w + a + w*a) equals uniform BOP."""
    target = (
        base_weight_ratio
        + base_activation_ratio
        + base_weight_ratio * base_activation_ratio
    )
    mean_shape = sum(shape) / len(shape)
    mean_shape_sq = sum(value * value for value in shape) / len(shape)
    linear = (base_weight_ratio + base_activation_ratio) * mean_shape
    quadratic = base_weight_ratio * base_activation_ratio * mean_shape_sq
    if quadratic == 0:
        scale = target / linear
    else:
        scale = (-linear + math.sqrt(linear * linear + 4.0 * quadratic * target)) / (
            2.0 * quadratic
        )
    weights = [scale * base_weight_ratio * value for value in shape]
    activations = [scale * base_activation_ratio * value for value in shape]
    actual = sum(
        weight + activation + weight * activation
        for weight, activation in zip(weights, activations)
    ) / len(shape)
    return weights, activations, actual


def load_fixed_probe(csv_path, task):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as reader:
        for row in csv.DictReader(reader):
            if row["task"] == task:
                rows.append((int(row["layer"]), float(row["major"])))
    rows.sort()
    by_layer = dict(rows)
    # Encoder block l produces representation L(l+1); L0 is the embedding output.
    return [by_layer[layer] for layer in range(1, 25)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probing-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="semeval")
    args = parser.parse_args()

    raw_scores = load_fixed_probe(args.probing_csv, args.task)
    smoothed_scores = moving_average_three(raw_scores)
    layer_count = len(smoothed_scores)
    depth = [1.0 - 2.0 * index / (layer_count - 1) for index in range(layer_count)]
    shapes = {
        "uniform": [1.0] * layer_count,
        "probing": exp_shape(smoothed_scores),
        "inverse": exp_shape([-value for value in smoothed_scores]),
        "early": normalize_mean([math.exp(value) for value in depth]),
        "late": normalize_mean([math.exp(-value) for value in depth]),
        "random_s17": random_shape(17, layer_count),
        "random_s29": random_shape(29, layer_count),
        "random_s43": random_shape(43, layer_count),
    }
    budgets = {
        "low": (0.001, 0.005),
        "boundary": (0.0025, 0.01),
    }
    allocations = []
    for budget_name, (weight_ratio, activation_ratio) in budgets.items():
        target = weight_ratio + activation_ratio + weight_ratio * activation_ratio
        for strategy, shape in shapes.items():
            weights, activations, nominal_bop = match_bop(
                shape, weight_ratio, activation_ratio
            )
            allocations.append(
                {
                    "budget": budget_name,
                    "strategy": strategy,
                    "uniform_weight_ratio": weight_ratio,
                    "uniform_activation_ratio": activation_ratio,
                    "target_nominal_bop_overhead": target,
                    "nominal_bop_overhead": nominal_bop,
                    "weight_layer_ratios": weights,
                    "activation_layer_ratios": activations,
                }
            )

    payload = {
        "task": args.task,
        "mapping": "encoder block l uses fixed-probe representation L(l+1)",
        "probe_preprocessing": "three-point moving average, then bounded exponential allocation",
        "bop_model": "mean_l(w_l + a_l + w_l*a_l) for W4A4 with outlier8",
        "raw_probe_scores_l1_l24": raw_scores,
        "smoothed_probe_scores_l1_l24": smoothed_scores,
        "allocations": allocations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
