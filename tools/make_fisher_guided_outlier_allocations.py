#!/usr/bin/env python3
"""Build Fisher and probing+Fisher layer-wise outlier allocations."""

import argparse
import csv
import json
import math
from pathlib import Path


def normalize_mean(values):
    mean = sum(values) / len(values)
    return [value / mean for value in values]


def moving_average_three(values):
    return [
        sum(values[max(0, index - 1) : min(len(values), index + 2)])
        / len(values[max(0, index - 1) : min(len(values), index + 2)])
        for index in range(len(values))
    ]


def minmax_z(values):
    logs = [math.log(max(value, 1e-30)) for value in values]
    low, high = min(logs), max(logs)
    if high == low:
        return [0.0] * len(logs)
    return [2.0 * (value - low) / (high - low) - 1.0 for value in logs]


def probe_z(csv_path, task):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as reader:
        for row in csv.DictReader(reader):
            if row["task"] == task:
                rows.append((int(row["layer"]), float(row["major"])))
    scores = dict(rows)
    smoothed = moving_average_three([scores[layer] for layer in range(1, 25)])
    low, high = min(smoothed), max(smoothed)
    return [2.0 * (value - low) / (high - low) - 1.0 for value in smoothed]


def shape_from_z(values):
    return normalize_mean([math.exp(value) for value in values])


def match_bop(
    weight_shape,
    activation_shape,
    base_weight,
    base_activation,
    floor_fraction=0.0,
):
    target = base_weight + base_activation + base_weight * base_activation
    cross = base_weight * base_activation
    linear = sum(
        base_weight * weight
        + base_activation * activation
        + cross * floor_fraction * (weight + activation)
        for weight, activation in zip(weight_shape, activation_shape)
    ) / len(weight_shape)
    quadratic = cross * sum(
        weight * activation
        for weight, activation in zip(weight_shape, activation_shape)
    ) / len(weight_shape)
    constant = (
        (base_weight + base_activation) * floor_fraction
        + cross * floor_fraction * floor_fraction
        - target
    )
    if quadratic == 0:
        scale = -constant / linear
    else:
        scale = (-linear + math.sqrt(linear * linear - 4 * quadratic * constant)) / (
            2 * quadratic
        )
    weight_factors = [floor_fraction + scale * value for value in weight_shape]
    activation_factors = [
        floor_fraction + scale * value for value in activation_shape
    ]
    weights = [base_weight * value for value in weight_factors]
    activations = [base_activation * value for value in activation_factors]
    actual = sum(
        weight + activation + weight * activation
        for weight, activation in zip(weights, activations)
    ) / len(weights)
    return weights, activations, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fisher-stats", type=Path, required=True)
    parser.add_argument("--probing-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="semeval")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--low-weight-ratio", type=float, default=0.001)
    parser.add_argument("--low-activation-ratio", type=float, default=0.005)
    parser.add_argument("--boundary-weight-ratio", type=float, default=0.0025)
    parser.add_argument("--boundary-activation-ratio", type=float, default=0.01)
    parser.add_argument("--shape-floor", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.shape_floor < 1.0:
        parser.error("--shape-floor must be in [0, 1)")

    fisher = json.loads(args.fisher_stats.read_text(encoding="utf-8"))
    layers = sorted(fisher["layers"], key=lambda row: row["layer"])
    fisher_weight_z = minmax_z([row["weight_score"] for row in layers])
    fisher_activation_z = minmax_z([row["activation_score"] for row in layers])
    probing_z = probe_z(args.probing_csv, args.task)
    shapes = {
        "fisher": (shape_from_z(fisher_weight_z), shape_from_z(fisher_activation_z)),
        "probefisher": (
            shape_from_z(
                [
                    args.beta * probe + (1 - args.beta) * fisher_score
                    for probe, fisher_score in zip(probing_z, fisher_weight_z)
                ]
            ),
            shape_from_z(
                [
                    args.beta * probe + (1 - args.beta) * fisher_score
                    for probe, fisher_score in zip(probing_z, fisher_activation_z)
                ]
            ),
        ),
    }
    budgets = {
        "low": (args.low_weight_ratio, args.low_activation_ratio),
        "boundary": (
            args.boundary_weight_ratio,
            args.boundary_activation_ratio,
        ),
    }
    allocations = []
    for budget, (base_weight, base_activation) in budgets.items():
        for strategy, (weight_shape, activation_shape) in shapes.items():
            weights, activations, bop = match_bop(
                weight_shape,
                activation_shape,
                base_weight,
                base_activation,
                floor_fraction=args.shape_floor,
            )
            allocations.append(
                {
                    "budget": budget,
                    "strategy": strategy,
                    "uniform_weight_ratio": base_weight,
                    "uniform_activation_ratio": base_activation,
                    "nominal_bop_overhead": bop,
                    "weight_layer_ratios": weights,
                    "activation_layer_ratios": activations,
                }
            )
    payload = {
        "task": args.task,
        "method": "separate Fisher W/A shapes; beta=0.5 z-score fusion for probefisher",
        "beta": args.beta,
        "shape_floor_fraction": args.shape_floor,
        "fisher_weight_z": fisher_weight_z,
        "fisher_activation_z": fisher_activation_z,
        "probing_z": probing_z,
        "allocations": allocations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
