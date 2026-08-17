#!/usr/bin/env python3
"""Match measured uniform BOP while preserving a fixed allocation floor."""

import argparse
import json
import re
from pathlib import Path


LAYER_PATTERN = re.compile(r"encoder\.layer\.(\d+)\.")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def predicted_bop(modules, weights, activations, floor_weights, floor_activations, scale):
    low_bit_macs = 0
    adjusted_macs = 0.0
    for module in modules:
        match = LAYER_PATTERN.search(module["module_name"])
        if not match:
            raise ValueError(f"Cannot parse layer from {module['module_name']}")
        layer = int(match.group(1))
        weight_ratio = floor_weights[layer] + scale * (
            weights[layer] - floor_weights[layer]
        )
        activation_ratio = floor_activations[layer] + scale * (
            activations[layer] - floor_activations[layer]
        )
        weight_factor = weight_ratio / weights[layer]
        activation_factor = activation_ratio / activations[layer]
        low_bit_macs += module["low_bit_macs"]
        adjusted_macs += (
            weight_factor * module["weight_outlier_macs"]
            + activation_factor * module["activation_outlier_macs"]
            + weight_factor * activation_factor * module["dual_outlier_macs"]
        )
    return adjusted_macs / low_bit_macs


def solve_scale(modules, weights, activations, floor_weights, floor_activations, target):
    low, high = 0.0, 1.0
    while predicted_bop(
        modules, weights, activations, floor_weights, floor_activations, high
    ) < target:
        high *= 2.0
        if high > 64.0:
            raise RuntimeError("Unable to bracket floor-preserving BOP scale")
    for _ in range(80):
        middle = (low + high) / 2.0
        value = predicted_bop(
            modules, weights, activations, floor_weights, floor_activations, middle
        )
        if value < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocations", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor-fraction", type=float, required=True)
    args = parser.parse_args()

    payload = load_json(args.allocations)
    uniform_targets = {}
    for item in payload["allocations"]:
        if item["strategy"] != "uniform":
            continue
        summary = load_json(args.run_root / item["source_run"] / "quant_summary.json")
        uniform_targets[item["budget"]] = summary["runtime_totals"][
            "normalized_bop_overhead"
        ]

    corrected = []
    for item in payload["allocations"]:
        if item["strategy"] == "uniform":
            continue
        run_dir = args.run_root / item["source_run"]
        summary = load_json(run_dir / "quant_summary.json")
        runtime = load_json(run_dir / "outlier_runtime_stats.json")
        weights = item["weight_layer_ratios"]
        activations = item["activation_layer_ratios"]
        floor_weights = [
            item["uniform_weight_ratio"] * args.floor_fraction
        ] * len(weights)
        floor_activations = [
            item["uniform_activation_ratio"] * args.floor_fraction
        ] * len(activations)
        target = uniform_targets[item["budget"]]
        scale = solve_scale(
            runtime["modules"],
            weights,
            activations,
            floor_weights,
            floor_activations,
            target,
        )
        corrected.append(
            {
                **item,
                "source_run": item["source_run"],
                "source_measured_bop_overhead": summary["runtime_totals"][
                    "normalized_bop_overhead"
                ],
                "target_measured_uniform_bop_overhead": target,
                "floor_fraction": args.floor_fraction,
                "floor_preserving_excess_scale": scale,
                "predicted_corrected_bop_overhead": predicted_bop(
                    runtime["modules"],
                    weights,
                    activations,
                    floor_weights,
                    floor_activations,
                    scale,
                ),
                "weight_layer_ratios": [
                    floor + scale * (value - floor)
                    for floor, value in zip(floor_weights, weights)
                ],
                "activation_layer_ratios": [
                    floor + scale * (value - floor)
                    for floor, value in zip(floor_activations, activations)
                ],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "method": "module-wise measured-MAC BOP correction with fixed floor",
                "uniform_measured_bop_targets": uniform_targets,
                "allocations": corrected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
