#!/usr/bin/env python3
"""Rescale layer allocations to match each budget's measured uniform BOP."""

import argparse
import json
import math
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_name(item, run_tag, run_suffix):
    prefix = f"ptq_pg_w4a4_out8_{item['budget']}_{item['strategy']}"
    if run_tag:
        prefix += f"_{run_tag}"
    return f"{prefix}_{run_suffix}"


def source_run_name(item, run_tag, run_suffix):
    return item.get("source_run", run_name(item, run_tag, run_suffix))


def solve_scale(runtime_totals, target_bop):
    total_macs = runtime_totals["low_bit_macs"]
    linear = (
        runtime_totals["weight_outlier_macs"]
        + runtime_totals["activation_outlier_macs"]
    ) / total_macs
    quadratic = runtime_totals["dual_outlier_macs"] / total_macs
    if quadratic == 0:
        return target_bop / linear
    return (-linear + math.sqrt(linear * linear + 4.0 * quadratic * target_bop)) / (
        2.0 * quadratic
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocations", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--run-suffix", default="cal16_semeval_e10")
    args = parser.parse_args()

    source = load_json(args.allocations)
    uniform_bop = {}
    for item in source["allocations"]:
        if item["strategy"] != "uniform":
            continue
        source_run = source_run_name(item, args.run_tag, args.run_suffix)
        summary = load_json(args.run_root / source_run / "quant_summary.json")
        uniform_bop[item["budget"]] = summary["runtime_totals"][
            "normalized_bop_overhead"
        ]

    corrected = []
    for item in source["allocations"]:
        if item["strategy"] == "uniform":
            continue
        source_run = source_run_name(item, args.run_tag, args.run_suffix)
        summary = load_json(args.run_root / source_run / "quant_summary.json")
        totals = summary["runtime_totals"]
        measured = totals["normalized_bop_overhead"]
        target = uniform_bop[item["budget"]]
        scale = solve_scale(totals, target)
        corrected.append(
            {
                **item,
                "source_run": source_run,
                "source_measured_bop_overhead": measured,
                "target_measured_uniform_bop_overhead": target,
                "bop_correction_scale": scale,
                "weight_layer_ratios": [
                    value * scale for value in item["weight_layer_ratios"]
                ],
                "activation_layer_ratios": [
                    value * scale for value in item["activation_layer_ratios"]
                ],
            }
        )

    payload = {
        "method": (
            "one-step scale from measured weight/activation/dual MAC decomposition; "
            "estimated overhead(s)=s*(Wmac+Amac)/total+s^2*dual/total"
        ),
        "uniform_measured_bop_targets": uniform_bop,
        "allocations": corrected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
