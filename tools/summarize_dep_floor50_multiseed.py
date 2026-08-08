#!/usr/bin/env python3
"""Summarize DEP floor50 uniform/probing BOP-fix pairs across four seeds."""

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (20260806, 20260807, 20260808, 20260809)
BUDGET_LABELS = {
    "low": "w0p055_a0p275",
    "boundary": "w0p070_a0p350",
}


def load_summary(path):
    return json.loads(path.read_text(encoding="utf-8"))


def uniform_run(seed, label):
    if seed == 20260806:
        return (
            "ptq_oa_w4a4_dualout8_boundaryrefine_"
            f"{label}_fixedseed{seed}_cal16_ms256_dep_e3_retry1"
        )
    return (
        f"ptq_pgfloor50_w4a4_out8_{label}_uniform_"
        f"fixedseed{seed}_cal16_dep_e3_retry1"
    )


def probing_run(seed, label):
    return (
        f"ptq_pgfloor50_w4a4_out8_{label}_probing_bopfix2_"
        f"fixedseed{seed}_cal16_dep_e3_retry1"
    )


def metrics(run_root, run_name):
    summary = load_summary(run_root / run_name / "quant_summary.json")
    task_metrics = summary["val_metrics"]["dep"]["metrics"]
    return {
        "major": float(task_metrics["major"]),
        "f1_micro": float(task_metrics["minor"]["f1_micro"]),
        "measured_bop": float(
            summary["runtime_totals"]["normalized_bop_overhead"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for budget, label in BUDGET_LABELS.items():
        for seed in SEEDS:
            uniform_name = uniform_run(seed, label)
            probing_name = probing_run(seed, label)
            uniform = metrics(args.run_root, uniform_name)
            probing = metrics(args.run_root, probing_name)
            rows.append(
                {
                    "budget": budget,
                    "seed": seed,
                    "uniform_major": uniform["major"],
                    "probing_major": probing["major"],
                    "delta_major": probing["major"] - uniform["major"],
                    "uniform_f1_micro": uniform["f1_micro"],
                    "probing_f1_micro": probing["f1_micro"],
                    "delta_f1_micro": probing["f1_micro"] - uniform["f1_micro"],
                    "uniform_measured_bop": uniform["measured_bop"],
                    "probing_measured_bop": probing["measured_bop"],
                    "bop_error": probing["measured_bop"] - uniform["measured_bop"],
                    "uniform_run": uniform_name,
                    "probing_run": probing_name,
                }
            )

    aggregates = []
    for budget in BUDGET_LABELS:
        subset = [row for row in rows if row["budget"] == budget]
        aggregate = {"budget": budget, "seed_count": len(subset)}
        for key in (
            "uniform_major",
            "probing_major",
            "delta_major",
            "uniform_f1_micro",
            "probing_f1_micro",
            "delta_f1_micro",
            "uniform_measured_bop",
            "probing_measured_bop",
            "bop_error",
        ):
            values = [row[key] for row in subset]
            aggregate[f"mean_{key}"] = statistics.mean(values)
            aggregate[f"std_{key}"] = statistics.pstdev(values)
        aggregate["major_wins"] = sum(row["delta_major"] > 0 for row in subset)
        aggregate["f1_wins"] = sum(row["delta_f1_micro"] > 0 for row in subset)
        aggregate["max_abs_bop_error"] = max(
            abs(row["bop_error"]) for row in subset
        )
        aggregates.append(aggregate)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_prefix.with_suffix(".tsv")
    json_path = args.output_prefix.with_suffix(".json")
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps({"rows": rows, "aggregates": aggregates}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(tsv_path)
    print(json_path)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
