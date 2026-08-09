#!/usr/bin/env python3
"""Summarize DEP floor50 six-strategy pilot and BOP-fix2 runs."""

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (20260806, 20260807, 20260808, 20260809)
BUDGETS = {
    "low": "w0p055_a0p275",
    "boundary": "w0p070_a0p350",
}
STRATEGIES = ("uniform", "probing", "inverse", "early", "late", "random_s29")
PHASES = ("pilot", "bopfix2")


def run_name(phase, seed, label, strategy):
    if strategy == "uniform":
        return (
            "ptq_oa_w4a4_dualout8_boundaryrefine_"
            f"{label}_fixedseed{seed}_cal16_ms256_dep_e3_retry1"
        )
    suffix = "_bopfix2" if phase == "bopfix2" else ""
    return (
        f"ptq_pgfloor50_w4a4_out8_{label}_{strategy}{suffix}_"
        f"fixedseed{seed}_cal16_dep_e3_retry1"
    )


def load_metrics(run_root, name):
    path = run_root / name / "quant_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["val_metrics"]["dep"]["metrics"]
    return {
        "major": float(metrics["major"]),
        "f1_micro": float(metrics["minor"]["f1_micro"]),
        "acc": float(metrics["minor"]["acc"]),
        "measured_bop": float(
            payload["runtime_totals"]["normalized_bop_overhead"]
        ),
    }


def aggregate(rows):
    output = []
    for phase in PHASES:
        for budget in BUDGETS:
            subset = [
                row
                for row in rows
                if row["phase"] == phase and row["budget"] == budget
            ]
            probing_by_seed = {
                row["seed"]: row
                for row in subset
                if row["strategy"] == "probing"
            }
            for strategy in STRATEGIES:
                strategy_rows = [
                    row for row in subset if row["strategy"] == strategy
                ]
                major = [row["major"] for row in strategy_rows]
                f1 = [row["f1_micro"] for row in strategy_rows]
                bop = [row["measured_bop"] for row in strategy_rows]
                output.append(
                    {
                        "phase": phase,
                        "budget": budget,
                        "strategy": strategy,
                        "seed_count": len(strategy_rows),
                        "mean_major": statistics.mean(major),
                        "std_major": statistics.pstdev(major),
                        "mean_f1_micro": statistics.mean(f1),
                        "std_f1_micro": statistics.pstdev(f1),
                        "mean_measured_bop": statistics.mean(bop),
                        "mean_delta_major_vs_uniform": statistics.mean(
                            row["delta_major_vs_uniform"]
                            for row in strategy_rows
                        ),
                        "mean_delta_f1_vs_uniform": statistics.mean(
                            row["delta_f1_vs_uniform"]
                            for row in strategy_rows
                        ),
                        "wins_vs_uniform_major": sum(
                            row["delta_major_vs_uniform"] > 0
                            for row in strategy_rows
                        ),
                        "wins_vs_uniform_f1": sum(
                            row["delta_f1_vs_uniform"] > 0
                            for row in strategy_rows
                        ),
                        "probing_wins_major": sum(
                            probing_by_seed[row["seed"]]["major"]
                            > row["major"]
                            for row in strategy_rows
                        ),
                        "probing_wins_f1": sum(
                            probing_by_seed[row["seed"]]["f1_micro"]
                            > row["f1_micro"]
                            for row in strategy_rows
                        ),
                        "max_abs_bop_error_vs_uniform": max(
                            abs(row["bop_error_vs_uniform"])
                            for row in strategy_rows
                        ),
                        "all_within_2pct_uniform_bop": all(
                            row["within_2pct_uniform_bop"]
                            for row in strategy_rows
                        ),
                    }
                )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for phase in PHASES:
        for budget, label in BUDGETS.items():
            for seed in SEEDS:
                metrics = {
                    strategy: load_metrics(
                        args.run_root,
                        run_name(phase, seed, label, strategy),
                    )
                    for strategy in STRATEGIES
                }
                uniform = metrics["uniform"]
                probing = metrics["probing"]
                for strategy in STRATEGIES:
                    item = metrics[strategy]
                    bop_error = item["measured_bop"] - uniform["measured_bop"]
                    rows.append(
                        {
                            "phase": phase,
                            "budget": budget,
                            "seed": seed,
                            "strategy": strategy,
                            **item,
                            "delta_major_vs_uniform": (
                                item["major"] - uniform["major"]
                            ),
                            "delta_f1_vs_uniform": (
                                item["f1_micro"] - uniform["f1_micro"]
                            ),
                            "bop_error_vs_uniform": bop_error,
                            "relative_bop_error_vs_uniform": (
                                bop_error / uniform["measured_bop"]
                            ),
                            "within_2pct_uniform_bop": (
                                abs(bop_error / uniform["measured_bop"])
                                <= 0.02
                            ),
                            "probing_minus_major": (
                                probing["major"] - item["major"]
                            ),
                            "probing_minus_f1": (
                                probing["f1_micro"] - item["f1_micro"]
                            ),
                            "run_name": run_name(
                                phase, seed, label, strategy
                            ),
                        }
                    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_prefix.with_suffix(".tsv")
    json_path = args.output_prefix.with_suffix(".json")
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {"rows": rows, "aggregates": aggregate(rows)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(tsv_path)
    print(json_path)


if __name__ == "__main__":
    main()
