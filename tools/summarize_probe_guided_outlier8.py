#!/usr/bin/env python3
"""Summarize fixed-seed probing-guided W4A4/outlier8 experiments."""

import argparse
import csv
import json
import statistics
from pathlib import Path


STRATEGIES = [
    "uniform",
    "probing",
    "inverse",
    "early",
    "late",
    "random_s17",
    "random_s29",
    "random_s43",
]


def load_summary(run_root, run_name):
    path = run_root / run_name / "quant_summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    totals = payload["runtime_totals"]
    return {
        "run_name": run_name,
        "major": payload["val_metrics"]["aggregated"],
        "measured_bop": totals["normalized_bop_overhead"],
        "weight_actual_ratio": totals["weight_outlier_ratio"],
        "activation_actual_ratio": totals["activation_outlier_ratio"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("fixed", "unfixed"), default="fixed")
    args = parser.parse_args()

    rows = []
    for budget in ("low", "boundary"):
        if args.variant == "fixed":
            uniform_name = (
                f"ptq_pg_w4a4_out8_{budget}_uniform_"
                "fixedseed20260806_cal16_semeval_e10"
            )
        else:
            uniform_name = f"ptq_pg_w4a4_out8_{budget}_uniform_cal16_semeval_e10"
        uniform = load_summary(args.run_root, uniform_name)
        if uniform is None:
            continue
        for strategy in STRATEGIES:
            if strategy == "uniform":
                phase = f"{args.variant}_nominal"
                result = uniform
            else:
                if args.variant == "fixed":
                    phase = "fixedseed_bopfix2"
                    name = (
                        f"ptq_pg_w4a4_out8_{budget}_{strategy}_"
                        "bopfix2_fixedseed20260806_cal16_semeval_e10"
                    )
                else:
                    phase = "unfixed_bopfix1"
                    name = (
                        f"ptq_pg_w4a4_out8_{budget}_{strategy}_"
                        "bopfix1_cal16_semeval_e10"
                    )
                result = load_summary(args.run_root, name)
            if result is None:
                continue
            rows.append(
                {
                    "budget": budget,
                    "strategy": strategy,
                    "phase": phase,
                    **result,
                    "delta_major_vs_uniform": result["major"] - uniform["major"],
                    "bop_error_vs_uniform": result["measured_bop"]
                    - uniform["measured_bop"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "budget",
        "strategy",
        "phase",
        "major",
        "measured_bop",
        "delta_major_vs_uniform",
        "bop_error_vs_uniform",
        "weight_actual_ratio",
        "activation_actual_ratio",
        "run_name",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    for budget in ("low", "boundary"):
        budget_rows = [row for row in rows if row["budget"] == budget]
        if not budget_rows:
            continue
        print(f"[{budget}]")
        for row in sorted(budget_rows, key=lambda item: item["major"], reverse=True):
            print(
                f"{row['strategy']:10s} major={row['major']:.6f} "
                f"delta={row['delta_major_vs_uniform']:+.6f} "
                f"BOP={100 * row['measured_bop']:.4f}% "
                f"BOPerr={100 * row['bop_error_vs_uniform']:+.4f}pp"
            )
        random_rows = [row for row in budget_rows if row["strategy"].startswith("random_")]
        if len(random_rows) == 3:
            random_majors = [row["major"] for row in random_rows]
            print(
                f"random mean={statistics.mean(random_majors):.6f}, "
                f"sample_std={statistics.stdev(random_majors):.6f}"
            )


if __name__ == "__main__":
    main()
