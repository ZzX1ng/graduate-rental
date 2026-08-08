#!/usr/bin/env python3
"""Summarize paired low-budget probing results across calibration seeds."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_name(strategy, seed):
    suffix = f"fixedseed{seed}"
    if strategy != "uniform":
        suffix = f"bopfix2_{suffix}"
    return f"ptq_pg_w4a4_out8_low_{strategy}_{suffix}_cal16_semeval_e10"


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_tsv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as writer:
        output = csv.DictWriter(writer, fieldnames=fieldnames, delimiter="\t")
        output.writeheader()
        output.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--strategies", nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    by_seed = {}
    run_rows = []
    for seed in args.seeds:
        by_seed[seed] = {}
        for strategy in args.strategies:
            name = run_name(strategy, seed)
            summary_path = args.run_root / name / "quant_summary.json"
            summary = load_json(summary_path)
            major = float(summary["val_metrics"]["aggregated"])
            measured_bop = float(
                summary["runtime_totals"]["normalized_bop_overhead"]
            )
            by_seed[seed][strategy] = {
                "run_name": name,
                "major": major,
                "measured_bop": measured_bop,
            }

        uniform = by_seed[seed]["uniform"]
        for strategy in args.strategies:
            item = by_seed[seed][strategy]
            run_rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "major": item["major"],
                    "delta_major_vs_uniform": item["major"] - uniform["major"],
                    "measured_bop": item["measured_bop"],
                    "bop_error_vs_uniform": (
                        item["measured_bop"] - uniform["measured_bop"]
                    ),
                    "run_name": item["run_name"],
                }
            )

    summary_rows = []
    for strategy in args.strategies:
        items = [by_seed[seed][strategy] for seed in args.seeds]
        deltas = [
            by_seed[seed][strategy]["major"]
            - by_seed[seed]["uniform"]["major"]
            for seed in args.seeds
        ]
        bop_errors = [
            by_seed[seed][strategy]["measured_bop"]
            - by_seed[seed]["uniform"]["measured_bop"]
            for seed in args.seeds
        ]
        summary_rows.append(
            {
                "strategy": strategy,
                "seed_count": len(args.seeds),
                "mean_major": statistics.mean(item["major"] for item in items),
                "std_major": sample_std([item["major"] for item in items]),
                "mean_delta_major_vs_uniform": statistics.mean(deltas),
                "std_delta_major_vs_uniform": sample_std(deltas),
                "wins_vs_uniform": sum(delta > 0 for delta in deltas),
                "mean_measured_bop": statistics.mean(
                    item["measured_bop"] for item in items
                ),
                "max_abs_bop_error_vs_uniform": max(map(abs, bop_errors)),
            }
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    run_path = args.output_prefix.with_suffix(".runs.tsv")
    summary_path = args.output_prefix.with_suffix(".summary.tsv")
    json_path = args.output_prefix.with_suffix(".json")
    write_tsv(run_path, run_rows, list(run_rows[0]))
    write_tsv(summary_path, summary_rows, list(summary_rows[0]))
    json_path.write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "strategies": args.strategies,
                "runs": run_rows,
                "summary": summary_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(run_path)
    print(summary_path)
    print(json_path)


if __name__ == "__main__":
    main()
