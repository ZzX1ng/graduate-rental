#!/usr/bin/env python3
"""Summarize one-seed, paired-BOP probing results for multiple budgets."""

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_name(budget, strategy, seed, run_suffix):
    if strategy == "uniform":
        tag = f"fixedseed{seed}"
    else:
        tag = f"bopfix2_fixedseed{seed}"
    return f"ptq_pg_w4a4_out8_{budget}_{strategy}_{tag}_{run_suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budgets", nargs="+", required=True)
    parser.add_argument("--strategies", nargs="+", required=True)
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for budget in args.budgets:
        items = {}
        for strategy in args.strategies:
            name = run_name(budget, strategy, args.seed, args.run_suffix)
            summary = load_json(args.run_root / name / "quant_summary.json")
            metrics = summary["val_metrics"][args.task]["metrics"]
            runtime = summary["runtime_totals"]
            items[strategy] = {
                "run_name": name,
                "major": float(metrics["major"]),
                "f1_micro": float(metrics["minor"]["f1_micro"]),
                "acc": float(metrics["minor"]["acc"]),
                "measured_bop": float(runtime["normalized_bop_overhead"]),
            }

        uniform = items["uniform"]
        for strategy in args.strategies:
            item = items[strategy]
            rows.append(
                {
                    "budget": budget,
                    "strategy": strategy,
                    "seed": args.seed,
                    "major": item["major"],
                    "f1_micro": item["f1_micro"],
                    "acc": item["acc"],
                    "delta_major_vs_uniform": item["major"] - uniform["major"],
                    "delta_f1_vs_uniform": item["f1_micro"] - uniform["f1_micro"],
                    "measured_bop": item["measured_bop"],
                    "bop_error_vs_uniform": (
                        item["measured_bop"] - uniform["measured_bop"]
                    ),
                    "run_name": item["run_name"],
                }
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_prefix.with_suffix(".tsv")
    json_path = args.output_prefix.with_suffix(".json")
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "task": args.task,
                "seed": args.seed,
                "budgets": args.budgets,
                "strategies": args.strategies,
                "runs": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(tsv_path)
    print(json_path)


if __name__ == "__main__":
    main()
