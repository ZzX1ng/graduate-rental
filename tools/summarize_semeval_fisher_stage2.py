#!/usr/bin/env python3
"""Summarize SemEval Fisher allocation results across calibration seeds."""

import argparse
import json
import statistics
from pathlib import Path


def load_run(run_root, run_name):
    payload = json.loads((run_root / run_name / "quant_summary.json").read_text())
    metrics = payload["val_metrics"]["semeval"]["metrics"]
    return {
        "major": metrics["major"],
        "f1_micro": metrics["minor"]["f1_micro"],
        "bop": payload["runtime_totals"]["normalized_bop_overhead"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    seeds = (20260806, 20260807, 20260808, 20260809)
    rows = []
    for budget in ("low", "boundary"):
        per_seed = {}
        for seed in seeds:
            uniform_name = f"ptq_pg_w4a4_out8_{budget}_uniform_fixedseed{seed}_cal16_semeval_e10"
            probing_name = f"ptq_pg_w4a4_out8_{budget}_probing_bopfix2_fixedseed{seed}_cal16_semeval_e10"
            per_seed[seed, "uniform"] = load_run(args.run_root, uniform_name)
            per_seed[seed, "probing"] = load_run(args.run_root, probing_name)
            for strategy in ("fisher", "probefisher"):
                name = f"ptq_fisher_w4a4_out8_{budget}_{strategy}_bopfix2_fixedseed{seed}_cal16_semeval_e10"
                per_seed[seed, strategy] = load_run(args.run_root, name)
        for strategy in ("uniform", "probing", "fisher", "probefisher"):
            values = [per_seed[seed, strategy] for seed in seeds]
            major_delta = [
                per_seed[seed, strategy]["major"] - per_seed[seed, "uniform"]["major"]
                for seed in seeds
            ]
            probe_delta = [
                per_seed[seed, strategy]["major"] - per_seed[seed, "probing"]["major"]
                for seed in seeds
            ]
            bop_error = [
                per_seed[seed, strategy]["bop"] - per_seed[seed, "uniform"]["bop"]
                for seed in seeds
            ]
            rows.append(
                {
                    "budget": budget,
                    "strategy": strategy,
                    "major_mean": statistics.fmean(value["major"] for value in values),
                    "major_std": statistics.pstdev(value["major"] for value in values),
                    "f1_mean": statistics.fmean(value["f1_micro"] for value in values),
                    "delta_major_vs_uniform": statistics.fmean(major_delta),
                    "wins_vs_uniform": sum(value > 0 for value in major_delta),
                    "delta_major_vs_probing": statistics.fmean(probe_delta),
                    "wins_vs_probing": sum(value > 0 for value in probe_delta),
                    "max_abs_bop_error": max(abs(value) for value in bop_error),
                }
            )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with args.output_prefix.with_suffix(".tsv").open("w", encoding="utf-8") as writer:
        writer.write("\t".join(columns) + "\n")
        for row in rows:
            writer.write("\t".join(str(row[column]) for column in columns) + "\n")
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
