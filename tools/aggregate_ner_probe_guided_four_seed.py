#!/usr/bin/env python3
"""Aggregate NER low/boundary paired-BOP results across four seeds."""

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (20260806, 20260807, 20260808, 20260809)
BUDGETS = ("low", "boundary")
STRATEGIES = ("uniform", "probing", "inverse", "early", "late", "random_s29")
PREFIX = "ner_w4a4_probe_guided_outlier8_low_boundary_fixedseed"

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
ROOT = args.input_dir
OUTPUT_ROOT = args.output_dir or ROOT


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_tsv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


records = {}
run_rows = []
for seed in SEEDS:
    path = ROOT / f"{PREFIX}{seed}.tsv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != len(BUDGETS) * len(STRATEGIES):
        raise SystemExit(f"Expected 12 rows in {path}, found {len(rows)}")
    for row in rows:
        key = (row["budget"], row["strategy"], seed)
        records[key] = {
            "major": float(row["major"]),
            "f1_micro": float(row["f1_micro"]),
            "measured_bop": float(row["measured_bop"]),
            "bop_error_vs_uniform": float(row["bop_error_vs_uniform"]),
        }
        run_rows.append(row)

summary_rows = []
for budget in BUDGETS:
    for strategy in STRATEGIES:
        items = [records[(budget, strategy, seed)] for seed in SEEDS]
        major_deltas = [
            records[(budget, strategy, seed)]["major"]
            - records[(budget, "uniform", seed)]["major"]
            for seed in SEEDS
        ]
        f1_deltas = [
            records[(budget, strategy, seed)]["f1_micro"]
            - records[(budget, "uniform", seed)]["f1_micro"]
            for seed in SEEDS
        ]
        summary_rows.append(
            {
                "budget": budget,
                "strategy": strategy,
                "seed_count": len(SEEDS),
                "mean_major": statistics.mean(x["major"] for x in items),
                "std_major": sample_std([x["major"] for x in items]),
                "mean_f1_micro": statistics.mean(x["f1_micro"] for x in items),
                "std_f1_micro": sample_std([x["f1_micro"] for x in items]),
                "mean_delta_major_vs_uniform": statistics.mean(major_deltas),
                "std_delta_major_vs_uniform": sample_std(major_deltas),
                "mean_delta_f1_vs_uniform": statistics.mean(f1_deltas),
                "std_delta_f1_vs_uniform": sample_std(f1_deltas),
                "wins_vs_uniform": sum(delta > 0 for delta in f1_deltas),
                "mean_measured_bop": statistics.mean(
                    x["measured_bop"] for x in items
                ),
                "max_abs_bop_error_vs_uniform": max(
                    abs(x["bop_error_vs_uniform"]) for x in items
                ),
            }
        )

pairwise_rows = []
for budget in BUDGETS:
    for opponent in ("inverse", "early", "late", "random_s29"):
        major_deltas = [
            records[(budget, "probing", seed)]["major"]
            - records[(budget, opponent, seed)]["major"]
            for seed in SEEDS
        ]
        f1_deltas = [
            records[(budget, "probing", seed)]["f1_micro"]
            - records[(budget, opponent, seed)]["f1_micro"]
            for seed in SEEDS
        ]
        pairwise_rows.append(
            {
                "budget": budget,
                "comparison": f"probing-{opponent}",
                "mean_delta_major": statistics.mean(major_deltas),
                "std_delta_major": sample_std(major_deltas),
                "mean_delta_f1": statistics.mean(f1_deltas),
                "std_delta_f1": sample_std(f1_deltas),
                "probing_wins": sum(delta > 0 for delta in f1_deltas),
            }
        )

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
summary_path = OUTPUT_ROOT / "ner_w4a4_probe_guided_outlier8_low_boundary_multiseed_20260806_09.summary.tsv"
pairwise_path = OUTPUT_ROOT / "ner_w4a4_probe_guided_outlier8_low_boundary_multiseed_20260806_09.pairwise.tsv"
json_path = OUTPUT_ROOT / "ner_w4a4_probe_guided_outlier8_low_boundary_multiseed_20260806_09.json"
write_tsv(summary_path, summary_rows)
write_tsv(pairwise_path, pairwise_rows)
json_path.write_text(
    json.dumps(
        {
            "seeds": SEEDS,
            "budgets": BUDGETS,
            "strategies": STRATEGIES,
            "summary": summary_rows,
            "pairwise": pairwise_rows,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(summary_path)
print(pairwise_path)
print(json_path)
