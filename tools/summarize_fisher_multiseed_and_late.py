#!/usr/bin/env python3
import json
import statistics
from pathlib import Path

ROOT = Path("base_exp/exp_edge/runs/bert-large-uncased")
OUT = Path("base_exp/exp_edge/analysis/ptq")
SEEDS = (20260806, 20260807, 20260808, 20260809)


def load(task, name):
    payload = json.loads((ROOT / task / name / "quant_summary.json").read_text())
    metrics = payload["val_metrics"][task]["metrics"]
    return {
        "major": metrics["major"],
        "f1_micro": metrics["minor"]["f1_micro"],
        "bop": payload["runtime_totals"]["normalized_bop_overhead"],
        "run_name": name,
    }


def uniform_name(task, budget, seed):
    if task == "ner":
        return f"ptq_pg_w4a4_out8_{budget}_uniform_fixedseed{seed}_cal16_ner_e3"
    label = "w0p055_a0p275" if budget == "low" else "w0p070_a0p350"
    return (
        f"ptq_oa_w4a4_dualout8_boundaryrefine_{label}_fixedseed{seed}_"
        "cal16_ms256_dep_e3_retry1"
    )


def probing_name(task, budget, seed):
    if task == "ner":
        return (
            f"ptq_pg_w4a4_out8_{budget}_probing_bopfix2_"
            f"fixedseed{seed}_cal16_ner_e3"
        )
    label = "w0p055_a0p275" if budget == "low" else "w0p070_a0p350"
    return (
        f"ptq_pgfloor50_w4a4_out8_{label}_probing_bopfix2_"
        f"fixedseed{seed}_cal16_dep_e3_retry1"
    )


def late_name(budget, seed):
    return (
        f"ptq_pg_w4a4_out8_{budget}_late_bopfix2_"
        f"fixedseed{seed}_cal16_ner_e3"
    )


def fisher_name(task, budget, strategy, seed):
    suffix = "ner_e3" if task == "ner" else "dep_e3_retry1"
    return (
        f"ptq_fisher_w4a4_out8_{budget}_{strategy}_bopfix2_"
        f"fixedseed{seed}_cal16_{suffix}"
    )


def write_tsv(path, rows):
    columns = list(rows[0])
    with path.open("w", encoding="utf-8") as writer:
        writer.write("\t".join(columns) + "\n")
        for row in rows:
            writer.write("\t".join(str(row[column]) for column in columns) + "\n")


def mean(values):
    return statistics.fmean(values)


def std(values):
    return statistics.stdev(values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    detail = []
    aggregates = []
    for task in ("ner", "dep"):
        for budget in ("low", "boundary"):
            seed_rows = []
            for seed in SEEDS:
                uniform = load(task, uniform_name(task, budget, seed))
                probing = load(task, probing_name(task, budget, seed))
                values = {
                    "uniform": uniform,
                    "probing": probing,
                    "fisher": load(task, fisher_name(task, budget, "fisher", seed)),
                    "probefisher": load(
                        task, fisher_name(task, budget, "probefisher", seed)
                    ),
                }
                if task == "ner":
                    values["late"] = load(task, late_name(budget, seed))
                for strategy, value in values.items():
                    row = {
                        "task": task,
                        "budget": budget,
                        "seed": seed,
                        "strategy": strategy,
                        "major": value["major"],
                        "f1_micro": value["f1_micro"],
                        "measured_bop": value["bop"],
                        "delta_major_vs_uniform": value["major"] - uniform["major"],
                        "delta_f1_vs_uniform": value["f1_micro"] - uniform["f1_micro"],
                        "delta_major_vs_probing": value["major"] - probing["major"],
                        "delta_f1_vs_probing": value["f1_micro"] - probing["f1_micro"],
                        "relative_bop_error_vs_uniform_pct": 100
                        * (value["bop"] / uniform["bop"] - 1),
                        "run_name": value["run_name"],
                    }
                    detail.append(row)
                    seed_rows.append(row)
            strategies = sorted({row["strategy"] for row in seed_rows})
            for strategy in strategies:
                selected = [row for row in seed_rows if row["strategy"] == strategy]
                bop_errors = [
                    row["relative_bop_error_vs_uniform_pct"] for row in selected
                ]
                aggregates.append(
                    {
                        "task": task,
                        "budget": budget,
                        "strategy": strategy,
                        "major_mean": mean([row["major"] for row in selected]),
                        "major_std": std([row["major"] for row in selected]),
                        "f1_mean": mean([row["f1_micro"] for row in selected]),
                        "f1_std": std([row["f1_micro"] for row in selected]),
                        "paired_delta_major_vs_uniform_mean": mean(
                            [row["delta_major_vs_uniform"] for row in selected]
                        ),
                        "paired_delta_f1_vs_uniform_mean": mean(
                            [row["delta_f1_vs_uniform"] for row in selected]
                        ),
                        "paired_delta_major_vs_probing_mean": mean(
                            [row["delta_major_vs_probing"] for row in selected]
                        ),
                        "paired_delta_f1_vs_probing_mean": mean(
                            [row["delta_f1_vs_probing"] for row in selected]
                        ),
                        "major_wins_vs_uniform": sum(
                            row["delta_major_vs_uniform"] > 0 for row in selected
                        ),
                        "major_wins_vs_probing": sum(
                            row["delta_major_vs_probing"] > 0 for row in selected
                        ),
                        "bop_error_pct_mean": mean(bop_errors),
                        "bop_error_pct_max_abs": max(map(abs, bop_errors)),
                    }
                )

    late_detail = []
    for budget in ("low", "boundary"):
        for seed in (20260810, 20260811, 20260812, 20260813):
            for phase in ("pilot", "bopfix2"):
                middle = "late" if phase == "pilot" else "late_bopfix2"
                name = (
                    f"ptq_pg_w4a4_out8_{budget}_{middle}_newseed{seed}_"
                    "cal16_ner_e3"
                )
                value = load("ner", name)
                late_detail.append(
                    {
                        "budget": budget,
                        "seed": seed,
                        "phase": phase,
                        "major": value["major"],
                        "f1_micro": value["f1_micro"],
                        "measured_bop": value["bop"],
                        "run_name": name,
                    }
                )
    late_summary = []
    for budget in ("low", "boundary"):
        target = json.loads(
            (OUT / "ner_late_new_seeds_only_fixed_targets.json").read_text()
        )[budget]["mean"]
        for phase in ("pilot", "bopfix2"):
            selected = [
                row
                for row in late_detail
                if row["budget"] == budget and row["phase"] == phase
            ]
            errors = [100 * (row["measured_bop"] / target - 1) for row in selected]
            late_summary.append(
                {
                    "budget": budget,
                    "phase": phase,
                    "major_mean": mean([row["major"] for row in selected]),
                    "major_std": std([row["major"] for row in selected]),
                    "f1_mean": mean([row["f1_micro"] for row in selected]),
                    "f1_std": std([row["f1_micro"] for row in selected]),
                    "fixed_bop_target": target,
                    "bop_error_pct_mean": mean(errors),
                    "bop_error_pct_max_abs": max(map(abs, errors)),
                }
            )

    prefix = OUT / "ner_dep_fisher_multiseed_20260806_09"
    write_tsv(prefix.with_suffix(".detail.tsv"), detail)
    write_tsv(prefix.with_suffix(".summary.tsv"), aggregates)
    prefix.with_suffix(".json").write_text(
        json.dumps({"detail": detail, "summary": aggregates}, indent=2) + "\n"
    )
    late_prefix = OUT / "ner_late_newseeds_20260810_13"
    write_tsv(late_prefix.with_suffix(".detail.tsv"), late_detail)
    write_tsv(late_prefix.with_suffix(".summary.tsv"), late_summary)
    late_prefix.with_suffix(".json").write_text(
        json.dumps({"detail": late_detail, "summary": late_summary}, indent=2) + "\n"
    )
    print(prefix.with_suffix(".summary.tsv"))
    print(late_prefix.with_suffix(".summary.tsv"))


if __name__ == "__main__":
    main()
