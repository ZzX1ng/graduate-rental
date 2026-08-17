#!/usr/bin/env python3
"""Summarize NER/DEP Fisher allocation pilots at seed 20260806."""

import argparse
import json
from pathlib import Path


def load_run(run_root, task, run_name):
    payload = json.loads((run_root / task / run_name / "quant_summary.json").read_text())
    metrics = payload["val_metrics"][task]["metrics"]
    return {
        "major": metrics["major"],
        "f1_micro": metrics["minor"]["f1_micro"],
        "bop": payload["runtime_totals"]["normalized_bop_overhead"],
        "weight_ratios": payload["weight_target_layer_ratios"],
        "activation_ratios": payload["activation_target_layer_ratios"],
        "run_name": run_name,
    }


def own_name(task, budget, strategy, seed):
    if task == "ner":
        return (
            f"ptq_pg_w4a4_out8_{budget}_{strategy}_bopfix2_"
            f"fixedseed{seed}_cal16_ner_e3"
        )
    label = "w0p055_a0p275" if budget == "low" else "w0p070_a0p350"
    return (
        f"ptq_pgfloor50_w4a4_out8_{label}_{strategy}_bopfix2_"
        f"fixedseed{seed}_cal16_dep_e3_retry1"
    )


def uniform_name(task, budget, seed):
    if task == "ner":
        return f"ptq_pg_w4a4_out8_{budget}_uniform_fixedseed{seed}_cal16_ner_e3"
    label = "w0p055_a0p275" if budget == "low" else "w0p070_a0p350"
    return (
        f"ptq_oa_w4a4_dualout8_boundaryrefine_{label}_fixedseed{seed}_"
        "cal16_ms256_dep_e3_retry1"
    )


def fisher_name(task, budget, strategy, seed, phase):
    suffix = "ner_e3" if task == "ner" else "dep_e3_retry1"
    phase_text = "_bopfix2" if phase == "bopfix2" else ""
    return (
        f"ptq_fisher_w4a4_out8_{budget}_{strategy}{phase_text}_"
        f"fixedseed{seed}_cal16_{suffix}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    seed = 20260806
    budgets = {
        "ner": {
            "low": (0.0003125, 0.0015625, 0.0),
            "boundary": (0.0004375, 0.0021875, 0.0),
        },
        "dep": {
            "low": (0.00055, 0.00275, 0.5),
            "boundary": (0.00070, 0.00350, 0.5),
        },
    }
    rows = []
    for task in ("ner", "dep"):
        for budget, (base_weight, base_activation, floor) in budgets[task].items():
            uniform = load_run(
                args.run_root, task, uniform_name(task, budget, seed)
            )
            probing = load_run(
                args.run_root, task, own_name(task, budget, "probing", seed)
            )
            late = (
                load_run(args.run_root, task, own_name(task, budget, "late", seed))
                if task == "ner"
                else None
            )
            candidates = [("uniform", "reference", uniform), ("probing", "bopfix2", probing)]
            if late is not None:
                candidates.append(("late", "bopfix2", late))
            for strategy in ("fisher", "probefisher"):
                for phase in ("pilot", "bopfix2"):
                    candidates.append(
                        (
                            strategy,
                            phase,
                            load_run(
                                args.run_root,
                                task,
                                fisher_name(task, budget, strategy, seed, phase),
                            ),
                        )
                    )
            for strategy, phase, value in candidates:
                rows.append(
                    {
                        "task": task,
                        "budget": budget,
                        "strategy": strategy,
                        "phase": phase,
                        "major": value["major"],
                        "f1_micro": value["f1_micro"],
                        "measured_bop": value["bop"],
                        "bop_error": value["bop"] - uniform["bop"],
                        "relative_bop_error_pct": 100 * (value["bop"] / uniform["bop"] - 1),
                        "delta_major_vs_uniform": value["major"] - uniform["major"],
                        "delta_f1_vs_uniform": value["f1_micro"] - uniform["f1_micro"],
                        "delta_major_vs_probing": value["major"] - probing["major"],
                        "delta_f1_vs_probing": value["f1_micro"] - probing["f1_micro"],
                        "delta_major_vs_late": (
                            value["major"] - late["major"] if late is not None else None
                        ),
                        "min_weight_ratio": min(value["weight_ratios"]),
                        "min_activation_ratio": min(value["activation_ratios"]),
                        "weight_floor_ok": (
                            min(value["weight_ratios"]) + 1e-15 >= base_weight * floor
                            if strategy in ("fisher", "probefisher")
                            else None
                        ),
                        "activation_floor_ok": (
                            min(value["activation_ratios"]) + 1e-15 >= base_activation * floor
                            if strategy in ("fisher", "probefisher")
                            else None
                        ),
                        "run_name": value["run_name"],
                    }
                )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with args.output_prefix.with_suffix(".tsv").open("w", encoding="utf-8") as writer:
        writer.write("\t".join(columns) + "\n")
        for row in rows:
            writer.write(
                "\t".join("" if row[column] is None else str(row[column]) for column in columns)
                + "\n"
            )
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        if row["phase"] in ("reference", "bopfix2"):
            print(
                row["task"], row["budget"], row["strategy"],
                f"major={row['major']:.9f}", f"f1={row['f1_micro']:.9f}",
                f"d_uniform={row['delta_major_vs_uniform']:+.9f}",
                f"d_probe={row['delta_major_vs_probing']:+.9f}",
                f"bop_err={row['relative_bop_error_pct']:+.3f}%",
            )


if __name__ == "__main__":
    main()
