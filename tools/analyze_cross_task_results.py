#!/usr/bin/env python3
"""Summarize SemEval boundary multiseed and single-seed cross-task allocations."""

import json
import math
import statistics
from pathlib import Path


ROOT = Path("base_exp/exp_edge")
RUNS = ROOT / "runs" / "bert-large-uncased"
ANALYSIS = ROOT / "analysis" / "ptq"
SEEDS = (20260806, 20260807, 20260808, 20260809)
STRATEGIES = ("uniform", "probing", "inverse", "early", "late", "random_s29")


def read_run(task, run_name):
    path = RUNS / task / run_name / "quant_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["val_metrics"][task]["metrics"]
    return {
        "major": metrics["major"],
        "f1_micro": metrics["minor"]["f1_micro"],
        "bop": payload["runtime_totals"]["normalized_bop_overhead"],
        "run_name": run_name,
    }


def standard_name(task, budget, strategy, seed, phase="bopfix2"):
    suffix = "cal16_semeval_e10" if task == "semeval" else "cal16_ner_e3"
    phase_text = "" if strategy == "uniform" or not phase else f"_{phase}"
    return f"ptq_pg_w4a4_out8_{budget}_{strategy}{phase_text}_fixedseed{seed}_{suffix}"


def dep_name(budget, strategy, seed):
    label = "w0p055_a0p275" if budget == "low" else "w0p070_a0p350"
    if strategy == "uniform":
        return (
            f"ptq_oa_w4a4_dualout8_boundaryrefine_{label}_fixedseed{seed}_"
            "cal16_ms256_dep_e3_retry1"
        )
    return (
        f"ptq_pgfloor50_w4a4_out8_{label}_{strategy}_bopfix2_fixedseed{seed}_"
        "cal16_dep_e3_retry1"
    )


def own_name(task, budget, strategy, seed):
    if task == "dep":
        return dep_name(budget, strategy, seed)
    return standard_name(task, budget, strategy, seed)


def cross_name(target, source, budget, seed, phase="bopfix2"):
    suffix = {
        "semeval": "cal16_semeval_e10",
        "ner": "cal16_ner_e3",
        "dep": "cal16_dep_e3_retry1",
    }[target]
    phase_text = f"_{phase}" if phase else ""
    return (
        f"ptq_xprobe_w4a4_out8_{budget}_src{source}{phase_text}_"
        f"fixedseed{seed}_{suffix}"
    )


def mean(values):
    return statistics.fmean(values)


def pstdev(values):
    return statistics.pstdev(values)


def pearson(left, right):
    lm, rm = mean(left), mean(right)
    numerator = sum((l - lm) * (r - rm) for l, r in zip(left, right))
    denominator = math.sqrt(
        sum((l - lm) ** 2 for l in left) * sum((r - rm) ** 2 for r in right)
    )
    return numerator / denominator


def write_tsv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as writer:
        writer.write("\t".join(columns) + "\n")
        for row in rows:
            writer.write("\t".join(str(row[column]) for column in columns) + "\n")


def semeval_boundary_summary():
    per_seed = {}
    for seed in SEEDS:
        for strategy in STRATEGIES:
            per_seed[seed, strategy] = read_run(
                "semeval", own_name("semeval", "boundary", strategy, seed)
            )

    rows = []
    for strategy in STRATEGIES:
        values = [per_seed[seed, strategy] for seed in SEEDS]
        major_delta = [
            per_seed[seed, strategy]["major"] - per_seed[seed, "uniform"]["major"]
            for seed in SEEDS
        ]
        f1_delta = [
            per_seed[seed, strategy]["f1_micro"]
            - per_seed[seed, "uniform"]["f1_micro"]
            for seed in SEEDS
        ]
        bop_error = [
            value["bop"] - per_seed[seed, "uniform"]["bop"]
            for seed, value in zip(SEEDS, values)
        ]
        rows.append(
            {
                "strategy": strategy,
                "major_mean": f"{mean([v['major'] for v in values]):.9f}",
                "major_sd": f"{pstdev([v['major'] for v in values]):.9f}",
                "f1_mean": f"{mean([v['f1_micro'] for v in values]):.9f}",
                "f1_sd": f"{pstdev([v['f1_micro'] for v in values]):.9f}",
                "delta_major_vs_uniform": f"{mean(major_delta):+.9f}",
                "delta_f1_vs_uniform": f"{mean(f1_delta):+.9f}",
                "major_wins_vs_uniform": sum(value > 0 for value in major_delta),
                "f1_wins_vs_uniform": sum(value > 0 for value in f1_delta),
                "mean_abs_bop_error": f"{mean([abs(v) for v in bop_error]):.9g}",
                "max_abs_bop_error": f"{max(abs(v) for v in bop_error):.9g}",
            }
        )
    return rows, per_seed


def cross_summary(seed=20260806):
    source_map = {
        "semeval": ("ner", "dep"),
        "ner": ("semeval", "dep"),
        "dep": ("semeval", "ner"),
    }
    rows = []
    for target, sources in source_map.items():
        for budget in ("low", "boundary"):
            uniform = read_run(target, own_name(target, budget, "uniform", seed))
            own = read_run(target, own_name(target, budget, "probing", seed))
            candidates = [("uniform", uniform), (target, own)]
            candidates.extend(
                (source, read_run(target, cross_name(target, source, budget, seed)))
                for source in sources
            )
            for allocation_source, value in candidates:
                rows.append(
                    {
                        "target": target,
                        "budget": budget,
                        "allocation_source": allocation_source,
                        "major": f"{value['major']:.9f}",
                        "f1_micro": f"{value['f1_micro']:.9f}",
                        "measured_bop": f"{value['bop']:.9f}",
                        "bop_error_vs_uniform": f"{value['bop'] - uniform['bop']:+.9g}",
                        "delta_major_vs_uniform": f"{value['major'] - uniform['major']:+.9f}",
                        "delta_f1_vs_uniform": f"{value['f1_micro'] - uniform['f1_micro']:+.9f}",
                        "delta_major_vs_own_probe": f"{value['major'] - own['major']:+.9f}",
                        "delta_f1_vs_own_probe": f"{value['f1_micro'] - own['f1_micro']:+.9f}",
                    }
                )
    return rows


def cross_phase_summary(seed=20260806):
    source_map = {
        "semeval": ("ner", "dep"),
        "ner": ("semeval", "dep"),
        "dep": ("semeval", "ner"),
    }
    uniform_ratios = {
        "semeval": {"low": (0.001, 0.005), "boundary": (0.0025, 0.01)},
        "ner": {
            "low": (0.0003125, 0.0015625),
            "boundary": (0.0004375, 0.0021875),
        },
        "dep": {"low": (0.00055, 0.00275), "boundary": (0.00070, 0.00350)},
    }
    rows = []
    for target, sources in source_map.items():
        for budget in ("low", "boundary"):
            uniform = read_run(target, own_name(target, budget, "uniform", seed))
            base_w, base_a = uniform_ratios[target][budget]
            for source in sources:
                for phase in ("pilot", "bopfix2"):
                    name = cross_name(
                        target, source, budget, seed, phase="" if phase == "pilot" else phase
                    )
                    value = read_run(target, name)
                    summary = json.loads(
                        (RUNS / target / name / "quant_summary.json").read_text(encoding="utf-8")
                    )
                    w_factors = [value / base_w for value in summary["weight_target_layer_ratios"]]
                    a_factors = [value / base_a for value in summary["activation_target_layer_ratios"]]
                    rows.append(
                        {
                            "target": target,
                            "source": source,
                            "budget": budget,
                            "phase": phase,
                            "major": f"{value['major']:.9f}",
                            "f1_micro": f"{value['f1_micro']:.9f}",
                            "measured_bop": f"{value['bop']:.9f}",
                            "relative_bop_error_pct": f"{100 * (value['bop'] / uniform['bop'] - 1):+.3f}",
                            "min_layer_factor": f"{min(w_factors + a_factors):.6f}",
                            "max_layer_factor": f"{max(w_factors + a_factors):.6f}",
                        }
                    )
    return rows


def heuristic_summary(seed=20260806):
    rows = []
    for task in ("semeval", "ner", "dep"):
        for budget in ("low", "boundary"):
            uniform = read_run(task, own_name(task, budget, "uniform", seed))
            for strategy in STRATEGIES:
                value = read_run(task, own_name(task, budget, strategy, seed))
                rows.append(
                    {
                        "target": task,
                        "budget": budget,
                        "strategy": strategy,
                        "major": f"{value['major']:.9f}",
                        "f1_micro": f"{value['f1_micro']:.9f}",
                        "measured_bop": f"{value['bop']:.9f}",
                        "delta_major_vs_uniform": f"{value['major'] - uniform['major']:+.9f}",
                        "delta_f1_vs_uniform": f"{value['f1_micro'] - uniform['f1_micro']:+.9f}",
                    }
                )
    return rows


def curve_correlations():
    curves = {}
    for task in ("semeval", "ner", "dep"):
        path = ANALYSIS / f"cross_probe_target_semeval_source_{task}_allocations.json"
        if not path.exists():
            path = ANALYSIS / f"cross_probe_target_ner_source_{task}_allocations.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        curves[task] = payload["smoothed_probe_scores_l1_l24"]
    rows = []
    for index, left in enumerate(curves):
        for right in list(curves)[index + 1 :]:
            rows.append(
                {"task_a": left, "task_b": right, "pearson": f"{pearson(curves[left], curves[right]):.9f}"}
            )
    return rows


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    boundary_rows, _ = semeval_boundary_summary()
    cross_rows = cross_summary()
    cross_phase_rows = cross_phase_summary()
    heuristic_rows = heuristic_summary()
    correlation_rows = curve_correlations()

    outputs = (
        (
            "semeval_boundary_six_strategy_multiseed_20260806_09",
            boundary_rows,
            list(boundary_rows[0]),
        ),
        ("cross_task_probe_single_seed_20260806", cross_rows, list(cross_rows[0])),
        (
            "cross_task_probe_pilot_vs_bopfix2_seed20260806",
            cross_phase_rows,
            list(cross_phase_rows[0]),
        ),
        ("own_task_six_strategy_seed20260806", heuristic_rows, list(heuristic_rows[0])),
        ("fixed_probe_curve_correlations", correlation_rows, list(correlation_rows[0])),
    )
    for name, rows, columns in outputs:
        write_tsv(ANALYSIS / f"{name}.tsv", rows, columns)
        (ANALYSIS / f"{name}.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        print(f"--- {name} ---")
        print("\t".join(columns))
        for row in rows:
            print("\t".join(str(row[column]) for column in columns))


if __name__ == "__main__":
    main()
