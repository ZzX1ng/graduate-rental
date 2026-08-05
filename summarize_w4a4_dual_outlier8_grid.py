#!/usr/bin/env python3
import csv
import glob
import json
import os


PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", "/root/autodl-tmp/master-gra/my_project"
)
RUN_GLOB = os.path.join(
    PROJECT_ROOT,
    "base_exp/exp_edge/runs/bert-large-uncased/semeval",
    "ptq_oa_w4a4_dualout8_*_ms256_semeval_e10",
    "quant_summary.json",
)
OUTPUT = os.path.join(
    PROJECT_ROOT,
    "base_exp/exp_edge/analysis/ptq/semeval_w4a4_dual_outlier8_grid_20260805.tsv",
)


rows = []
for path in glob.glob(RUN_GLOB):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    weight_target = data.get("weight_target_outlier_ratio")
    activation_target = data.get("activation_target_outlier_ratio")
    runtime = data.get("runtime_totals") or {}
    task_metrics = data["val_metrics"]["semeval"]["metrics"]
    minor = task_metrics["minor"]
    rows.append(
        {
            "run_name": os.path.basename(os.path.dirname(path)),
            "weight_target_ratio": weight_target,
            "activation_target_ratio": activation_target,
            "weight_actual_ratio": runtime.get("weight_outlier_ratio"),
            "activation_actual_ratio": runtime.get("activation_outlier_ratio"),
            "major": task_metrics["major"],
            "acc": minor["acc"],
            "f1_micro": minor["f1_micro"],
            "normalized_bop_overhead": runtime.get("normalized_bop_overhead"),
        }
    )

rows.sort(key=lambda row: (row["weight_target_ratio"], row["activation_target_ratio"]))
if len(rows) != 30:
    raise SystemExit(f"expected 30 completed combinations, found {len(rows)}")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

pareto = []
for candidate in rows:
    dominated = any(
        other["normalized_bop_overhead"] <= candidate["normalized_bop_overhead"]
        and other["major"] >= candidate["major"]
        and (
            other["normalized_bop_overhead"] < candidate["normalized_bop_overhead"]
            or other["major"] > candidate["major"]
        )
        for other in rows
    )
    if not dominated:
        pareto.append(candidate)
pareto.sort(key=lambda row: row["normalized_bop_overhead"])

weights = sorted({row["weight_target_ratio"] for row in rows})
activations = sorted({row["activation_target_ratio"] for row in rows})
lookup = {
    (row["weight_target_ratio"], row["activation_target_ratio"]): row
    for row in rows
}

print(f"OUTPUT={OUTPUT}")
print(f"COUNT={len(rows)}")
best = max(rows, key=lambda row: row["major"])
print(
    "BEST="
    f"W{best['weight_target_ratio'] * 100:g}%/A{best['activation_target_ratio'] * 100:g}% "
    f"major={best['major']:.6f} overhead={best['normalized_bop_overhead']:.6f}"
)
print("MAJOR_MATRIX")
print("W\\A\t" + "\t".join(f"{ratio * 100:g}%" for ratio in activations))
for weight in weights:
    values = [lookup[(weight, activation)]["major"] for activation in activations]
    print(f"{weight * 100:g}%\t" + "\t".join(f"{value:.6f}" for value in values))
print("PARETO")
for row in pareto:
    print(
        f"W{row['weight_target_ratio'] * 100:g}%/A{row['activation_target_ratio'] * 100:g}%\t"
        f"major={row['major']:.6f}\toverhead={row['normalized_bop_overhead']:.6f}"
    )

for threshold_name, threshold in (("W8A8", 0.876101), ("FP32_MINUS_0P005", 0.878601)):
    eligible = [row for row in rows if row["major"] >= threshold]
    chosen = min(eligible, key=lambda row: row["normalized_bop_overhead"])
    print(
        f"MIN_COST_{threshold_name}="
        f"W{chosen['weight_target_ratio'] * 100:g}%/A{chosen['activation_target_ratio'] * 100:g}% "
        f"major={chosen['major']:.6f} overhead={chosen['normalized_bop_overhead']:.6f}"
    )
