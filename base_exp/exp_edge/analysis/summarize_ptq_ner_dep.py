import json
from pathlib import Path

root = Path("base_exp/exp_edge/runs/bert-large-uncased")

jobs = {
    "ner": {
        "base": "formal_e3_ms256_tner",
        "runs": [
            ("W8A8", "18662", "ptq_uniform_w8a8_ms256_ner_e3"),
            ("W6A8", "18663", "ptq_uniform_w6a8_ms256_ner_e3"),
            ("W8A6", "18664", "ptq_uniform_w8a6_ms256_ner_e3"),
            ("W6A6", "18665", "ptq_uniform_w6a6_ms256_ner_e3"),
            ("W4A8", "18666", "ptq_uniform_w4a8_ms256_ner_e3"),
            ("W8A4", "18667", "ptq_uniform_w8a4_ms256_ner_e3"),
            ("W4A4", "18668", "ptq_uniform_w4a4_ms256_ner_e3"),
        ],
    },
    "dep": {
        "base": "formal_e3_ms256_retry1",
        "runs": [
            ("W8A8", "18669", "ptq_uniform_w8a8_ms256_dep_e3_retry1"),
            ("W6A8", "18670", "ptq_uniform_w6a8_ms256_dep_e3_retry1"),
            ("W8A6", "18671", "ptq_uniform_w8a6_ms256_dep_e3_retry1"),
            ("W6A6", "18672", "ptq_uniform_w6a6_ms256_dep_e3_retry1"),
            ("W4A8", "18673", "ptq_uniform_w4a8_ms256_dep_e3_retry1"),
            ("W8A4", "18674", "ptq_uniform_w8a4_ms256_dep_e3_retry1"),
            ("W4A4", "18675", "ptq_uniform_w4a4_ms256_dep_e3_retry1"),
        ],
    },
}


def load_metrics(path):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    task_metrics = None
    for key in ("ner", "dep", "semeval"):
        if key in metrics:
            task_metrics = metrics[key]
            break
    if task_metrics is not None:
        nested = task_metrics.get("metrics", {})
        minor = nested.get("minor", {})
        return {
            "major": nested.get("major", metrics.get("aggregated")),
            "acc": minor.get("acc"),
            "f1_micro": minor.get("f1_micro"),
            "loss": task_metrics.get("loss"),
        }
    major = metrics.get("major", metrics.get("acc_and_f1_micro", metrics.get("aggregated")))
    return {
        "major": major,
        "acc": metrics.get("acc"),
        "f1_micro": metrics.get("f1_micro"),
        "loss": metrics.get("loss"),
    }


for task, spec in jobs.items():
    base = load_metrics(root / task / spec["base"] / "val_metrics.json")
    print(f"## {task} baseline {spec['base']}")
    print(f"baseline_major={base['major']:.6f}")
    print("| 配置 | Job ID | major | Δmajor | acc | f1_micro | loss |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| `Baseline` | - | {base['major']:.6f} | +0.000000 | "
        f"{base['acc']:.6f} | {base['f1_micro']:.6f} | {base['loss']:.6f} |"
    )
    for cfg, job_id, run_name in spec["runs"]:
        metrics = load_metrics(root / task / run_name / "val_metrics.json")
        print(
            f"| `{cfg}` | `{job_id}` | {metrics['major']:.6f} | "
            f"{metrics['major'] - base['major']:+.6f} | {metrics['acc']:.6f} | "
            f"{metrics['f1_micro']:.6f} | {metrics['loss']:.6f} |"
        )
    print()
