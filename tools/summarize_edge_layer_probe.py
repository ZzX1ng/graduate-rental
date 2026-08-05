#!/usr/bin/env python3
"""Summarize fixed-layer edge probing val metrics.

Run after one or more edge_lXX_msYYY_<task> jobs complete:

  python tools/summarize_edge_layer_probe.py

The script writes:
  base_exp/exp_edge/runs/bert-large-uncased/layer_probe_summary.tsv
"""

import argparse
import json
import re
from pathlib import Path


RUN_RE = re.compile(r"^edge_l(?P<layer>\d+)_ms(?P<max_seq_length>\d+)_(?P<task>.+)$")


def read_metrics(path: Path, task: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    task_data = data.get(task, {})
    metrics = task_data.get("metrics", {})
    minor = metrics.get("minor", {})
    return {
        "loss": task_data.get("loss", ""),
        "acc": minor.get("acc", ""),
        "f1_micro": minor.get("f1_micro", ""),
        "major": metrics.get("major", data.get("aggregated", "")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        default="base_exp/exp_edge/runs/bert-large-uncased",
        help="Root containing per-task run directories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output TSV path. Defaults to <runs-root>/layer_probe_summary.tsv.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output = Path(args.output) if args.output else runs_root / "layer_probe_summary.tsv"

    rows = []
    for metrics_path in sorted(runs_root.glob("*/edge_l*_ms*/val_metrics.json")):
        task = metrics_path.parent.parent.name
        run_name = metrics_path.parent.name
        match = RUN_RE.match(run_name)
        if not match:
            continue
        if match.group("task") != task:
            continue
        metrics = read_metrics(metrics_path, task=task)
        rows.append(
            {
                "task": task,
                "layer": int(match.group("layer")),
                "max_seq_length": int(match.group("max_seq_length")),
                "run_name": run_name,
                **metrics,
            }
        )

    rows.sort(key=lambda row: (row["task"], row["layer"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("task\tlayer\tmax_seq_length\trun_name\tloss\tacc\tf1_micro\tmajor\n")
        for row in rows:
            f.write(
                "\t".join(
                    str(row[key])
                    for key in [
                        "task",
                        "layer",
                        "max_seq_length",
                        "run_name",
                        "loss",
                        "acc",
                        "f1_micro",
                        "major",
                    ]
                )
                + "\n"
            )
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
