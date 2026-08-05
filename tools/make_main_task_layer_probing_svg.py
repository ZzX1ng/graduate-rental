#!/usr/bin/env python3
"""Plot full fixed-layer probing curves for the three selected main tasks."""

import csv
import json
from pathlib import Path


ROOT = Path("/cluster/home/zhangzx/my_project")
RUNS_ROOT = ROOT / "base_exp/exp_edge/runs/bert-large-uncased"
PLOT_DIR = ROOT / "base_exp/exp_edge/plots"
OUT_SVG = PLOT_DIR / "main_tasks_layer_probing_major.svg"
OUT_CSV = PLOT_DIR / "main_tasks_layer_probing_major.csv"

TASKS = [
    {
        "task": "ner",
        "label": "NER",
        "color": "#2563eb",
        "run_pattern": "edge_l{layer:02d}_ms256_ner_lr1e4",
        "baseline_run": "formal_e3_ms256_tner",
        "baseline_label": "baseline e3",
    },
    {
        "task": "dep",
        "label": "DEP",
        "color": "#059669",
        "run_pattern": "edge_l{layer:02d}_ms256_dep_lr1e4",
        "baseline_run": "formal_e3_ms256_retry1",
        "baseline_label": "baseline e3",
    },
    {
        "task": "semeval",
        "label": "SemEval",
        "color": "#7c3aed",
        "run_pattern": "edge_l{layer:02d}_ms256_semeval_e10_lr1e4_noearly",
        "baseline_run": "formal_e10_ms256",
        "baseline_label": "baseline e10",
    },
]


def read_major(task: str, run_name: str) -> float:
    path = RUNS_ROOT / task / run_name / "val_metrics.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return float(data["aggregated"])


def load_task(task_cfg):
    task = task_cfg["task"]
    values = []
    for layer in range(25):
        run_name = task_cfg["run_pattern"].format(layer=layer)
        values.append(read_major(task, run_name))
    baseline = read_major(task, task_cfg["baseline_run"])
    peak_layer = max(range(25), key=lambda idx: values[idx])
    return {
        **task_cfg,
        "values": values,
        "baseline": baseline,
        "peak_layer": peak_layer,
        "peak_value": values[peak_layer],
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def draw_panel(parts, panel, idx, dims):
    w = dims["w"]
    margin = dims["margin"]
    panel_h = dims["panel_h"]
    plot_w = dims["plot_w"]
    y_top = dims["y_top"](idx)
    left = margin["left"]
    right = w - margin["right"]
    color = panel["color"]
    values = panel["values"]
    baseline = panel["baseline"]

    y_min = min(min(values), baseline) - 0.015
    y_max = max(max(values), baseline) + 0.008
    if panel["task"] in {"ner", "dep"}:
        y_min = min(y_min, min(values) - 0.01)
    if panel["task"] == "semeval":
        y_min = min(y_min, 0.54)

    def sx(layer):
        return left + (layer / 24) * plot_w

    def sy(value):
        return y_top + (y_max - value) / (y_max - y_min) * panel_h

    parts.append(
        f'<rect x="{left}" y="{y_top}" width="{plot_w}" height="{panel_h}" '
        'fill="#fbfbfd" stroke="#d7dce2" stroke-width="1"/>'
    )

    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="13" fill="#53606d">{value:.3f}</text>')

    for layer in [0, 4, 8, 12, 16, 20, 24]:
        x = sx(layer)
        parts.append(f'<line x1="{x:.2f}" y1="{y_top+panel_h}" x2="{x:.2f}" y2="{y_top+panel_h+6}" stroke="#53606d" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{y_top+panel_h+23}" text-anchor="middle" font-size="13" fill="#53606d">{layer}</text>')

    by = sy(baseline)
    parts.append(f'<line x1="{left}" y1="{by:.2f}" x2="{right}" y2="{by:.2f}" stroke="#c2410c" stroke-width="2.1" stroke-dasharray="8 6"/>')
    parts.append(f'<text x="{right-4}" y="{by-7:.2f}" text-anchor="end" font-size="13" fill="#9a3412">{panel["baseline_label"]} {fmt(baseline)}</text>')

    pts = " ".join(f"{sx(layer):.2f},{sy(value):.2f}" for layer, value in enumerate(values))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
    for layer, value in enumerate(values):
        parts.append(f'<circle cx="{sx(layer):.2f}" cy="{sy(value):.2f}" r="3.9" fill="{color}" stroke="white" stroke-width="1.2"/>')

    peak_layer = panel["peak_layer"]
    px, py = sx(peak_layer), sy(panel["peak_value"])
    parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6.3" fill="#16a34a" stroke="white" stroke-width="1.8"/>')
    label_x = min(px + 14, right - 150)
    parts.append(f'<text x="{label_x:.2f}" y="{py-13:.2f}" font-size="13" fill="#166534">peak L{peak_layer}: {fmt(panel["peak_value"])}</text>')
    parts.append(f'<text x="{left}" y="{y_top-18}" font-size="21" font-weight="700" fill="#111827">{panel["label"]} fixed-layer probing</text>')


def main() -> None:
    panels = [load_task(task) for task in TASKS]
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "layer", "major", "baseline_run", "baseline_major", "peak_layer", "peak_major"])
        for panel in panels:
            for layer, value in enumerate(panel["values"]):
                writer.writerow(
                    [
                        panel["task"],
                        layer,
                        f"{value:.8f}",
                        panel["baseline_run"],
                        f"{panel['baseline']:.8f}",
                        panel["peak_layer"],
                        f"{panel['peak_value']:.8f}",
                    ]
                )

    w, h = 1250, 1180
    margin = {"left": 88, "right": 42, "top": 92, "bottom": 70}
    gap = 76
    panel_h = (h - margin["top"] - margin["bottom"] - gap * 2) / 3
    dims = {
        "w": w,
        "margin": margin,
        "panel_h": panel_h,
        "plot_w": w - margin["left"] - margin["right"],
        "y_top": lambda idx: margin["top"] + idx * (panel_h + gap),
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="625" y="38" text-anchor="middle" font-size="27" font-weight="700" fill="#111827">Selected Main Tasks: Fixed-Layer Probing Curves</text>',
        '<text x="625" y="64" text-anchor="middle" font-size="15" fill="#53606d">BERT-large hidden state index: 0 = embedding output, 24 = final Transformer layer output; metric = aggregated major</text>',
    ]
    for idx, panel in enumerate(panels):
        draw_panel(parts, panel, idx, dims)
    parts.append(f'<text x="{w/2}" y="{h-26}" text-anchor="middle" font-size="16" fill="#374151">BERT hidden state index</text>')
    parts.append(f'<text x="26" y="{h/2}" text-anchor="middle" font-size="16" fill="#374151" transform="rotate(-90 26 {h/2})">major</text>')
    parts.append("</svg>")
    OUT_SVG.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(OUT_SVG)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
