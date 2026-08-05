#!/usr/bin/env python3
"""Render fixed-layer probing curves for SemEval, NER, and DEP on one chart."""

import argparse
import csv
import html
from pathlib import Path


COLORS = {
    "semeval": "#4c78a8",
    "ner": "#d95f02",
    "dep": "#1b9e77",
}
LABELS = {
    "semeval": "SemEval",
    "ner": "NER",
    "dep": "DEP",
}
ORDER = ["semeval", "ner", "dep"]


def esc(value):
    return html.escape(str(value), quote=True)


def read_rows(path):
    by_task = {task: [] for task in ORDER}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task = row["task"]
            if task not in by_task:
                continue
            by_task[task].append(
                {
                    "layer": int(row["layer"]),
                    "major": float(row["major"]),
                    "peak_layer": int(row["peak_layer"]),
                    "peak_major": float(row["peak_major"]),
                    "baseline_major": float(row["baseline_major"]),
                }
            )
    for task in ORDER:
        by_task[task].sort(key=lambda r: r["layer"])
    return by_task


def render_svg(by_task, out_path):
    width, height = 1080, 620
    left, right, top, bottom = 86, 34, 74, 500
    plot_w = width - left - right
    plot_h = bottom - top
    values = [r["major"] for rows in by_task.values() for r in rows]
    ymin = min(values) - 0.04
    ymax = max(values) + 0.02

    def x(layer):
        return left + layer * plot_w / 24

    def y(value):
        return bottom - ((value - ymin) / (ymax - ymin)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700" fill="#111827">Fixed-layer Probing Across Tasks</text>',
        f'<text x="{width/2:.1f}" y="58" text-anchor="middle" font-family="Arial" font-size="13" fill="#6b7280">Three task series: SemEval, NER, DEP. Layer 0 is embedding output; layers 1-24 are Transformer layers.</text>',
    ]

    for i in range(6):
        tick = ymin + (ymax - ymin) * i / 5
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#6b7280">{tick:.3f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}" stroke="#111827"/>')

    legend_x, legend_y = left + 10, top + 16
    for idx, task in enumerate(ORDER):
        rows = by_task[task]
        color = COLORS[task]
        points = " ".join(f'{x(r["layer"]):.2f},{y(r["major"]):.2f}' for r in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.8"/>')
        peak_layer = rows[0]["peak_layer"]
        for r in rows:
            radius = 5.2 if r["layer"] == peak_layer else 3.0
            parts.append(f'<circle cx="{x(r["layer"]):.2f}" cy="{y(r["major"]):.2f}" r="{radius}" fill="{color}" stroke="white" stroke-width="1.0"/>')
        ly = legend_y + idx * 24
        peak = next(r for r in rows if r["layer"] == peak_layer)
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+28}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+38}" y="{ly+4}" font-family="Arial" font-size="13" fill="#111827">{esc(LABELS[task])}: peak L{peak_layer}={peak["major"]:.4f}</text>')

    for layer in range(0, 25, 2):
        xx = x(layer)
        parts.append(f'<line x1="{xx:.2f}" y1="{bottom}" x2="{xx:.2f}" y2="{bottom+5}" stroke="#111827"/>')
        parts.append(f'<text x="{xx:.2f}" y="{bottom+22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">{layer}</text>')

    parts.append(f'<text x="{width/2:.1f}" y="550" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">BERT-large layer</text>')
    parts.append(f'<text transform="translate(24,{(top+bottom)/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">Fixed-layer major</text>')
    parts.append(f'<text x="{left}" y="586" font-family="Arial" font-size="12" fill="#6b7280">Raw task major scores are plotted; compare curve shapes and peak layer positions more than absolute task difficulty.</text>')
    parts.append("</svg>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def write_html(out_dir):
    text = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Combined fixed-layer probing figure</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #111827; }
    img { display: block; max-width: 100%; margin: 16px 0 32px; border: 1px solid #e5e7eb; }
  </style>
</head>
<body>
  <h1>Combined fixed-layer probing figure</h1>
  <img src="combined_fixed_layer_probing_major.svg" alt="Fixed-layer probing curves across tasks">
</body>
</html>
"""
    (out_dir / "combined_fixed_layer_probing.html").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/cluster/home/zhangzx/my_project")
    parser.add_argument(
        "--input-csv",
        default="base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="base_exp/exp_edge/analysis/fixed_layer_probe/combined_three_tasks",
    )
    args = parser.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    by_task = read_rows(root / args.input_csv)
    svg_path = out_dir / "combined_fixed_layer_probing_major.svg"
    render_svg(by_task, svg_path)
    write_html(out_dir)
    print("wrote", svg_path)
    print("wrote", out_dir / "combined_fixed_layer_probing.html")


if __name__ == "__main__":
    main()
