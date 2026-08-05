#!/usr/bin/env python3
"""Render combined Tenney-style probing figures for SemEval, NER, and DEP."""

import argparse
import csv
import html
from pathlib import Path


TASKS = [
    ("semeval", "SemEval", "base_exp/exp_edge/analysis/tenney_probe/semeval_e10_lr1e4_noearly"),
    ("ner", "NER", "base_exp/exp_edge/analysis/tenney_probe/ner_e3_lr1e4_noearly"),
    ("dep", "DEP", "base_exp/exp_edge/analysis/tenney_probe/dep_e3_lr1e4_noearly"),
]

COLORS = {
    "SemEval": "#4c78a8",
    "NER": "#d95f02",
    "DEP": "#1b9e77",
}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        out.append(
            {
                "layer": int(row["layer"]),
                "scalar_mix_weight": float(row["scalar_mix_weight"]),
                "cumulative_major": float(row["cumulative_major"]),
            }
        )
    return out


def esc(value):
    return html.escape(str(value), quote=True)


def tick_values(vmin, vmax, count=5):
    if vmax <= vmin:
        return [vmin]
    return [vmin + (vmax - vmin) * i / count for i in range(count + 1)]


def polyline_points(rows, value_key, left, bottom, plot_w, plot_h, ymin, ymax):
    x_step = plot_w / 24
    points = []
    for row in rows:
        x = left + row["layer"] * x_step
        y = bottom - ((row[value_key] - ymin) / (ymax - ymin)) * plot_h
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def render_line_chart(series, value_key, title, ylabel, out_path, y_min=None, y_max=None, note=""):
    width, height = 1080, 620
    left, right, top, bottom = 86, 34, 74, 500
    plot_w = width - left - right
    plot_h = bottom - top
    values = [row[value_key] for _, _, rows in series for row in rows]
    ymin = min(values) if y_min is None else y_min
    ymax = max(values) if y_max is None else y_max
    pad = (ymax - ymin) * 0.08 if ymax > ymin else 0.01
    ymin = ymin - pad if y_min is None else y_min
    ymax = ymax + pad if y_max is None else y_max
    if ymax <= ymin:
        ymax = ymin + 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{esc(title)}</text>',
        f'<text x="{width/2:.1f}" y="58" text-anchor="middle" font-family="Arial" font-size="13" fill="#6b7280">Three task series: SemEval, NER, DEP. Layer 0 is embedding output; layers 1-24 are Transformer layers.</text>',
    ]

    for tick in tick_values(ymin, ymax, count=5):
        y = bottom - ((tick - ymin) / (ymax - ymin)) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#6b7280">{tick:.3f}</text>')

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}" stroke="#111827"/>')

    if value_key == "scalar_mix_weight":
        uniform = 1.0 / 25.0
        y = bottom - ((uniform - ymin) / (ymax - ymin)) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#9ca3af" stroke-dasharray="6,5"/>')
        parts.append(f'<text x="{width-right-6}" y="{y-7:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#6b7280">uniform 1/25 = 0.040</text>')

    legend_x = left + 10
    legend_y = top + 16
    for idx, (_, label, rows) in enumerate(series):
        color = COLORS[label]
        points = polyline_points(rows, value_key, left, bottom, plot_w, plot_h, ymin, ymax)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.8"/>')
        for row in rows:
            x = left + row["layer"] * plot_w / 24
            y = bottom - ((row[value_key] - ymin) / (ymax - ymin)) * plot_h
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.0" fill="{color}"/>')
        ly = legend_y + idx * 24
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+28}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+38}" y="{ly+4}" font-family="Arial" font-size="13" fill="#111827">{esc(label)}</text>')

    for layer in range(0, 25, 2):
        x = left + layer * plot_w / 24
        parts.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom+5}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom+22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">{layer}</text>')

    parts.append(f'<text x="{width/2:.1f}" y="550" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">BERT-large layer</text>')
    parts.append(f'<text transform="translate(24,{(top+bottom)/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">{esc(ylabel)}</text>')
    if note:
        parts.append(f'<text x="{left}" y="586" font-family="Arial" font-size="12" fill="#6b7280">{esc(note)}</text>')
    parts.append("</svg>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def write_html(out_dir, figures):
    lines = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>Combined Tenney-style probing figures</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#111827}img{display:block;max-width:100%;margin:16px 0 32px;border:1px solid #e5e7eb}</style>",
        "</head><body>",
        "<h1>Combined Tenney-style probing figures</h1>",
    ]
    for title, name in figures:
        lines.append(f"<h2>{esc(title)}</h2>")
        lines.append(f'<img src="{esc(name)}" alt="{esc(title)}">')
    lines.append("</body></html>")
    (out_dir / "combined_tenney_figures.html").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/cluster/home/zhangzx/my_project")
    parser.add_argument(
        "--output-dir",
        default="base_exp/exp_edge/analysis/tenney_probe/combined_three_tasks",
    )
    args = parser.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    series = []
    for task, label, rel_dir in TASKS:
        rows = read_rows(root / rel_dir / "tenney_figure2_layerwise.csv")
        series.append((task, label, rows))

    scalar_name = "combined_tenney_scalar_mix_weights.svg"
    cumulative_name = "combined_tenney_cumulative_major.svg"
    render_line_chart(
        series,
        "scalar_mix_weight",
        "Tenney-style Scalar-mix Weights Across Tasks",
        "Scalar-mix weight",
        out_dir / scalar_name,
        y_min=0.0,
        note="Scalar-mix weights are directly comparable as learned layer-selection distributions.",
    )
    render_line_chart(
        series,
        "cumulative_major",
        "Tenney-style Cumulative Probing Across Tasks",
        "Cumulative major",
        out_dir / cumulative_name,
        note="Raw task major scores are plotted; compare curve shapes more than absolute task difficulty.",
    )
    write_html(
        out_dir,
        [
            ("Scalar-mix weights", scalar_name),
            ("Cumulative major", cumulative_name),
        ],
    )
    print("wrote", out_dir / scalar_name)
    print("wrote", out_dir / cumulative_name)
    print("wrote", out_dir / "combined_tenney_figures.html")


if __name__ == "__main__":
    main()
