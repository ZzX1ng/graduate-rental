#!/usr/bin/env python3
"""Render Tenney-style probing CSV outputs as SVG figures."""

import argparse
import csv
import html
from pathlib import Path


COLORS = {
    "axis": "#111827",
    "grid": "#e5e7eb",
    "muted": "#6b7280",
    "bar": "#4c78a8",
    "highlight": "#d95f02",
    "cog": "#c51b7d",
    "line": "#1b9e77",
    "delta": "#7570b3",
    "negative": "#bdbdbd",
}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["layer"] = int(row["layer"])
        for key in [
            "scalar_mix_weight",
            "cumulative_major",
            "contextual_differential_from_previous",
            "positive_normalized_differential",
        ]:
            row[key] = float(row[key])
    return rows


def read_summary(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = dict(rows[0])
    for key in [
        "scalar_mix_center_of_gravity",
        "scalar_mix_entropy",
        "best_cumulative_layer",
        "best_cumulative_major",
        "p0_cumulative_major",
        "pL_cumulative_major",
        "full_scalar_mix_major",
    ]:
        if key in out and out[key] != "":
            try:
                out[key] = float(out[key])
            except ValueError:
                pass
    return out


def esc(text):
    return html.escape(str(text), quote=True)


def nice_ticks(max_value, count=5):
    if max_value <= 0:
        return [0.0]
    raw_step = max_value / count
    bases = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    step = bases[-1]
    for base in bases:
        if raw_step <= base:
            step = base
            break
    top = step
    while top < max_value:
        top += step
    ticks = []
    x = 0.0
    while x <= top + step / 2:
        ticks.append(round(x, 6))
        x += step
    return ticks


def render_scalar(rows, summary, task, out_path):
    width, height = 1040, 560
    left, right, top, bottom = 82, 32, 68, 480
    plot_w = width - left - right
    plot_h = bottom - top
    weights = [r["scalar_mix_weight"] for r in rows]
    max_w = max(max(weights), 1.0 / 25.0) * 1.12
    ticks = nice_ticks(max_w, count=5)
    y_top = max(ticks)
    bar_gap = 7.5
    slot = plot_w / len(rows)
    bar_w = slot - bar_gap
    cog = float(summary.get("scalar_mix_center_of_gravity", 0.0))
    top_rows = sorted(rows, key=lambda r: r["scalar_mix_weight"], reverse=True)[:5]

    def x_center(layer):
        return left + slot * (layer + 0.5)

    def y(value):
        return bottom - (value / y_top) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700" fill="{COLORS["axis"]}">{esc(task.upper())} Tenney-style Scalar-mix Layer Weights</text>',
        f'<text x="{width/2:.1f}" y="55" text-anchor="middle" font-family="Arial" font-size="13" fill="{COLORS["muted"]}">Layer 0 is embedding output; layers 1-24 are BERT-large Transformer layers</text>',
    ]

    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="{COLORS["muted"]}">{tick:.3f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{COLORS["axis"]}"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}" stroke="{COLORS["axis"]}"/>')

    top_layers = {r["layer"] for r in top_rows}
    for r in rows:
        layer = r["layer"]
        value = r["scalar_mix_weight"]
        x0 = left + slot * layer + bar_gap / 2
        yy = y(value)
        color = COLORS["highlight"] if layer in top_layers else COLORS["bar"]
        parts.append(f'<rect x="{x0:.2f}" y="{yy:.2f}" width="{bar_w:.2f}" height="{bottom-yy:.2f}" fill="{color}"/>')

    uniform = 1.0 / 25.0
    uniform_y = y(uniform)
    parts.append(f'<line x1="{left}" y1="{uniform_y:.2f}" x2="{width-right}" y2="{uniform_y:.2f}" stroke="{COLORS["muted"]}" stroke-dasharray="6,5" stroke-width="1.5"/>')
    parts.append(f'<text x="{width-right-6}" y="{uniform_y-7:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="{COLORS["muted"]}">uniform 1/25 = 0.040</text>')

    cog_x = x_center(cog)
    parts.append(f'<line x1="{cog_x:.2f}" y1="{top}" x2="{cog_x:.2f}" y2="{bottom}" stroke="{COLORS["cog"]}" stroke-width="2"/>')
    parts.append(f'<text x="{cog_x+8:.2f}" y="{top+18}" font-family="Arial" font-size="12" fill="{COLORS["cog"]}">COG={cog:.2f}</text>')

    for layer in range(0, 25, 2):
        xx = x_center(layer)
        parts.append(f'<line x1="{xx:.2f}" y1="{bottom}" x2="{xx:.2f}" y2="{bottom+5}" stroke="{COLORS["axis"]}"/>')
        parts.append(f'<text x="{xx:.2f}" y="{bottom+22}" text-anchor="middle" font-family="Arial" font-size="12" fill="{COLORS["axis"]}">{layer}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="536" text-anchor="middle" font-family="Arial" font-size="14" fill="{COLORS["axis"]}">BERT-large layer</text>')
    parts.append(f'<text transform="translate(22,{(top+bottom)/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14" fill="{COLORS["axis"]}">Scalar-mix weight</text>')
    label = ", ".join([f'L{r["layer"]}={r["scalar_mix_weight"]:.4f}' for r in top_rows])
    parts.append(f'<text x="{left}" y="552" font-family="Arial" font-size="12" fill="{COLORS["muted"]}">Top weights: {esc(label)}</text>')
    parts.append("</svg>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def render_cumulative(rows, summary, task, out_path):
    width, height = 1080, 720
    left, right = 84, 34
    top1, bottom1 = 70, 410
    top2, bottom2 = 468, 646
    plot_w = width - left - right
    slot = plot_w / len(rows)

    scores = [r["cumulative_major"] for r in rows]
    deltas = [r["contextual_differential_from_previous"] for r in rows]
    min_s = min(scores)
    max_s = max(scores)
    pad = max((max_s - min_s) * 0.12, 0.003)
    y_min = min_s - pad
    y_max = max_s + pad
    d_abs = max(max(abs(x) for x in deltas), 0.001)
    d_min, d_max = -d_abs * 1.12, d_abs * 1.12
    best_layer = int(summary.get("best_cumulative_layer", max(rows, key=lambda r: r["cumulative_major"])["layer"]))
    best_score = float(summary.get("best_cumulative_major", max_s))

    def x_center(layer):
        return left + slot * (layer + 0.5)

    def y_score(value):
        return bottom1 - ((value - y_min) / (y_max - y_min)) * (bottom1 - top1)

    def y_delta(value):
        return bottom2 - ((value - d_min) / (d_max - d_min)) * (bottom2 - top2)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700" fill="{COLORS["axis"]}">{esc(task.upper())} Tenney-style Cumulative Probing</text>',
        f'<text x="{width/2:.1f}" y="55" text-anchor="middle" font-family="Arial" font-size="13" fill="{COLORS["muted"]}">Top: cumulative score using layers 0..L; Bottom: incremental contribution from adding layer L</text>',
    ]

    for i in range(6):
        tick = y_min + (y_max - y_min) * i / 5.0
        yy = y_score(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="{COLORS["grid"]}"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="{COLORS["muted"]}">{tick:.3f}</text>')
    parts.append(f'<line x1="{left}" y1="{top1}" x2="{left}" y2="{bottom1}" stroke="{COLORS["axis"]}"/>')
    parts.append(f'<line x1="{left}" y1="{bottom1}" x2="{width-right}" y2="{bottom1}" stroke="{COLORS["axis"]}"/>')

    points = " ".join(f'{x_center(r["layer"]):.2f},{y_score(r["cumulative_major"]):.2f}' for r in rows)
    parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS["line"]}" stroke-width="2.5"/>')
    for r in rows:
        layer = r["layer"]
        yy = y_score(r["cumulative_major"])
        radius = 5.5 if layer == best_layer else 3.6
        color = COLORS["highlight"] if layer == best_layer else COLORS["line"]
        parts.append(f'<circle cx="{x_center(layer):.2f}" cy="{yy:.2f}" r="{radius}" fill="{color}"/>')
    best_x = x_center(best_layer)
    best_y = y_score(best_score)
    parts.append(f'<text x="{min(best_x+8, width-right-150):.2f}" y="{best_y-10:.2f}" font-family="Arial" font-size="12" fill="{COLORS["highlight"]}">best L{best_layer}={best_score:.4f}</text>')
    parts.append(f'<text transform="translate(24,{(top1+bottom1)/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14" fill="{COLORS["axis"]}">Cumulative major</text>')

    for i in range(5):
        tick = d_min + (d_max - d_min) * i / 4.0
        yy = y_delta(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="{COLORS["grid"]}"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="{COLORS["muted"]}">{tick:+.3f}</text>')
    parts.append(f'<line x1="{left}" y1="{top2}" x2="{left}" y2="{bottom2}" stroke="{COLORS["axis"]}"/>')
    parts.append(f'<line x1="{left}" y1="{bottom2}" x2="{width-right}" y2="{bottom2}" stroke="{COLORS["axis"]}"/>')
    zero_y = y_delta(0.0)
    parts.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="{COLORS["muted"]}" stroke-width="1"/>')

    bar_gap = 10
    bar_w = slot - bar_gap
    for r in rows:
        layer = r["layer"]
        value = r["contextual_differential_from_previous"]
        x0 = left + slot * layer + bar_gap / 2
        yy = y_delta(value)
        color = COLORS["delta"] if value >= 0 else COLORS["negative"]
        y0 = min(yy, zero_y)
        h = abs(zero_y - yy)
        parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" opacity="0.82"/>')

    for layer in range(0, 25, 2):
        xx = x_center(layer)
        parts.append(f'<line x1="{xx:.2f}" y1="{bottom2}" x2="{xx:.2f}" y2="{bottom2+5}" stroke="{COLORS["axis"]}"/>')
        parts.append(f'<text x="{xx:.2f}" y="{bottom2+22}" text-anchor="middle" font-family="Arial" font-size="12" fill="{COLORS["axis"]}">{layer}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="700" text-anchor="middle" font-family="Arial" font-size="14" fill="{COLORS["axis"]}">BERT-large layer</text>')
    parts.append(f'<text transform="translate(24,{(top2+bottom2)/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14" fill="{COLORS["axis"]}">Delta major</text>')
    parts.append("</svg>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def write_html(task, scalar_name, cumulative_name, out_path):
    title = f"{task.upper()} Tenney-style probing figures"
    text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{esc(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    img {{ display: block; max-width: 100%; margin: 16px 0 32px; border: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <h2>Scalar-mix Layer Weights</h2>
  <img src="{esc(scalar_name)}" alt="Scalar-mix layer weights">
  <h2>Cumulative Probing</h2>
  <img src="{esc(cumulative_name)}" alt="Cumulative probing and layer delta">
</body>
</html>
"""
    Path(out_path).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    task = args.task
    input_dir = Path(args.input_dir)
    rows = read_rows(input_dir / "tenney_figure2_layerwise.csv")
    summary = read_summary(input_dir / "tenney_figure1_summary.csv")
    scalar_name = f"{task}_tenney_scalar_mix_weights.svg"
    cumulative_name = f"{task}_tenney_cumulative_probe.svg"
    html_name = f"{task}_tenney_figures.html"
    render_scalar(rows, summary, task, input_dir / scalar_name)
    render_cumulative(rows, summary, task, input_dir / cumulative_name)
    write_html(task, scalar_name, cumulative_name, input_dir / html_name)
    print("wrote", input_dir / scalar_name)
    print("wrote", input_dir / cumulative_name)
    print("wrote", input_dir / html_name)


if __name__ == "__main__":
    main()
