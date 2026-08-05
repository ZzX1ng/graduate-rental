import json
from pathlib import Path


ROOT = Path("/cluster/home/zhangzx/my_project")
RUNS = ROOT / "base_exp/exp_edge/runs/bert-large-uncased/semeval"
OUT = ROOT / "base_exp/exp_edge/plots/semeval_layer_probing_major.svg"


def read_major(run_name: str) -> float:
    with (RUNS / run_name / "val_metrics.json").open() as f:
        data = json.load(f)
    return float(data["semeval"]["metrics"]["major"])


values = {
    layer: read_major(f"edge_l{layer:02d}_ms256_semeval_e10_lr1e4_noearly")
    for layer in range(25)
}
baseline = read_major("formal_e10_ms256")

W, H = 1200, 520
M = {"left": 84, "right": 38, "top": 92, "bottom": 78}
plot_w = W - M["left"] - M["right"]
plot_h = H - M["top"] - M["bottom"]


def sx(layer: int) -> float:
    return M["left"] + (layer / 24) * plot_w


y_min = min(min(values.values()), baseline) - 0.018
y_max = max(max(values.values()), baseline) + 0.010


def sy(v: float) -> float:
    return M["top"] + (y_max - v) / (y_max - y_min) * plot_h


def fmt(v: float) -> str:
    return f"{v:.4f}"


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<text x="600" y="38" text-anchor="middle" font-size="26" font-weight="700" fill="#111827">SemEval Layer Probing Major Metric with Baseline</text>',
    '<text x="600" y="64" text-anchor="middle" font-size="15" fill="#53606d">BERT-large hidden state index: 0 = embedding output, 24 = final Transformer layer output</text>',
    f'<rect x="{M["left"]}" y="{M["top"]}" width="{plot_w}" height="{plot_h}" fill="#fbfbfd" stroke="#d7dce2" stroke-width="1"/>',
]

for i in range(5):
    val = y_min + (y_max - y_min) * i / 4
    y = sy(val)
    parts.append(f'<line x1="{M["left"]}" y1="{y:.2f}" x2="{W-M["right"]}" y2="{y:.2f}" stroke="#e6e9ee" stroke-width="1"/>')
    parts.append(f'<text x="{M["left"]-12}" y="{y+4:.2f}" text-anchor="end" font-size="14" fill="#53606d">{val:.3f}</text>')

for layer in [0, 4, 8, 12, 16, 20, 24]:
    x = sx(layer)
    parts.append(f'<line x1="{x:.2f}" y1="{M["top"]+plot_h}" x2="{x:.2f}" y2="{M["top"]+plot_h+6}" stroke="#53606d" stroke-width="1"/>')
    parts.append(f'<text x="{x:.2f}" y="{M["top"]+plot_h+25}" text-anchor="middle" font-size="14" fill="#53606d">{layer}</text>')

by = sy(baseline)
parts.append(f'<line x1="{M["left"]}" y1="{by:.2f}" x2="{W-M["right"]}" y2="{by:.2f}" stroke="#c2410c" stroke-width="2.2" stroke-dasharray="9 6"/>')
parts.append(f'<text x="{W-M["right"]-4}" y="{by-8:.2f}" text-anchor="end" font-size="14" fill="#9a3412">baseline e10 {fmt(baseline)}</text>')

pts = " ".join(f"{sx(layer):.2f},{sy(values[layer]):.2f}" for layer in range(25))
parts.append(f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')

for layer, value in values.items():
    parts.append(f'<circle cx="{sx(layer):.2f}" cy="{sy(value):.2f}" r="4.2" fill="#2563eb" stroke="white" stroke-width="1.3"/>')
    parts.append(f'<text x="{sx(layer):.2f}" y="{sy(value)-9:.2f}" text-anchor="middle" font-size="11" fill="#1f2937">{layer}</text>')

peak_layer = max(values, key=values.get)
px, py = sx(peak_layer), sy(values[peak_layer])
parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6.5" fill="#16a34a" stroke="white" stroke-width="1.8"/>')
parts.append(f'<text x="{min(px + 16, W - M["right"] - 150):.2f}" y="{py-14:.2f}" font-size="14" fill="#166534">peak L{peak_layer}: {fmt(values[peak_layer])}</text>')

parts.append(f'<text x="{W/2}" y="{H-26}" text-anchor="middle" font-size="16" fill="#374151">BERT hidden state index</text>')
parts.append(f'<text x="26" y="{H/2}" text-anchor="middle" font-size="16" fill="#374151" transform="rotate(-90 26 {H/2})">major</text>')
parts.append('<text x="84" y="488" font-size="13" fill="#6b7280">Note: e10 no-early probing was run for all BERT-large hidden state indices 0-24.</text>')
parts.append("</svg>")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts) + "\n")
print(OUT)
