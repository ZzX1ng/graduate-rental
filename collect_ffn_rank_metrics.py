import json
from pathlib import Path


ROOT = Path("/cluster/home/zhangzx/my_project/base_exp/exp_edge/runs/bert-large-uncased/spr2")
RUNS = [
    ("rank512", "ffn_rank512_e3_ms256_spr2_uniform_exnode6_retry1"),
    ("rank1024", "ffn_rank1024_e3_ms256_spr2_uniform_exnode6_retry1"),
    ("rank2048", "ffn_rank2048_e3_ms256_spr2_uniform_exnode6_retry1"),
]

print("setting\tloss\tacc\tf1_micro\tmajor")
for name, run_name in RUNS:
    path = ROOT / run_name / "val_metrics.json"
    if not path.exists():
        print(f"{name}\tMISSING\t{path}")
        continue
    data = json.loads(path.read_text())
    task = data["spr2"]
    minor = task["metrics"]["minor"]
    print(
        f"{name}\t{task['loss']:.5f}\t{minor['acc']:.5f}\t"
        f"{minor['f1_micro']:.5f}\t{task['metrics']['major']:.5f}"
    )
