import json
from pathlib import Path


ROOT = Path("/cluster/home/zhangzx/my_project/base_exp/exp_edge/runs/bert-large-uncased")

runs = [
    ("dep aware12-17", "dep", "gelu_pwl1_l00_11_pwl2_l12_17_pwl1_l18_23_e3_ms256_dep_avg125_aware12_17"),
    ("dep early6", "dep", "gelu_pwl2_l00_05_pwl1_l06_23_e3_ms256_dep_avg125_early6"),
    ("dep late6", "dep", "gelu_pwl1_l00_17_pwl2_l18_23_e3_ms256_dep_avg125_late6"),
    ("dep aware15-16", "dep", "gelu_pwl1_l00_14_pwl2_l15_16_pwl1_l17_23_e3_ms256_dep_avg108_aware15_16"),
    ("dep early2", "dep", "gelu_pwl2_l00_01_pwl1_l02_23_e3_ms256_dep_avg108_early2"),
    ("dep late2", "dep", "gelu_pwl1_l00_21_pwl2_l22_23_e3_ms256_dep_avg108_late2"),
    ("spr2 single l01", "spr2", "gelu_pwl2_l00_00_pwl4_l01_pwl2_l02_23_e3_ms256_spr2_single_l01_exnode6"),
    ("spr2 single l02", "spr2", "gelu_pwl2_l00_01_pwl4_l02_pwl2_l03_23_e3_ms256_spr2_single_l02_exnode6"),
    ("spr2 single l03", "spr2", "gelu_pwl2_l00_02_pwl4_l03_pwl2_l04_23_e3_ms256_spr2_single_l03_exnode6"),
    ("spr2 single l04", "spr2", "gelu_pwl2_l00_03_pwl4_l04_pwl2_l05_23_e3_ms256_spr2_single_l04_exnode6"),
    ("spr2 single l05", "spr2", "gelu_pwl2_l00_04_pwl4_l05_pwl2_l06_23_e3_ms256_spr2_single_l05_exnode6"),
    ("spr2 single l06", "spr2", "gelu_pwl2_l00_05_pwl4_l06_pwl2_l07_23_e3_ms256_spr2_single_l06_exnode6"),
    ("spr2 single l07", "spr2", "gelu_pwl2_l00_06_pwl4_l07_pwl2_l08_23_e3_ms256_spr2_single_l07_exnode6"),
    ("spr2 single l08", "spr2", "gelu_pwl2_l00_07_pwl4_l08_pwl2_l09_23_e3_ms256_spr2_single_l08_exnode6"),
    ("spr2 single l09", "spr2", "gelu_pwl2_l00_08_pwl4_l09_pwl2_l10_23_e3_ms256_spr2_single_l09_exnode6"),
    ("spr2 single l10", "spr2", "gelu_pwl2_l00_09_pwl4_l10_pwl2_l11_23_e3_ms256_spr2_single_l10_exnode6"),
    ("spr2 single l11", "spr2", "gelu_pwl2_l00_10_pwl4_l11_pwl2_l12_23_e3_ms256_spr2_single_l11_exnode6"),
    ("spr2 single l12", "spr2", "gelu_pwl2_l00_11_pwl4_l12_pwl2_l13_23_e3_ms256_spr2_single_l12_exnode6"),
    ("spr2 single l13", "spr2", "gelu_pwl2_l00_12_pwl4_l13_pwl2_l14_23_e3_ms256_spr2_single_l13_exnode6"),
    ("spr2 single l14", "spr2", "gelu_pwl2_l00_13_pwl4_l14_pwl2_l15_23_e3_ms256_spr2_single_l14_exnode6"),
    ("spr2 single l15", "spr2", "gelu_pwl2_l00_14_pwl4_l15_pwl2_l16_23_e3_ms256_spr2_single_l15_exnode6"),
    ("spr2 single l16", "spr2", "gelu_pwl2_l00_15_pwl4_l16_pwl2_l17_23_e3_ms256_spr2_single_l16_exnode6"),
    ("spr2 single l17", "spr2", "gelu_pwl2_l00_16_pwl4_l17_pwl2_l18_23_e3_ms256_spr2_single_l17_exnode6"),
    ("spr2 single l19", "spr2", "gelu_pwl2_l00_18_pwl4_l19_pwl2_l20_23_e3_ms256_spr2_single_l19_exnode6"),
    ("spr2 single l20", "spr2", "gelu_pwl2_l00_19_pwl4_l20_pwl2_l21_23_e3_ms256_spr2_single_l20_exnode6"),
    ("spr2 single l21", "spr2", "gelu_pwl2_l00_20_pwl4_l21_pwl2_l22_23_e3_ms256_spr2_single_l21_exnode6"),
    ("spr2 single l22", "spr2", "gelu_pwl2_l00_21_pwl4_l22_pwl2_l23_23_e3_ms256_spr2_single_l22_exnode6"),
]

print("name\ttask\tloss\tacc\tf1_micro\tmajor")
for name, task, run_name in runs:
    path = ROOT / task / run_name / "val_metrics.json"
    if not path.exists():
        print(f"{name}\t{task}\tMISSING\t{path}")
        continue
    data = json.loads(path.read_text())
    task_data = data[task]
    minor = task_data["metrics"]["minor"]
    print(
        f"{name}\t{task}\t{task_data['loss']:.5f}\t"
        f"{minor['acc']:.5f}\t{minor['f1_micro']:.5f}\t{task_data['metrics']['major']:.5f}"
    )
