import os
import sys

from jiant.proj.main.modeling import model_setup
from jiant.proj.main.components.container_setup import TaskmodelsConfig


PROJECT_ROOT = "/cluster/home/zhangzx/my_project"
EXP_DIR = f"{PROJECT_ROOT}/base_exp/exp_edge"
MODEL_PATH = f"{EXP_DIR}/models/bert-large-uncased/model/model.p"
CONFIG_PATH = f"{EXP_DIR}/models/bert-large-uncased/model/config.json"


def main():
    ranks = [int(sys.argv[1])] if len(sys.argv) > 1 else [512, 1024, 2048]
    for rank in ranks:
        os.environ["FFN_INTERMEDIATE_RANK"] = str(rank)
        jiant_model = model_setup.setup_jiant_model(
            hf_pretrained_model_name_or_path="bert-large-uncased",
            model_config_path=CONFIG_PATH,
            task_dict={},
            taskmodels_config=TaskmodelsConfig(task_to_taskmodel_map={}),
        )
        model_setup.delegate_load_from_path(
            jiant_model=jiant_model,
            weights_path=MODEL_PATH,
            load_mode="from_transformers",
        )
        layer0 = jiant_model.encoder.encoder.layer[0]
        wi = layer0.intermediate.dense.weight.shape
        wo = layer0.output.dense.weight.shape
        total = sum(p.numel() for p in jiant_model.encoder.parameters())
        trainable = sum(p.numel() for p in jiant_model.encoder.parameters() if p.requires_grad)
        print(f"rank={rank} intermediate={tuple(wi)} output={tuple(wo)} params={total} trainable={trainable}", flush=True)


if __name__ == "__main__":
    main()
