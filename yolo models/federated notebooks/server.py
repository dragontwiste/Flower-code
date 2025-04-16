# server.py

import flwr as fl
from typing import Dict, List, Tuple
import torch
from ultralytics import YOLO
import warnings
from flwr.common import parameters_to_ndarrays
from collections import OrderedDict
from flwr.server.strategy import DPFedAvgFixed, FedAvg, DPFedAvgAdaptive



# ✅ Global model
model = YOLO(r"E:\PFE\Flower-code\yolo models\yolo11m_mass.pt", task="detect")
model.model.nc = 2
model.model.names = {0: "no_mass", 1: "mass"}
model.fuse()
model.model = model.model.to("cuda")


# ✅ Path to validation YAML (use any one full dataset)
VAL_YAML = r"E:\PFE\Flower-code\data created\client_cbis_ddsm0\cbis_ddsm.yaml"


# ✅ Helper: Set weights
def set_weights(weights):
    state_dict = OrderedDict({
        k: torch.tensor(v) for k, v in zip(model.model.state_dict().keys(), weights)
    })
    with torch.inference_mode():
        model.model.load_state_dict(state_dict, strict=True)



# ✅ Helper: Get weights
def get_weights():
    return [val.cpu().numpy() for _, val in model.model.named_parameters()]

# ✅ Helper: Save model
def save_model(round_num):
    model_path = fr"E:\PFE\Flower-code\yolo models\global_model_round_{round_num}.pt"
    model.save(model_path)
    print(f"💾 Saved global model at: {model_path}")



# ✅ Custom strategy
class YOLOStrategy(FedAvg):
    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def evaluate(self, rnd, parameters):
        return None



    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException]
    ) -> Tuple[List[torch.Tensor], Dict]:
        aggregated_weights, metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_weights is not None:
            weights_list = parameters_to_ndarrays(aggregated_weights)
            set_weights(weights_list)

            # ✅ Save global model
            save_model(rnd)

            # ✅ Evaluate on central validation and train set
            metrics_val = model.val(data=VAL_YAML, split="val")
            metrics_train = model.val(data=VAL_YAML, split="train")

            print(f"\n🌍 [Global model after round {rnd}]")
            print(f"Train:  mAP50={metrics_train.box.map50:.4f}, Recall={metrics_train.box.mr:.4f}")
            print(f"Val:    mAP50={metrics_val.box.map50:.4f}, Recall={metrics_val.box.mr:.4f}\n")

        return aggregated_weights, metrics or {}
    

base_strategy = YOLOStrategy(
fraction_fit=1.0,
min_fit_clients=11,
min_available_clients=11,
)
strategy = DPFedAvgFixed(strategy=base_strategy,
                         noise_multiplier=0.5,
                         clip_norm = 1.0,
                         num_sampled_clients=11
                        )


# ✅ Start server
fl.server.start_server(
    server_address="localhost:9675",
    config=fl.server.ServerConfig(num_rounds=100, round_timeout= None),
    strategy = strategy
)
