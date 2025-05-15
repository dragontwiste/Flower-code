import os
import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import flwr as fl
import torch
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

# ✅ Paths
DATA_ROOT = r"/home_nfs/benyemnam/Flower-code/data original/DATA/Mass"
CLIENT_ROOT = r"/home_nfs/benyemnam/Flower-code/data created"
TRAIN_PATH = os.path.join(DATA_ROOT, "Train")
TEST_PATH = os.path.join(DATA_ROOT, "Test")
NUM_CLIENTS = 3  # Set how many clients you want

def find_bounding_boxes(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            boxes.append([x, y, x + w, y + h])
    return boxes

def collect_data():
    data = []
    for label in ["BENIGN", "MALIGNANT"]:
        path = os.path.join(TRAIN_PATH, label)
        files = sorted(os.listdir(path))
        for i in range(0, len(files), 2):
            img_file = files[i]
            mask_file = files[i + 1] if i + 1 < len(files) else None
            if "MASK" not in mask_file:
                continue
            img_path = os.path.join(path, img_file)
            mask_path = os.path.join(path, mask_file)
            if not os.path.exists(mask_path):
                continue
            boxes = find_bounding_boxes(mask_path)
            data.append([img_path, boxes, 0])
    return pd.DataFrame(data, columns=["image", "boxes", "label"])

def process_and_save(df, img_dir, lbl_dir):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path, boxes, label = row["image"], row["boxes"], row["label"]
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(img_dir, filename), image)
        lbl_path = os.path.join(lbl_dir, filename.replace(".png", ".txt"))
        if not boxes:
            open(lbl_path, "w").close()
            continue
        yolo_boxes = [f"{label} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}" for x1, y1, x2, y2 in boxes]
        with open(lbl_path, "w") as f:
            f.write("/n".join(yolo_boxes))

def prepare_data(client_id):
    client_path = os.path.join(CLIENT_ROOT, f"client_cbis_ddsm{client_id}")
    yaml_path = os.path.join(client_path, "cbis_ddsm.yaml")
    if os.path.exists(yaml_path):
        return yaml_path, client_path

    train_img = os.path.join(client_path, "train/images")
    train_lbl = os.path.join(client_path, "train/labels")
    val_img = os.path.join(client_path, "valid/images")
    val_lbl = os.path.join(client_path, "valid/labels")
    [Path(p).mkdir(parents=True, exist_ok=True) for p in [train_img, train_lbl, val_img, val_lbl]]

    df = collect_data()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split across clients
    total = len(df)
    part = total // NUM_CLIENTS
    start = client_id * part
    end = (client_id + 1) * part if client_id < NUM_CLIENTS - 1 else total
    df = df.iloc[start:end].reset_index(drop=True)

    # Stratified train/val
    train_df, val_df = pd.DataFrame(), pd.DataFrame()
    for label in df["label"].unique():
        group = df[df["label"] == label].sample(frac=1, random_state=42)
        split = int(len(group) * 0.8)
        train_df = pd.concat([train_df, group.iloc[:split]])
        val_df = pd.concat([val_df, group.iloc[split:]])

    process_and_save(train_df, train_img, train_lbl)
    process_and_save(val_df, val_img, val_lbl)

    config = {
        "path": client_path,
        "train": "train/images",
        "val": "valid/images",
        "names": {0: "no_mass", 1: "mass"}
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    return yaml_path, client_path

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.model.named_parameters()]

def set_parameters(model, parameters):
    keys = [k for k, _ in model.model.named_parameters()]
    state_dict = OrderedDict()
    for k, v in zip(keys, parameters):
        t = torch.tensor(v)
        if t.is_floating_point():
            t.requires_grad = True
        state_dict[k] = t
    with torch.inference_mode():
        model.model.load_state_dict(state_dict, strict=False)

def evaluate_and_log(model, yaml_path, tag="Client"):
    val = model.val(data=yaml_path, split="val", plots=False)
    train = model.val(data=yaml_path, split="train", plots=False)
    print(f"📊 [{tag}] Train: mAP50={train.box.map50:.4f}, Recall={train.box.mr:.4f}")
    print(f"📊 [{tag}] Val:   mAP50={val.box.map50:.4f}, Recall={val.box.mr:.4f}")
    return {
        "map50_train": train.box.map50,
        "recall_train": train.box.mr,
        "map50_val": val.box.map50,
        "recall_val": val.box.mr,
    }

class CBISClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.yaml_path, self.client_data_path = prepare_data(client_id)
        self.model = YOLO(r"/home_nfs/benyemnam/Flower-code/yolo models/yolo11m_mass.pt", task="detect")
        self.model.fuse = False

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(f"🚀 Training CBIS-DDSM client {self.client_id}")
        self.model.train(
            data=self.yaml_path,
            epochs=1,
            imgsz=640,
            batch=1,
            device="cuda",
            workers=0,
            verbose=False,
        )
        metrics = evaluate_and_log(self.model, self.yaml_path, tag=f"CBIS-DDSM-{self.client_id}")
        train_size = len(os.listdir(os.path.join(self.client_data_path, "train/images")))
        return get_parameters(self.model), train_size, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate_and_log(self.model, self.yaml_path, tag=f"CBIS-DDSM-{self.client_id} Eval")
        val_size = len(os.listdir(os.path.join(self.client_data_path, "valid/images")))
        loss = 1.0 - metrics["map50_val"]
        return loss, val_size, metrics

def start_client(client_id):
    fl.client.start_client(server_address="localhost:9675", client=CBISClient(client_id).to_client())
