# client.py

import os
import shutil
import cv2
import yaml
import pydicom
import numpy as np
import pandas as pd
import plistlib
from tqdm import tqdm
from ultralytics import YOLO
import flwr as fl
from pathlib import Path
import torch
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
# ✅ Paths
XLS_PATH = r"E:\PFE\Flower code\data original\INbreast Release 1.0\INbreast.xls"
ALLDICOMs = r"E:\PFE\Flower code\data original\INbreast Release 1.0\AllDICOMs"
ALLXML = r"E:\PFE\Flower code\data original\INbreast Release 1.0\AllXML"
CLIENT_ROOT = r"E:\PFE\Flower code\data created"
NUM_CLIENTS = 5

def map_label(bi_rads):
    return 0 if str(bi_rads).strip() == "1" else 1

def load_calcifications(xml_path, imshape):
    def load_point(point_string):
        x, y = tuple(map(float, point_string.strip('()').split(',')))
        return y, x
    boxes = []
    with open(xml_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        for roi in plist_dict['ROIs']:
            if "Mass" not in roi['Name']:
                continue
            points = [load_point(pt) for pt in roi['Point_px']]
            x, y = zip(*points)
            x1, y1, x2, y2 = min(y), min(x), max(y), max(x)
            width, height = x2 - x1, y2 - y1
            if width > 0 and height > 0:
                boxes.append((x1, y1, x2, y2))
    return boxes

def process_and_save(df, img_dir, lbl_dir):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_name = str(int(row['File Name']))
        birads = str(row['Bi-Rads']).strip()
        label = 0 if birads == "1" else 1

        dicom_file = next((f for f in os.listdir(ALLDICOMs) if f.startswith(file_name)), None)
        if not dicom_file:
            continue

        dcm_path = os.path.join(ALLDICOMs, dicom_file)
        image = pydicom.dcmread(dcm_path).pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = np.stack([image, image, image], axis=-1)

        img_filename = f"{file_name}.png"
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, image)

        xml_file = next((f for f in os.listdir(ALLXML) if f.startswith(file_name)), None)
        lbl_filename = f"{file_name}.txt"
        lbl_path = os.path.join(lbl_dir, lbl_filename)

        if label == 0:
            open(lbl_path, "w").close()
            continue

        if not xml_file:
            continue

        xml_path = os.path.join(ALLXML, xml_file)
        boxes = load_calcifications(xml_path, image.shape[:2])
        h, w = image.shape[:2]

        yolo_boxes = []
        for (x1, y1, x2, y2) in boxes:
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            yolo_boxes.append(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(lbl_path, "w") as f:
            f.write("\n".join(yolo_boxes))

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
        model.model.load_state_dict(state_dict, strict=False)  # not strict




def evaluate_and_log(model, yaml_path, tag="Client"):
    metrics_val = model.val(data=yaml_path, split="val", plots = False)
    metrics_train = model.val(data=yaml_path, split="train", plots = False)
    print(f"📊 [{tag}] Train: mAP50={metrics_train.box.map50:.4f}, Recall={metrics_train.box.mr:.4f}")
    print(f"📊 [{tag}] Val:   mAP50={metrics_val.box.map50:.4f}, Recall={metrics_val.box.mr:.4f}")
    return {
        "map50_train": metrics_train.box.map50,
        "recall_train": metrics_train.box.mr,
        "map50_val": metrics_val.box.map50,
        "recall_val": metrics_val.box.mr,
    }

class YOLOClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.client_data_path = os.path.join(CLIENT_ROOT, f"client_inbreast{client_id}")
        self.yaml_path = os.path.join(self.client_data_path, "inbreast.yaml")
        self.model = YOLO(r"E:\PFE\Flower code\yolo models\yolo11m_mass.pt", task="detect")
        self.model.fuse = False 

        self.prepare_data()

    def prepare_data(self):
        if os.path.exists(self.yaml_path):
            return

        Path(self.client_data_path).mkdir(parents=True, exist_ok=True)
        train_img = os.path.join(self.client_data_path, "train/images")
        train_lbl = os.path.join(self.client_data_path, "train/labels")
        val_img = os.path.join(self.client_data_path, "valid/images")
        val_lbl = os.path.join(self.client_data_path, "valid/labels")
        [Path(p).mkdir(parents=True, exist_ok=True) for p in [train_img, train_lbl, val_img, val_lbl]]

        df = pd.read_excel(XLS_PATH)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['File Name'])
        birads_followup = ["4", "4a", "4b", "4c", "4d", "5", "6", "49", "2", "3"]
        df = df[(df['Bi-Rads'].astype(str) == "1") | (df['Bi-Rads'].astype(str).isin(birads_followup)) & (df["Mass"] == "X")].reset_index(drop=True)
        df['label'] = df['Bi-Rads'].apply(map_label)

        total = len(df)
        part = total // NUM_CLIENTS
        start = self.client_id * part
        end = (self.client_id + 1) * part if self.client_id < NUM_CLIENTS - 1 else total
        # df = df.iloc[1:10].reset_index(drop=True)
        df = df.iloc[start:end].reset_index(drop=True)

        # Stratify
        train_df, valid_df = pd.DataFrame(), pd.DataFrame()
        for label in df["label"].unique():
            group = df[df["label"] == label].sample(frac=1, random_state=42).reset_index(drop=True)
            split_idx = int(len(group) * 0.8)
            train_df = pd.concat([train_df, group.iloc[:split_idx]])
            valid_df = pd.concat([valid_df, group.iloc[split_idx:]])
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)

        # Multiply
        train_df = pd.concat([pd.concat([g]*2 if k == 0 else [g]*7) for k, g in train_df.groupby('label')], ignore_index=True)
        valid_df = pd.concat([pd.concat([g]*2 if k == 0 else [g]*7) for k, g in valid_df.groupby('label')], ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)

        process_and_save(train_df, train_img, train_lbl)
        process_and_save(valid_df, val_img, val_lbl)

        # YAML
        yolo_config = {
            "path": self.client_data_path,
            "train": "train/images",
            "val": "valid/images",
            "names": {0: "no_mass", 1: "mass"}
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(yolo_config, f)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("start training")
        self.model.train(
            data=self.yaml_path,
            epochs=1,
            imgsz=640,
            batch=1,
            device="cuda",
            workers=0,
            verbose=False,
        )

        print("start evaluating")
        raw_metrics = evaluate_and_log(self.model, self.yaml_path, tag=f"Client {self.client_id}")
        metrics = {k: float(v) for k, v in raw_metrics.items()}  # ✅ convert to Python float
        train_size = len(os.listdir(os.path.join(self.client_data_path, "train/images")))
        print("returning weights")
        return get_parameters(self.model), train_size, metrics


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        raw_metrics = evaluate_and_log(self.model, self.yaml_path, tag=f"Client {self.client_id} Eval")
        metrics = {k: float(v) for k, v in raw_metrics.items()}
        val_size = len(os.listdir(os.path.join(self.client_data_path, "valid/images")))
        
        loss = 1.0 - metrics["map50_val"]  # Or any other placeholder or estimation
        return loss, val_size, metrics



def start_client(client_id):
    fl.client.start_client(server_address="localhost:9675", client=YOLOClient(client_id).to_client())
