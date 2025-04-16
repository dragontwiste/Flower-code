
# client_cbis_ddsm.py

import os
import torch
import flwr as fl
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import torch.nn as nn
from collections import OrderedDict

warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH = r"E:\PFE\Flower code\data original\DATA\Mass\Train"
TEST_PATH  = r"E:\PFE\Flower code\data original\DATA\Mass\Test"
NUM_CLIENTS = 2  #  how many CBIS‑DDSM clients you want

# ─── Data collection ───────────────────────────────────────────────────────────
def collect_data(root_path: str) -> pd.DataFrame:
    data = []
    for label_name, label_id in [("BENIGN", 0), ("MALIGNANT", 1)]:
        class_dir = os.path.join(root_path, label_name)
        files = sorted(os.listdir(class_dir))
        # assume masks alternate too; skip any with 'MASK'
        for fname in files:
            if "MASK" in fname:
                continue
            data.append([os.path.join(class_dir, fname), label_id])
    return pd.DataFrame(data, columns=["image", "label"])

# ─── Dataset ───────────────────────────────────────────────────────────────────
class MassDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path, label = self.df.loc[idx, ["image", "label"]]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(label)

# ─── Model factory ─────────────────────────────────────────────────────────────
def create_model():
    m = models.resnet50(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m

# ─── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
val_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ─── Flower client ────────────────────────────────────────────────────────────
class CBISDDDSMClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        # 1️⃣ Build model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # 2️⃣ Load and split data
        df_train = collect_data(TRAIN_PATH)
        df_test  = collect_data(TEST_PATH)
        df_all   = pd.concat([df_train, df_test], ignore_index=True).sample(frac=1, random_state=42)

        # split among clients
        total = len(df_all)
        part  = total // NUM_CLIENTS
        start, end = client_id * part, (client_id+1)*part if client_id < NUM_CLIENTS-1 else total
        df_part = df_all.iloc[1:10].reset_index(drop=True)

        # stratified 80/20
        train_df, val_df = train_test_split(
            df_part, 
            stratify=df_part["label"], 
            test_size=0.2, 
            random_state=42
        )

        self.train_loader = DataLoader(MassDataset(train_df, train_transform), batch_size=8, shuffle=True)
        self.val_loader   = DataLoader(MassDataset(val_df,   val_transform),   batch_size=8)

    def get_parameters(self, config=None):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v).to(self.device) for k, v in params
        })
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # load global weights
        self.set_parameters(parameters)
        self.model.train()

        # one epoch per round
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            preds = self.model(images)
            loss = self.criterion(preds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # local eval before sending
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                y_pred += preds.argmax(1).cpu().tolist()
                y_true += labels.cpu().tolist()

        acc = accuracy_score(y_true, y_pred)
        rec = recall_score( y_true, y_pred)
        print(f"📊 [Client {self.client_id}]  Acc: {acc:.4f}  |  Recall: {rec:.4f}")

        return self.get_parameters(), len(self.train_loader.dataset), {"accuracy": acc, "recall": rec}

    def evaluate(self, parameters, config):
        # not used — we do local eval in fit()
        return 0.0, len(self.val_loader.dataset), {}

def start_client(client_id):
    fl.client.start_client(server_address="localhost:9675", client=CBISDDDSMClient(client_id).to_client())
