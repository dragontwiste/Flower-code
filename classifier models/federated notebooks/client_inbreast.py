import os
import torch
import flwr as fl
import warnings
import pydicom
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from collections import OrderedDict
from flwr.client import start_client as flwr_start_client

warnings.filterwarnings("ignore")

# ✅ Config
XLS_PATH = r"E:\PFE\Flower code\data original\INbreast Release 1.0\INbreast.xls"
ALLDICOMs = r"E:\PFE\Flower code\data original\INbreast Release 1.0\AllDICOMs"
NUM_CLIENTS = 2

# ✅ Label map
def map_label(bi_rads):
    return 0 if str(bi_rads).strip() == "2" else 1

# ✅ Dataset class
class InbreastDataset(Dataset):
    def __init__(self, df, dicom_dir, transform=None):
        self.df = df
        self.dicom_dir = dicom_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = str(int(row['File Name']))
        label = row['label']

        dicom_file = next((f for f in os.listdir(self.dicom_dir) if f.startswith(file_name)), None)
        dcm_path = os.path.join(self.dicom_dir, dicom_file)
        image = pydicom.dcmread(dcm_path).pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        image = Image.fromarray(image).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# ✅ Model
def create_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ✅ Transformations
train_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ Flower client
class InbreastClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = create_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load data for this client
        df = pd.read_excel(XLS_PATH)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["File Name"])
        df = df[df["Mass"] == "X"]
        df = df[df["Bi-Rads"].astype(str).isin(["2", "3", "4", "5", "6", "1"])].copy()
        df["label"] = df["Bi-Rads"].apply(map_label)

        # Split among clients
        total = len(df)
        part = total // NUM_CLIENTS
        start = client_id * part
        end = (client_id + 1) * part if client_id < NUM_CLIENTS - 1 else total
        df = df.iloc[1:10].reset_index(drop=True)

        # Stratified split
        self.train_df, self.val_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=42)

        self.train_loader = DataLoader(InbreastDataset(self.train_df, ALLDICOMs, train_transform), batch_size=8, shuffle=True)
        self.val_loader = DataLoader(InbreastDataset(self.val_df, ALLDICOMs, val_transform), batch_size=8)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict({
            k: torch.tensor(v).to(self.device) for k, v in zip(self.model.state_dict().keys(), parameters)
        })
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(1):  # single epoch per round
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                loss = self.criterion(preds, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # ✅ Evaluate before sending weights
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                y_pred += preds.argmax(1).cpu().tolist()
                y_true += labels.cpu().tolist()

        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        metrics = {"accuracy": float(acc), "recall": float(recall)}

        print(f"📊 [Client {self.client_id}] Accuracy: {acc:.4f} | Recall: {recall:.4f}")

        return self.get_parameters(), len(self.train_loader.dataset), metrics


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                y_pred += preds.argmax(1).cpu().tolist()
                y_true += labels.cpu().tolist()

        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        return float(1.0 - acc), len(self.val_loader.dataset), {"accuracy": float(acc), "recall": float(recall)}

# ✅ Start client
def start_client(client_id):
    fl.client.start_client(server_address="localhost:9675", client=InbreastClient(client_id).to_client())
