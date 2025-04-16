# Flower-Federated-Mammo-Detection and Classification

This repository contains a Flower-based federated learning setup for training YOLO-based mass detection and ResNet-based mass classification models on mammography datasets (INbreast, CBIS‑DDSM). The project includes:

- **Server** scripts to coordinate federated rounds using Flower’s `FedAvg` (with optional Differential Privacy wrappers).
- **Client** scripts/notebooks for:
  - INbreast mass detection (YOLO)
  - INbreast mass classification (ResNet50)
  - CBIS‑DDSM mass classification (ResNet50)

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Server](#running-the-server)
  - [Launching Clients](#launching-clients)
- [Contributing](#contributing)
- [License](#license)

---

## 🛠 Prerequisites

- Python 3.8+ (tested on 3.11)
- CUDA-enabled GPU (optional but recommended for training)
- `pip` package manager

---

## ⚙️ Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Flower-Federated-Mammo.git
   cd Flower-Federated-Mammo
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > **Note:** If you don’t have a `requirements.txt` yet, you can generate one:
>
>   ```bash
>   pip freeze > requirements.txt
>   ```

---

## 📁 Project Structure

```
├── client_inbreast.py         # INbreast YOLO detection client
├── client_inbreast_classifier.py  # INbreast mass classification client
├── client_cbis_ddsm.py        # CBIS‑DDSM classification client
├── server.py                  # Federated server script
├── notebooks/                 # Colab/Jupyter notebooks for centralized experiments
├── data_original/             # Raw datasets (INbreast, CBIS‑DDSM) — *not checked in*
└── data_created/              # Processed client data folders (generated at runtime)
```

---

## 🛠️ Configuration

Many scripts use hard‑coded paths rooted at:

```
E:\PFE\Flower-code
```

Before running, **replace** all occurrences of this base path with your local folder path. For example, if you cloned your project to:

```
/home/alice/projects/Flower-Federated-Mammo
```

then change:

```diff
- XLS_PATH = r"E:\PFE\Flower-code\data_original\INbreast ..."
+ XLS_PATH = r"/home/alice/projects/Flower-Federated-Mammo/data_original/INbreast ..."
```

Use your editor’s find‑and‑replace to update all path constants accordingly.

---

## ▶️ Usage

### 1. Running the Server

Open a terminal, activate your environment, and:

```bash
python server.py
```

This will start the Flower server on `localhost:9675` and wait for clients to connect.

### 2. Launching Clients

In **separate** terminals (one per client), run:

```bash
# INbreast detection client (example client 0)
python client_inbreast.py --client_id 0

# INbreast classification client (example client 1)
python client_inbreast_classifier.py --client_id 1

# CBIS‑DDSM classification client (example client 2)
python client_cbis_ddsm.py --client_id 2
```

Adjust the `--client_id` flag from `0` to `NUM_CLIENTS-1` for each client script.

> **Tip:** You can also run clients inside Jupyter by importing and calling the `start_client(...)` function.

---

## 🤝 Contributing

Contributions welcome! Please open issues or PRs for:

- Bug reports or feature requests
- Adding new datasets or models
- Improving documentation

---

## 📜 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

