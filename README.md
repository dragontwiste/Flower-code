# Flower-code-Detection and Classification

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

- Python 3.8+ (tested on 3.11.9)
- CUDA-enabled GPU (optional but recommended for training)
- `pip` package manager

---

## ⚙️ Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/dragontwiste/Flower-code.git
   cd Flower-code
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv/Scripts/activate    # Windows
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



---

## 🛠️ Configuration

Many scripts use hard‑coded paths rooted at:

```
/home_nfs/benyemnam/Flower-code
```

Before running, **replace** all occurrences of this base path with your local folder path. For example, if you cloned your project to:

```
/home/alice/projects/Flower-code
```

then change:

```diff
- XLS_PATH = r"/home_nfs/benyemnam/Flower-code"
+ XLS_PATH = r"/home/alice/projects/Flower-code"
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



## 📜 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

