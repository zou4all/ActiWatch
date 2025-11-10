
# 🕵️‍♂️ ActiWatch-Full

### AI-powered package theft and suspicious behavior detection system

*(First version — still under improvement)*

**ActiWatch-Full** is an open-source project that analyzes surveillance videos to detect **people**, **packages**, and **suspicious actions**.
It combines *object detection*, *proximity tracking*, and *behavioral cues* (like looking around or hiding the face) to estimate a **risk score** for theft-like behavior.

---

## 📁 Project Structure

```
ActiWatch-Full/
│
├── configs/                # YAML configuration files
│   ├── behavioral.yaml     # Behavior-based detection (look around, face hidden, etc.)
│   ├── mvp.yaml            # Minimal prototype config (ROI detection)
│   └── suspicion.yaml      # Rule-based proximity detection
│
├── data/                   # Input videos and test clips
│   ├── video_sucp.mp4
│   └── ...
│
├── outputs/                # Generated results (videos, logs, JSON events)
│
├── runs/                   # YOLO runs (auto-created by Ultralytics)
│
├── src/                    # Optional source utilities (custom scripts)
│
├── run_behavioral.py       # Behavioral detection (looking around, hidden face)
├── run_suspicion.py        # Suspicious proximity logic (person + package)
├── run_mvp.py              # Minimal proof of concept
├── run_mvp_roi.py          # ROI-based version of MVP
│
├── make_synthetic.py       # Utility to synthesize example frames
├── pick_roi.py             # Helper to manually select ROI zones
│
├── yolov8s.pt              # YOLOv8 base model (COCO)
│
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

---

## 🧠 Core Modules

| Script                  | Description                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| **`run_mvp_roi.py`**    | Minimal pipeline: detects package presence/disappearance in a Region of Interest (ROI).    |
| **`run_suspicion.py`**  | Rule-based logic using YOLOv8: detects people near objects and flags suspicious proximity. |
| **`run_behavioral.py`** | Advanced module: analyzes *behavioral cues* (head movement, face visibility, mask/hat).    |
| **`make_synthetic.py`** | Generates synthetic frames for testing and visualization.                                  |
| **`pick_roi.py`**       | Helps you draw and save a polygon ROI interactively.                                       |

---

## 🧩 Features

* ✅ **Object Detection** (people, boxes, bags, etc.)
* 👀 **Behavior Analysis** (looking around, face hidden, wearing hat/mask)
* 🚨 **Risk Scoring** (`0–1`) with thresholds for `WATCH` and `ACTION`
* 🧾 **Logs & Alerts** in CSV and JSON formats
* 🖥️ **CPU and GPU** compatible (GPU recommended for live use)

---

## ⚙️ Installation

```bash
conda create -n actiwatch python=3.10 -y
conda activate actiwatch
pip install -r requirements.txt
```

For GPU acceleration (RTX 4070 example):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## ▶️ How to Use

### 1️⃣ Minimal Proof of Concept (ROI)

Detect package presence/disappearance inside a fixed region:

```bash
python run_mvp_roi.py --video data/video_sucp.mp4 --out outputs/mvp_out.mp4 --config configs/mvp.yaml
```

### 2️⃣ Suspicious Proximity Detection

Track when a person gets close to a package and stays too long:

```bash
python run_suspicion.py --video data/video_sucp.mp4 --config configs/suspicion.yaml --out outputs/suspicion_out.mp4
```

### 3️⃣ Behavioral Detection

Detect when a person looks around, hides face, or wears a mask/hat:

```bash
python run_behavioral.py --video data/video_sucp.mp4 --config configs/behavioral.yaml --out outputs/behavioral_out.mp4
```

---

## ⚡ Risk Model Logic

| Behavior                     | Risk Contribution |
| ---------------------------- | ----------------- |
| Person near package          | +0.55 – 0.70      |
| Looking around (head motion) | +0.15             |
| Face hidden or mask detected | +0.20             |
| Hat or hood                  | +0.05             |

---

## 🔁 Processing Flow

```
[Input Video]
   ↓
YOLOv8 → detect person/package
   ↓
Compute proximity + behavior (face/pose)
   ↓
Estimate risk → WATCH/ACTION alert
   ↓
Output annotated video + CSV + JSON logs
```

---

## 📅 Version Note

This is the **first version** of ActiWatch-Full.
