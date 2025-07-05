# ActiWatch 🔍🏠

**ActiWatch** is a personal and progressive open-source project aiming to detect **package theft** using AI-powered video analysis. It combines **object detection** and **action recognition** to build a smart surveillance system for homes and retail spaces.

This repo brings together state-of-the-art models like **YOLOv8**, **MoViNet**, and **SlowFast**, along with example pipelines and tools to analyze behaviors such as "picking up a package", "walking away", or "suspicious lingering".

---

## 🎯 Objective

To build a real-world system that can:

* 🎯 Detect the presence of a package (e.g., at the door)
* 🧍 Recognize when someone interacts with it (e.g., picks it up)
* ❌ Detect when the package disappears
* 📤 Log or alert if a possible theft is detected

This project is being developed progressively and is fully modular.

---

## 🧠 Project Pipeline (High-Level)

```text
[Input video / webcam feed] 
      ↓
[YOLOv8] → Detect package presence in frame
      ↓
[MoViNet / SlowFast] → Recognize actions like "picking up"
      ↓
[YOLOv8 again] → Check if the package is gone
      ↓
🚨 Trigger alert if: package was present → action detected → package gone
```

---

## 🧩 Key Components and Sources

### 1. `movinet-pytorch`

* 🔗 [Atze00/MoViNet-pytorch](https://github.com/Atze00/MoViNet-pytorch)
* MoViNet stream model to detect actions like "walking", "picking up"

### 2. `movinet-violence-detection`

* 🔗 [engares/MoViNets-for-Violence-Detection-in-Live-Video-Streaming](https://github.com/engares/MoViNets-for-Violence-Detection-in-Live-Video-Streaming)
* Uses MoViNet-A3 for real-time action detection on edge devices (TFLite)

### 3. `yolov8-retail`

* 🔗 [vmc-7645/YOLOv8-retail](https://github.com/vmc-7645/YOLOv8-retail)
* Detects packages and other objects in video frames

### 4. `custom-movinet`

* 🔗 [naseemap47/Custom-MoViNet](https://github.com/naseemap47/Custom-MoViNet)
* Fine-tune MoViNet on custom action classes (e.g., "stealing")

### 5. `slowfast-pytorchvideo`

* 🔗 [facebookresearch/pytorchvideo](https://pytorchvideo.org/docs/tutorial_torchhub_classification)
* TorchHub version of SlowFast model for action classification

### 6. `slowfast-detectron2`

* 🔗 [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)
* Precise spatio-temporal action detection with bounding boxes

### 7. `mmaction2`

* 🔗 [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2)
* Full-featured framework supporting multiple action detection models

---

## 🚧 Project Progress Plan

* ✅ Research models and design pipeline
* ✅ Run baseline MoViNet action detection
* 🔄 Integrate YOLOv8 for package tracking
* 🔄 Combine MoViNet + YOLOv8 + alert logic
* 🔄 Export results as timestamped logs (CSV/JSON)

---


💡 *This project is in active development. The goal is to create a working proof-of-concept for intelligent home surveillance — starting with video inputs and ending with smart, timestamped alerts.*

Stay tuned — and feel free to contribute or suggest improvements! 🚀
