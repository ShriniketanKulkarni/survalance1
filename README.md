# 🎯 Smart Surveillance System — CodeFiesta 6.0

> **Domain:** Artificial Intelligence  
> **Problem 5:** Smart Surveillance System for Suspicious Activity Detection Using Object Detection

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the system
python app.py

# 3. Open dashboard
# http://localhost:5000
```

---

## 🗂 Project Structure

```
├── app.py            # Flask web server + REST API
├── surveillance.py   # Core pipeline (detect → track → analyse → annotate)
├── detector.py       # YOLOv11 object detection (Ultralytics + PyTorch)
├── tracker.py        # DeepSORT multi-object tracking
├── behavior.py       # Suspicious activity rules engine
├── alerting.py       # Alert manager (dedup + sound + history)
├── utils.py          # Drawing helpers, frame encoding
├── config.py         # All thresholds and settings
├── templates/
│   └── dashboard.html  # Dark-mode Flask dashboard
└── requirements.txt
```

---

## 🔍 Detection Capabilities

| # | Activity | How Detected |
|---|---|---|
| 1 | **Loitering** | Person stays within 80px radius for > 10 seconds |
| 2 | **Crowd Surge** | ≥ 5 persons in frame simultaneously |
| 3 | **Running / Sudden Movement** | Centroid velocity > 30 px/frame |
| 4 | **Abandoned Object** | Bag/luggage alone for > 8 seconds |
| 5 | **Restricted Zone Breach** | Person enters annotated polygon zone |

---

## ⚙️ Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolo11n.pt` | YOLO model (auto-downloaded) |
| `YOLO_CONF` | `0.45` | Detection confidence threshold |
| `LOITER_SECONDS` | `10` | Seconds before loitering alert |
| `CROWD_THRESHOLD` | `5` | People count for crowd alert |
| `ABANDON_SECONDS` | `8` | Seconds for abandoned object alert |
| `SPEED_THRESHOLD` | `30` | Pixels/frame for running alert |

All thresholds are **adjustable live** from the web dashboard.

---

## 🛠 Technology Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv11 (Ultralytics) + PyTorch |
| Object Tracking | DeepSORT (deep_sort_realtime) |
| Video Processing | OpenCV |
| Web Dashboard | Flask + MJPEG streaming |
| Alert Sound | `winsound` (Windows) |

---

## 🌐 Dashboard Features

- **Live MJPEG Video Feed** — annotated with bounding boxes and track IDs
- **Real-time Stats** — person count, FPS, total alerts
- **Alert Log** — colour-coded by severity (HIGH/MEDIUM/LOW)
- **Toast Notifications** — pops on every new alert
- **Configurable Thresholds** — live sliders for all detection rules
- **Video Source Switcher** — webcam, MP4, or RTSP URL

---

## 📽 Supported Video Sources

```python
# Webcam
source = 0

# Video file
source = "path/to/video.mp4"

# RTSP stream
source = "rtsp://192.168.1.100:554/stream"
```
