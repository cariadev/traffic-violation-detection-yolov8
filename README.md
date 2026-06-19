# Traffic Violation Detection using YOLOv8

## Overview

Traffic Violation Detection using YOLOv8 is a Computer Vision and Deep Learning project designed to automatically detect vehicles, recognize traffic signs, and identify traffic violations in real time.

The system processes images, videos, and live camera streams using YOLOv8 object detection models trained on custom traffic datasets. By combining object detection with rule-based analysis, the system can monitor traffic situations and detect violations such as prohibited entry, wrong-way driving, and speeding.

---

## Features

* Real-time vehicle detection
* Traffic sign recognition
* Traffic light detection
* Traffic violation analysis
* Image, video, webcam, and IP camera support
* Custom YOLOv8 model training
* Performance evaluation using mAP, Precision, Recall, and F1-Score
* Intelligent traffic monitoring dashboard

---

## Technologies Used

### Programming Languages

* Python

### AI & Machine Learning

* YOLOv8
* PyTorch
* OpenCV
* NumPy
* Pandas

### Development Tools

* VS Code
* Git & GitHub
* Jupyter Notebook
* Kaggle

---

## Dataset

The project uses multiple datasets:

* COCO Dataset (Vehicle Classes)
* Vietnam Traffic Sign Dataset
* Custom traffic images and videos

### Detected Classes

#### Vehicles

* Car
* Motorcycle
* Truck
* Bus

#### Traffic Signs

* Speed Limit
* No Entry
* No Parking
* Stop Sign
* Other Vietnamese traffic signs

#### Traffic Lights

* Red Light
* Yellow Light
* Green Light

---

## System Architecture

Input Source
(Image / Video / Camera)

↓

YOLOv8 Vehicle Detection

↓

YOLOv8 Traffic Sign Detection

↓

YOLOv8 Traffic Light Detection

↓

Violation Analysis Engine

↓

Visualization Dashboard

---

## Training Configuration

| Parameter  | Value      |
| ---------- | ---------- |
| Model      | YOLOv8n    |
| Image Size | 640 × 640  |
| Optimizer  | AdamW      |
| Epochs     | 100        |
| Framework  | PyTorch    |
| Device     | GPU (CUDA) |

---

## Evaluation Metrics

The model performance is evaluated using:

* Precision
* Recall
* F1-Score
* mAP@50
* mAP@50-95

---

## Installation

### Clone Repository

```bash
git clone https://github.com/cariadev/traffic-violation-detection-yolov8.git
cd traffic-violation-detection-yolov8
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Detection

### Image

```bash
python detect_image.py
```

### Video

```bash
python detect_video.py
```

### Webcam

```bash
python detect_camera.py
```

---

## Project Structure

```text
traffic-violation-detection-yolov8/
│
├── dataset/
├── models/
├── weights/
├── training/
├── detection/
├── violation/
├── dashboard/
├── results/
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Vehicle tracking using DeepSORT
* License plate recognition
* Multi-camera monitoring
* Web-based dashboard
* Cloud deployment
* Advanced traffic analytics

---

## Author

Le Thi Kieu Loan

* GitHub: https://github.com/cariadev
* LinkedIn: https://www.linkedin.com/in/kiều-loan-lê-6aa4483b0

## License

This project is developed for educational and research purposes.
