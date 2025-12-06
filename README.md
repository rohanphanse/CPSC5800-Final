# CPSC 5800 Final Project: Real-Time Hand Tracking for Gesture-Based Tennis Simulation

## Setup

Create the following virtual enviroments to install the required packages using Python version >=3.8 (recommended by [Ultralytics](https://docs.ultralytics.com/quickstart/)).

```sh
conda create -n cv-final-cpu python=3.10 -y
conda create -n cv-final-gpu python=3.10 -y
```

### CPU Enviroment Setup:

```sh
conda activate cv-final-cpu
# Install PyTorch for CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install Ultralytics to use the YOLO models
pip install ultralytics
# Install MediaPipe from Google
pip install mediapipe==0.10.14
# Other dependencies
pip install sckit-learn scikit-image
```

### To run yolo7_test.py
[YOLOv7 documentation](https://docs.ultralytics.com/models/yolov7/#onnx-export)
```sh
pip install opencv-python torch torchvision
# Download yolo7 pretrained weights
curl -L -o yolov7.pt \
  https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

### To train yolo_with_hand_classes
Add dataset to hand_dataset or replace paths in .yaml files for datasets elsewhere.

- If training **2-class YOLO** (open_hand, closed_hand):
  - Remember to label open_hand data = 0 and closed_hand data = 1
  - In yolo command below, use `data=two_class_yolo.yaml`
- If training **82-class YOLO** (COCO classes, open_hand, closed_hand):
  - Remember to label open_hand data = 80 and closed_hand data = 81
  - In yolo command below, use `data=eighty_two_class_yolo.yaml`

```sh
yolo detect train \
  model=yolov8n.pt \
  data=*.yaml \ 
  epochs= \
  imgsz= \
  batch= \
  name=
```

