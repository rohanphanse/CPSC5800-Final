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
pip install scikit-learn scikit-image kaggle pandas
```

### GPU Environment Setup:

```sh
conda activate cv-final-gpu
# Install PyTorch for CPU
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# Install Ultralytics to use the YOLO models
pip install ultralytics
# Install MediaPipe from Google
pip install mediapipe==0.10.14
# Other dependencies
pip install scikit-learn scikit-image kaggle pandas pycocotools
```

## 1. Download datasets and prepare bounding boxes

Run the scripts in `dataset_downloads/download.sh` to download the datasets used in this project, including COCO-Hand, HaGRIDv2 (4 categories), and other datasets from Kaggle and Roboflow

```sh
# Requires 150+ GB of downloads (we also provide our processed training dataset on Hugging Face so this step can be skipped)
sh dataset_downloads/download.sh
```

Then, run all the Python scripts in `/bounding_box_preparation` to extract the bounding boxes into a common CSV format that will be used to building training datasets later on.

```sh
python3 bounding_box_preparation/convert_coco_hand_big.py
python3 bounding_box_preparation/convert open_closed_1k.py
python3 bounding_box_preparation/process_hagridv2_subset.py
python3 bounding_box_preparation/generate_bounding_boxes.py
```

The `bounding_boxes` directory created in this step is also available on Hugging Face at https://huggingface.co/datasets/rohanphanse/cpsc5800-hand-detection

# 2. Hand Detection

```sh
# Build the Hand dataset for training YOLO from /bounding_boxes
python3 hand_detection/build_yolo_hand_dataset.py

# Train YOLO models

# Train from pretrained weights
yolo detect train model=yolov8n.pt data=yolo_hand/data.yaml epochs=50 imgsz=640 batch=32 device=0

# Train from scratch
yolo detect train model=yolov8n.yaml data=yolo_hand/data.yaml epochs=50 imgsz=640 batch=32 device=1

# Evaluation on Hand dataset
python3 hand_detection/evaluate_hand_models.py
python3 hand_detection/evaluate_mediapipe_coco.py
```

# 3. Open/Closed Hand Detection

```sh
# Build training dataset
python3 open_closed_hand_detection/build_yolo_open_closed_hand_dataset.py

# Train from pretrained weights
yolo detect train model=yolov8n.pt data=yolo_open_closed_hand/data.yaml epochs=50 imgsz=640 batch=32 device=2

# Train from scratch
yolo detect train model=yolov8n.yaml data=yolo_open_closed_hand/data.yaml epochs=50 imgsz=640 batch=32 device=0

# Evaluate on Open-Closed-Hand dataset
python3 open_closed_hand_detection/evaluate_mediapipe_open_closed_coco.py

python3 open_closed_hand_detection/evaluate_mediapipe_open_closed_coco.py
```

# 4. Open/Closed Hand Classification

```sh
# Build dataset
python3 open_closed_hand_classification/build_open_closed_classification_dataset.py

# Train
ckpt=checkpoints/resnet18_open_closed.pt
model=resnet18
python3 evaluate_resnet_open_closed.py \
  --checkpoint $ckpt \
  --csv bounding_boxes/hagridv2_subset_test_bounding_boxes.csv \
       bounding_boxes/open_closed_1k_all_bounding_boxes.csv \
  --model $model \
  --pretrained \
  --img-size 224

  python3 train_resnet.py \
  --data /home/rap93/CPSC5800-Final/open_closed_classification \
  --model resnet18 \
  --epochs 20 \
  --batch 64 \
  --img-size 224 \
  --output checkpoints/resnet18_open_closed_scratch.pt

python3 train_resnet.py \
  --data /home/rap93/CPSC5800-Final/open_closed_classification \
  --model resnet50 \
  --pretrained \
  --epochs 20 \
  --batch 32 \
  --img-size 224 \
  --output checkpoints/resnet50_open_closed.pt

python3 train_resnet.py \
  --data /home/rap93/CPSC5800-Final/open_closed_classification \
  --model resnet50 \
  --epochs 20 \
  --batch 32 \
  --img-size 224 \
  --output checkpoints/resnet50_open_closed_scratch.pt

# Evaluate with scripts in /open_closed_hand_classification directory
```