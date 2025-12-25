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
pip install scikit-learn scikit-image kaggle pandas huggingface-hub
```

### GPU Environment Setup:

```sh
conda activate cv-final-gpu
# Install PyTorch for GPU
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# Install Ultralytics to use the YOLO models
pip install ultralytics
# Install MediaPipe from Google
pip install mediapipe==0.10.14
# Other dependencies
pip install scikit-learn scikit-image kaggle pandas pycocotools
```

## 0. Training and test datasets and model weights available on Hugging Face

We provide all training datasets created in Step 1 and weights for the YOLO and ResNet models trained during Steps 2-4 in our Hugging Face repository: https://huggingface.co/datasets/rohanphanse/cpsc5800-hand-detection

We provide our test datasets and trained model weights in the following Hugging Face repository: https://huggingface.co/datasets/rohanphanse/cpsc5800-hand-detection-test/tree/main


```sh
# Recommended: download dataset using git-xet (https://hf.co/docs/hub/git-xet)
brew install git-xet
git xet install

# Download datasets (~25 GB in total, ~50 GB via git clone) and move the repository contents to the top level of your directory
git clone --depth=1 https://huggingface.co/datasets/rohanphanse/cpsc5800-hand-detection
git clone --depth=1 https://huggingface.co/datasets/rohanphanse/cpsc5800-hand-detection-test
# Optional: delete the unnecessary .git folder (~25 GB) before moving the repository contents out
# rm -rf cpsc5800-hand-detection/.git
# rm -rf cpsc5800-hand-detection-test/.git

# To avoid duplicate uploads, the HaGRIDv2 data (already in /yolo_hand) isn't included in /yolo_open_closed_hand
# Run this script to copy it over
python3 copy_over_hagridv2.py
```

## 1. Download datasets and prepare bounding boxes

Run the scripts in `dataset_downloads/download.sh` to download the datasets used in this project, including COCO-Hand, HaGRIDv2 (4 categories), and other datasets from Kaggle and Roboflow.

```sh
# Requires 150+ GB of downloads (we also provide our processed training datasets on Hugging Face so Step 1 can be skipped)
sh dataset_downloads/download.sh
```

Then, run all the Python scripts in `/bounding_box_preparation` to extract the bounding boxes into a common CSV format that will be used for building training datasets.

```sh
# Extract bounding boxes into a common CSV format
# For COCO-Hand and HaGRIDv2, images are randomly sampled to form the subsets used during training
# Use our chosen subsets with --selection_json or sample new subsets with --write-selection
# COCO-Hand
python3 bounding_box_preparation/convert_coco_hand_big.py --selection_json bounding_boxes/coco_hand_big_selection.json
# python3 bounding_box_preparation/convert_coco_hand_big.py --write-selection
# HaGRIDv2
python3 bounding_box_preparation/process_hagridv2_subset.py --selection_json bounding_boxes/hagridv2_subset_selection.json
# python3 bounding_box_preparation/process_hagridv2_subset.py --write-selection
python3 bounding_box_preparation/process_hagridv2_subset.py
# Open-Closed-1k
python3 bounding_box_preparation/convert_open_closed_1k.py
# Generate bounding boxes with MediaPipe Hands for ASL 1, ASL 2, and RPS datasets
python3 bounding_box_preparation/generate_bounding_boxes.py
```

Finally, we can build the training datasets using the data in the `/data` and `/bounding_boxes` directories.

```sh
# Build the Hand dataset /yolo_hand for training YOLO (also available at our Hugging Face repository)
python3 hand_detection/build_yolo_hand_dataset.py

# Build the Open/Closed Hand dataset
python3 open_closed_hand_classification/build_open_closed_classification_dataset.py
```

## 2. Hand Detection

```sh
# Train from pretrained weights
yolo detect train model=yolov8n.pt data=yolo_hand/data.yaml epochs=50 imgsz=640 batch=32

# Train from scratch
yolo detect train model=yolov8n.yaml data=yolo_hand/data.yaml epochs=50 imgsz=640 batch=32

# Evaluation on Hand dataset
# Evaluate trained YOLO models
python3 hand_detection/evaluate_hand_models.py
# Evaluate MediaPipe Hands
python3 hand_detection/evaluate_mediapipe_coco.py
```

## 3. Open/Closed Hand Detection

```sh
# Train from pretrained weights
yolo detect train model=yolov8n.pt data=yolo_open_closed_hand/data.yaml epochs=50 imgsz=640 batch=32

# Train from scratch
yolo detect train model=yolov8n.yaml data=yolo_open_closed_hand/data.yaml epochs=50 imgsz=640 batch=32

# Evaluate on Open-Closed-Hand dataset
# Evaluate trained YOLO models
python3 open_closed_hand_detection/evaluate_open_closed_hand_models.py
# Evaluate MediaPipe Hands
python3 open_closed_hand_detection/evaluate_mediapipe_open_closed_coco.py
```

## 4. Open/Closed Hand Classification

```sh
# Build classification dataset from YOLO Open/Closed Hand dataset
python3 open_closed_hand_classification/build_open_closed_classification_dataset.py

# Train ResNet18 from pretrained weights
python3 open_closed_hand_classification/train_resnet.py \
  --data open_closed_classification \
  --model resnet18 \
  --pretrained \
  --epochs 20 \
  --batch 64 \
  --img-size 224 \
  --output checkpoints/resnet18_open_closed.pt

# Train ResNet18 from scratch
python3 open_closed_hand_classification/train_resnet.py \
  --data open_closed_classification \
  --model resnet18 \
  --epochs 20 \
  --batch 64 \
  --img-size 224 \
  --output checkpoints/resnet18_open_closed_scratch.pt

# Train ResNet50 from pretrained weights
python3 open_closed_hand_classification/train_resnet.py \
  --data open_closed_classification \
  --model resnet50 \
  --pretrained \
  --epochs 20 \
  --batch 32 \
  --img-size 224 \
  --output checkpoints/resnet50_open_closed.pt

# Train ResNet50 from scratch
python3 open_closed_hand_classification/train_resnet.py \
  --data open_closed_classification \
  --model resnet50 \
  --epochs 20 \
  --batch 32 \
  --img-size 224 \
  --output checkpoints/resnet50_open_closed_scratch.pt

# Evaluation
# Evaluate ResNet model
ckpt=checkpoints/resnet18_open_closed.pt
model=resnet18
python3 open_closed_hand_classification/evaluate_resnet_open_closed.py \
  --checkpoint $ckpt \
  --csv bounding_boxes/hagridv2_subset_test_bounding_boxes.csv \
       bounding_boxes/open_closed_1k_all_bounding_boxes.csv \
  --model $model \
  --pretrained \
  --img-size 224
# Evaluate MediaPipe Hands
python3 open_closed_hand_classification/evaluate_mediapipe_open_closed_classification.py \
  --csv bounding_boxes/hagridv2_subset_test_bounding_boxes.csv \
        bounding_boxes/open_closed_1k_all_bounding_boxes.csv
```
