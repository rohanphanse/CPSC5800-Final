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