import os
import sys
import cv2
import torch
import numpy as np

# Ensure we can import YOLOv7 modules (relative to this file)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


def load_model(weights_path, img_size=640, device_str="cpu"):
    device = select_device(device_str)  # "cpu" or "" to auto-select
    model = attempt_load(weights_path, map_location=device)
    stride = int(model.stride.max())
    img_size = (img_size // stride) * stride
    model.eval()

    # Class names
    names = model.module.names if hasattr(model, "module") else model.names
    return model, device, img_size, stride, names


def preprocess(frame, img_size, stride, device):
    # Letterbox resize
    img = letterbox(frame, img_size, stride=stride)[0]

    # BGR -> RGB, HWC -> CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def draw_detections(frame, det, names, color=(0, 255, 0)):
    """
    det: tensor [N, 6] with [x1, y1, x2, y2, conf, cls]
    """
    if det is None or len(det) == 0:
        return frame

    for *xyxy, conf, cls in det:
        x1, y1, x2, y2 = map(int, xyxy)
        cls = int(cls)
        label = f"{names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def main():
    # === CONFIG ===
    weights = "yolov7.pt"  # or "yolov7-tiny.pt" if you prefer
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    device_str = "cpu"      # you’re on a CPU env

    if not os.path.exists(weights):
        print(f"Could not find weights at {weights}")
        return

    print("Loading YOLOv7 model...")
    model, device, img_size, stride, names = load_model(
        weights, img_size=img_size, device_str=device_str
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access webcam")
        return

    print("YOLOv7 detection running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Preprocess
        img = preprocess(frame, img_size, stride, device)

        # Inference
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(
                pred,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                classes=None,
                agnostic=False,
            )

        det = pred[0]

        # Rescale boxes to original image size
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

        # Draw
        annotated = draw_detections(frame, det, names)

        cv2.imshow("YOLOv7 Webcam (Press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

