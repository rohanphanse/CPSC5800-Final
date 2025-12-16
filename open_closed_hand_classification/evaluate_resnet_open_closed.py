#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from tqdm import tqdm

# Label mapping: 0=open, 1=closed
def map_label(dataset: str, lbl: int) -> int:
    if dataset == "hagridv2":
        return 1 if lbl == 1 else 0
    return 1 if lbl == 1 else 0


def clamp_bbox(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0.0, min(xmin, w - 1))
    ymin = max(0.0, min(ymin, h - 1))
    xmax = max(0.0, min(xmax, w - 1))
    ymax = max(0.0, min(ymax, h - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return int(xmin), int(ymin), int(xmax), int(ymax)


def load_records(csv_path: Path) -> List[Dict]:
    records = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
                w = float(row["width"])
                h = float(row["height"])
                label = int(row["label"])
                img_rel = row["relative_path"]
                dataset = row.get("dataset", "")
            except (KeyError, ValueError):
                continue
            records.append(
                {
                    "img_rel": img_rel,
                    "dataset": dataset,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "width": w,
                    "height": h,
                    "label": map_label(dataset, label),
                }
            )
    return records


def build_model(model_name: str, num_classes: int, pretrained: bool):
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_transform(img_size: int, pretrained: bool):
    default_mean, default_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if pretrained:
        try:
            tfms = ResNet18_Weights.IMAGENET1K_V1.transforms()
            norm_mean, norm_std = tfms.mean, tfms.std
        except Exception:
            norm_mean, norm_std = default_mean, default_std
    else:
        norm_mean, norm_std = default_mean, default_std

    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )


def evaluate(model, records: List[Dict], tfm, device, label_to_idx: Dict[str, int]) -> Tuple[int, int, int]:
    model.eval()
    correct = 0
    total = 0
    missing = 0
    with torch.no_grad():
        for rec in tqdm(records, desc="eval", leave=False):
            img_rel = rec["img_rel"]
            src = img_rel if os.path.isabs(img_rel) else str(Path(os.getcwd()) / img_rel)
            if not os.path.exists(src):
                missing += 1
                continue
            image = cv2.imread(src)
            if image is None:
                missing += 1
                continue
            h, w = image.shape[:2]
            bbox = clamp_bbox(*rec["bbox"], w, h)
            if bbox is None:
                missing += 1
                continue
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                missing += 1
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp = tfm(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
            out = model(inp)
            pred = out.argmax(dim=1).item()
            gt_idx = label_to_idx["open"] if rec["label"] == 0 else label_to_idx["closed"]
            if pred == gt_idx:
                correct += 1
            total += 1
    return correct, total, missing


def main():
    ap = argparse.ArgumentParser(description="Evaluate a ResNet open/closed checkpoint on CSV bounding boxes.")
    ap.add_argument("--checkpoint", required=True, type=Path, help="Path to checkpoint .pt")
    ap.add_argument("--csv", required=True, type=Path, nargs="+", help="CSV file(s) with bounding boxes")
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet normalization (if model was pretrained)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt.get("classes", ["open", "closed"])
    model = build_model(args.model, num_classes=len(classes), pretrained=args.pretrained).to(device)
    model.load_state_dict(ckpt["model_state"])

    tfm = get_transform(args.img_size, args.pretrained)

    label_to_idx = {name: idx for idx, name in enumerate(classes)}

    for csv_path in args.csv:
        if not csv_path.is_absolute():
            csv_path = Path(os.getcwd()) / csv_path
        if not csv_path.exists():
            print(f"Missing CSV: {csv_path}, skipping")
            continue
        records = load_records(csv_path)
        if not records:
            print(f"No records loaded for {csv_path}, skipping")
            continue
        correct, total, missing = evaluate(model, records, tfm, device, label_to_idx)
        acc = correct / total if total > 0 else 0.0
        print(f"{csv_path.name}: acc={acc:.4f} ({correct}/{total}), missing={missing}")


if __name__ == "__main__":
    main()
