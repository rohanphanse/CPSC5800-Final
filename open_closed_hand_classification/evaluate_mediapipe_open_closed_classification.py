import argparse
import csv
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

MAX_HANDS = 1
MIN_DET_CONF = 0.15
OPEN_THRESHOLD = 1.1


def map_label(dataset, lbl):
    if dataset == "hagridv2":
        return 1 if lbl == 1 else 0
    return 1 if lbl == 1 else 0


def load_records(csv_path):
    grouped = {}
    per_image_count = {}
    rows_by_image = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_rel = row.get("relative_path", "")
            if not img_rel:
                continue
            per_image_count[img_rel] = per_image_count.get(img_rel, 0) + 1
            rows_by_image.setdefault(img_rel, row)
    for img_rel, count in per_image_count.items():
        if count != 1:
            continue
        row = rows_by_image[img_rel]
        try:
            label = int(row["label"])
            dataset = row.get("dataset", "")
        except (KeyError, ValueError):
            continue
        grouped[img_rel] = {
            "dataset": dataset,
            "label": map_label(dataset, label),
        }
    return grouped


def mediapipe_open_closed(image):
    results = MP_HANDS.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    xs = np.array([p.x for p in lm.landmark])
    ys = np.array([p.y for p in lm.landmark])
    wrist = np.array([xs[0], ys[0]])
    tip_idxs = [4, 8, 12, 16, 20]
    pip_idxs = [2, 6, 10, 14, 18]
    ratios = []
    for t_idx, p_idx in zip(tip_idxs, pip_idxs):
        tip = np.array([xs[t_idx], ys[t_idx]])
        pip = np.array([xs[p_idx], ys[p_idx]])
        pip_dist = np.linalg.norm(pip - wrist)
        if pip_dist <= 1e-6:
            continue
        tip_dist = np.linalg.norm(tip - wrist)
        ratios.append(tip_dist / pip_dist)
    if not ratios:
        return None
    open_score = float(sum(ratios) / len(ratios))
    return 0 if open_score >= OPEN_THRESHOLD else 1


def evaluate(records):
    correct = 0
    total = 0
    missing = 0
    for img_rel, rec in tqdm(records.items(), desc="eval", leave=False):
        src = img_rel if os.path.isabs(img_rel) else str(Path(os.getcwd()) / img_rel)
        if not os.path.exists(src):
            missing += 1
            continue
        image = cv2.imread(src)
        if image is None:
            missing += 1
            continue
        pred = mediapipe_open_closed(image)
        if pred is None:
            missing += 1
            continue
        if pred == rec["label"]:
            correct += 1
        total += 1
    return correct, total, missing


def main():
    ap = argparse.ArgumentParser(description="Mediapipe open/closed classification on cropped boxes.")
    ap.add_argument("--csv", type=Path, nargs="+", required=True, help="CSV file(s) with bounding boxes.")
    args = ap.parse_args()

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
        correct, total, missing = evaluate(records)
        acc = correct / total if total > 0 else 0.0
        print(f"{csv_path.name}: acc={acc:.4f} ({correct}/{total}), missing={missing}")


if __name__ == "__main__":
    MP_HANDS = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DET_CONF,
    )
    main()
    MP_HANDS.close()
