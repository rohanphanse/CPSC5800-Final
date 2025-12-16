import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import cv2
from tqdm import tqdm

ANN_PATH = "data/coco/COCO-Hand/COCO-Hand-Big/COCO-Hand-Big_annotations.txt"
IMG_DIR = "data/coco/COCO-Hand/COCO-Hand-Big/COCO-Hand-Big_Images"
OUTPUT_DIR = "bounding_boxes"
OUTPUT_SELECTION_JSON = os.path.join(OUTPUT_DIR, "coco_hand_big_selection.json")
SEED = 42
SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 4_000}
SAMPLE_VIS_COUNT = 50


def parse_args():
    p = argparse.ArgumentParser(description="Sample COCO-Hand-Big into train/val/test with CSVs and visuals.")
    p.add_argument(
        "--selection-json",
        help="Optional JSON with preselected image ids per split: {'train': [...], 'val': [...], 'test': [...]}",
    )
    p.add_argument(
        "--write-selection",
        action="store_true",
        help="Write the sampled ids to bounding_boxes/coco_hand_big_selection.json",
    )
    return p.parse_args()


def load_annotations():
    data = defaultdict(list)
    with open(ANN_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="annotations", leave=False):
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            fname = parts[0]
            try:
                xmin = float(parts[1])
                xmax = float(parts[2])
                ymin = float(parts[3])
                ymax = float(parts[4])
            except ValueError:
                continue
            data[fname].append((xmin, ymin, xmax, ymax))
    return data


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vis_root = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_root, exist_ok=True)
    return vis_root


def pick_splits(img_ids, selection_json):
    if selection_json:
        with open(selection_json, "r", encoding="utf-8") as f:
            return json.load(f)
    rng = random.Random(SEED)
    splits = {}
    ids = list(img_ids)
    rng.shuffle(ids)
    start = 0
    for split in ["train", "val", "test"]:
        count = min(SPLIT_SIZES[split], len(ids) - start)
        splits[split] = ids[start : start + count]
        start += count
    return splits


def main():
    args = parse_args()
    random.seed(SEED)
    vis_root = ensure_dirs()

    ann_data = load_annotations()
    available_imgs = {f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")}
    img_ids = sorted(set(ann_data.keys()) & available_imgs)
    if not img_ids:
        print("No images found to process.", file=sys.stderr)
        return

    splits = pick_splits(img_ids, args.selection_json)
    generated_selection = splits if not args.selection_json else None

    total_records = 0
    for split, ids in splits.items():
        out_csv = os.path.join(OUTPUT_DIR, f"coco_hand_big_{split}_bounding_boxes.csv")
        vis_dir = os.path.join(vis_root, f"coco_hand_big_{split}")
        os.makedirs(vis_dir, exist_ok=True)

        records = []
        samples = []
        for fname in tqdm(ids, desc=f"{split}"):
            if fname not in ann_data:
                continue
            img_path = os.path.join(IMG_DIR, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            boxes = []
            for xmin, ymin, xmax, ymax in ann_data[fname]:
                boxes.append((xmin, ymin, xmax, ymax))
                records.append(
                    {
                        "dataset": "COCO-Hand-Big",
                        "relative_path": os.path.relpath(img_path),
                        "label": "hand",
                        "width": w,
                        "height": h,
                        "xmin": int(xmin),
                        "ymin": int(ymin),
                        "xmax": int(xmax),
                        "ymax": int(ymax),
                    }
                )
            if boxes:
                samples.append((img_path, boxes))

        fields = ["dataset", "relative_path", "label", "width", "height", "xmin", "ymin", "xmax", "ymax"]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

        take = min(len(samples), SAMPLE_VIS_COUNT)
        for img_path, boxes in random.sample(samples, take):
            img = cv2.imread(img_path)
            if img is None:
                continue
            for xmin, ymin, xmax, ymax in boxes:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            out_path = os.path.join(vis_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, img)

        total_records += len(records)
        print(f"{split}: {len(records)} boxes -> {out_csv}; visuals: {take}")

    if args.write_selection and generated_selection:
        with open(OUTPUT_SELECTION_JSON, "w", encoding="utf-8") as f:
            json.dump(generated_selection, f, indent=2)
        print(f"Saved selection to {OUTPUT_SELECTION_JSON}")

    print(f"Total boxes written: {total_records}")


if __name__ == "__main__":
    main()
