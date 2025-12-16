import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Any

import cv2
from tqdm import tqdm

ROOT = "data/hagridv2"
ANNOTATIONS_DIR = os.path.join(ROOT, "annotations")
OUTPUT_DIR = "bounding_boxes"
OUTPUT_SELECTION_JSON = os.path.join(OUTPUT_DIR, "hagridv2_subset_selection.json")

SEED = 42
TRAIN_SPECS = [
    ("fist", 1, 7500),
    ("palm", 0, 2500),
    ("stop", 0, 2500),
    ("stop_inverted", 0, 2500),
]
VAL_SPECS = [
    ("fist", 1, 750),
    ("palm", 0, 250),
    ("stop", 0, 250),
    ("stop_inverted", 0, 250),
]
TEST_SPECS = [
    ("fist", 1, 3_000),
    ("palm", 0, 1_000),
    ("stop", 0, 1_000),
    ("stop_inverted", 0, 1_000),
]
SAMPLE_VIS_COUNT = 50


def load_annotations(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_abs_boxes(bboxes: List[List[float]], w: int, h: int):
    abs_boxes = []
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        x, y, bw, bh = bbox  # top-left + width/height normalized
        xmin = int(x * w)
        ymin = int(y * h)
        xmax = int((x + bw) * w)
        ymax = int((y + bh) * h)
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))
        if xmax <= xmin or ymax <= ymin:
            continue
        abs_boxes.append((xmin, ymin, xmax, ymax))
    return abs_boxes


def parse_args():
    p = argparse.ArgumentParser(description="Build HaGRIDv2 subset CSVs with visualizations.")
    p.add_argument(
        "--selection-json",
        help="Optional JSON mapping split->class->list of image ids (without .jpg).",
    )
    p.add_argument(
        "--write-selection",
        action="store_true",
        help="Write the sampled IDs to a JSON so runs are portable/reusable.",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split(s) to process (train/val/test/all). Default: all.",
    )
    return p.parse_args()


def process_split(split: str, specs, selections, generated_selection):
    records = []
    samples = []
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations", f"hagridv2_subset_{split}")
    os.makedirs(vis_dir, exist_ok=True)

    ann_split_dir = os.path.join(ANNOTATIONS_DIR, split)
    out_csv = os.path.join(OUTPUT_DIR, f"hagridv2_subset_{split}_bounding_boxes.csv")

    selection_for_split = selections.get(split, selections)

    for cls_name, label, target_count in specs:
        ann_path = os.path.join(ann_split_dir, f"{cls_name}.json")
        if not os.path.exists(ann_path):
            print(f"Missing annotations for {split}/{cls_name}: {ann_path}", file=sys.stderr)
            continue

        anns = load_annotations(ann_path)
        img_dir = os.path.join(ROOT, cls_name)
        available = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")}
        keys = [k for k in anns.keys() if k in available]
        random.shuffle(keys)
        if cls_name in selection_for_split:
            keys = [k for k in selection_for_split[cls_name] if k in keys]
        if len(keys) > target_count:
            keys = keys[:target_count]
        generated_selection.setdefault(split, {})[cls_name] = keys

        print(f"{split}/{cls_name}: target {target_count}, pool {len(keys)}")

        for img_id in tqdm(keys, leave=False, desc=f"{split}-{cls_name}"):
            entry = anns[img_id]
            bboxes = entry.get("bboxes", [])
            if not bboxes:
                continue

            img_path = os.path.join(ROOT, cls_name, f"{img_id}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            abs_boxes = to_abs_boxes(bboxes, w, h)
            for xmin, ymin, xmax, ymax in abs_boxes:
                records.append(
                    {
                        "dataset": "hagridv2",
                        "relative_path": img_path,
                        "label": label,
                        "width": w,
                        "height": h,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                )
            if abs_boxes:
                samples.append((img_path, abs_boxes))

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
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        rel_path = os.path.relpath(img_path, ROOT)
        out_path = os.path.join(vis_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)

    print(f"Wrote {len(records)} rows to {out_csv} and {take} visuals to {vis_dir}")
    return len(records)


def main():
    args = parse_args()
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selections = {}
    if args.selection_json:
        try:
            with open(args.selection_json, "r", encoding="utf-8") as f:
                selections = json.load(f)
        except Exception as exc:
            print(f"Failed to read selection JSON {args.selection_json}: {exc}", file=sys.stderr)
            return
    generated_selection = {}

    split_specs = {"train": TRAIN_SPECS, "val": VAL_SPECS, "test": TEST_SPECS}
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    total = 0
    for split in splits:
        total += process_split(split, split_specs[split], selections, generated_selection)

    if args.write_selection:
        with open(OUTPUT_SELECTION_JSON, "w", encoding="utf-8") as f:
            json.dump(generated_selection, f, indent=2)

    if len(splits) > 1:
        print(f"Done. Total rows across splits: {total}")


if __name__ == "__main__":
    main()
