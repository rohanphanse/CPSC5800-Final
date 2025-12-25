import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "open_closed_classification"
YOLO_ROOT = REPO_ROOT / "yolo_open_closed_hand" / "images"

CSV_MAP = {
    "train": [
        REPO_ROOT / "bounding_boxes" / "hagridv2_subset_train_bounding_boxes.csv",
        REPO_ROOT / "bounding_boxes" / "asl_1_dataset_labeled_bounding_boxes.csv",
        REPO_ROOT / "bounding_boxes" / "asl_2_dataset_labeled_bounding_boxes.csv",
        REPO_ROOT / "bounding_boxes" / "rps_dataset_labeled_bounding_boxes.csv",
    ],
    "val": [
        REPO_ROOT / "bounding_boxes" / "hagridv2_subset_val_bounding_boxes.csv",
    ],
}

DATASET_BASES = {
    "hagridv2": {
        "train": YOLO_ROOT / "train" / "hagridv2",
        "val": YOLO_ROOT / "val" / "hagridv2"
    },
    "asl_1_dataset_labeled": {
        "train": YOLO_ROOT / "train" / "asl_1_dataset_labeled",
        "val": YOLO_ROOT / "val" / "asl_1_dataset_labeled",
    },
    "asl_2_dataset_labeled": {
        "train": YOLO_ROOT / "train" / "asl_2_dataset_labeled",
        "val": YOLO_ROOT / "val" / "asl_2_dataset_labeled",
    },
    "rps_dataset_labeled": {
        "train": YOLO_ROOT / "train" / "rps_dataset_labeled",
        "val": YOLO_ROOT / "val" / "rps_dataset_labeled"
    },
}

CLASS_NAMES = {0: "open", 1: "closed"}

def load_csv_records(csv_path):
    records = []
    if not csv_path.exists():
        print(f"Skip missing CSV: {csv_path}")
        return records
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"load {csv_path.name}", leave=False):
            try:
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
                w = float(row["width"])
                h = float(row["height"])
            except (KeyError, ValueError):
                continue
            records.append(
                {
                    "dataset": row.get("dataset", ""),
                    "relative_path": row.get("relative_path", ""),
                    "label": row.get("label", ""),
                    "width": w,
                    "height": h,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "source_csv": csv_path.name,
                }
            )
    return records


def resolve_image_path(rel_path, dataset, split):
    rp = Path(rel_path)
    base_map = DATASET_BASES.get(dataset)
    if base_map:
        base = base_map.get(split)
        if base:
            parts = rp.parts
            if len(parts) > 1 and parts[0] in ("data", "data_good") and parts[1] == dataset:
                rp = Path(*parts[2:])
            elif len(parts) > 0 and parts[0] == dataset:
                rp = Path(*parts[1:])
            try:
                base_rel = base.relative_to(REPO_ROOT)
                if rp.parts[: len(base_rel.parts)] == base_rel.parts:
                    rp = rp.relative_to(base_rel)
            except ValueError:
                pass
            cand = (base / rp).resolve()
            return cand if cand.exists() else None
    return None


def clamp_bbox(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0.0, min(xmin, w - 1))
    ymin = max(0.0, min(ymin, h - 1))
    xmax = max(0.0, min(xmax, w - 1))
    ymax = max(0.0, min(ymax, h - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return int(xmin), int(ymin), int(xmax), int(ymax)


def label_id(dataset, label_value):
    if dataset == "hagridv2":
        try:
            lv = int(label_value)
            return 1 if lv == 1 else 0
        except ValueError:
            return 0
    try:
        lv = int(label_value)
        return 1 if lv == 1 else 0
    except ValueError:
        return 0


def process_split(split, records):
    out_root = OUTPUT_ROOT / split
    for cls in CLASS_NAMES.values():
        (out_root / cls).mkdir(parents=True, exist_ok=True)

    per_image_counts = defaultdict(int)
    written = 0
    for rec in tqdm(records, desc=f"write {split}", leave=False):
        rel_path = rec.get("relative_path", "")
        dataset = rec.get("dataset", "")
        if not rel_path:
            continue
        src = resolve_image_path(rel_path, dataset, split)
        if src is None or not src.exists():
            continue

        image = cv2.imread(str(src))
        if image is None:
            continue
        ih, iw = image.shape[:2]

        bbox = rec.get("bbox")
        if not bbox:
            continue
        clamped = clamp_bbox(bbox[0], bbox[1], bbox[2], bbox[3], iw, ih)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped
        crop = image[y1:y2, x1:x2]

        cls_id = label_id(dataset, rec.get("label"))
        cls_name = CLASS_NAMES.get(cls_id, "open")

        rp = Path(rel_path)
        fname = rp.stem
        ext = rp.suffix if rp.suffix else ".jpg"
        per_image_counts[rel_path] += 1
        idx = per_image_counts[rel_path] - 1
        dest = out_root / cls_name / dataset / f"{fname}_{idx}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), crop)
        written += 1

    print(f"{split}: wrote {written} crops to {out_root}")


def main():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split, csvs in CSV_MAP.items():
        all_records = []
        for csv_path in csvs:
            all_records.extend(load_csv_records(csv_path))
        if not all_records:
            print(f"{split}: no records, skipping.")
            continue
        process_split(split, all_records)
    print(f"Done. Classification crops at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
