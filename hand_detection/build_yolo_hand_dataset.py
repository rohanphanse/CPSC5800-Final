import os
from collections import defaultdict
from pathlib import Path

import csv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "yolo_hand"

CSV_MAP = {
    "train": [
        REPO_ROOT / "bounding_boxes" / "coco_hand_big_train_bounding_boxes.csv",
        REPO_ROOT / "bounding_boxes" / "hagridv2_subset_train_bounding_boxes.csv",
        REPO_ROOT / "bounding_boxes" / "asl_1_dataset_raw_bounding_boxes.csv",
        REPO_ROOT / "bounding_boxes" / "asl_2_dataset_raw_bounding_boxes.csv",
        REPO_ROOT / "bounding_boxes" / "rps_dataset_raw_bounding_boxes.csv",
    ],
    "val": [
        REPO_ROOT / "bounding_boxes" / "coco_hand_big_val_bounding_boxes.csv",
        REPO_ROOT / "bounding_boxes" / "hagridv2_subset_val_bounding_boxes.csv",
    ],
}

SPLITS = ["train", "val"]
CLASS_ID = 0 

DATASET_BASES = {
    "COCO-Hand-Big": REPO_ROOT / "data" / "coco" / "COCO-Hand" / "COCO-Hand-Big" / "COCO-Hand-Big_Images",
    "hagridv2": REPO_ROOT / "data" / "hagridv2",
    "asl_1_dataset_raw": REPO_ROOT / "data" / "asl_1_dataset_raw",
    "asl_2_dataset_raw": REPO_ROOT / "data" / "asl_2_dataset_raw",
    "rps_dataset_raw": REPO_ROOT / "data" / "rps_dataset_raw",
}


def ensure_dirs():
    for split in SPLITS:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


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
                    "dataset": row.get("dataset", "unknown"),
                    "relative_path": row.get("relative_path", ""),
                    "width": w,
                    "height": h,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "source_csv": csv_path.name,
                }
            )
    return records


def gather_split(records):
    grouped = {}
    for rec in records:
        img_key = rec["relative_path"]
        if not img_key:
            continue
        entry = grouped.setdefault(
            img_key,
            {
                "dataset": rec["dataset"],
                "path": rec["relative_path"],
                "width": rec["width"],
                "height": rec["height"],
                "bboxes": [],
                "sources": set(),
            },
        )
        entry["bboxes"].append(rec["bbox"])
        entry["sources"].add(rec.get("source_csv", ""))
    return grouped


def resolve_image_path(rel_path, dataset):
    rp = Path(rel_path)
    base = DATASET_BASES.get(dataset)
    if base:
        base_rel = base.relative_to(REPO_ROOT)
        if rp.parts[: len(base_rel.parts)] == base_rel.parts:
            rp = rp.relative_to(base_rel)
        cand = (base / rp).resolve()
        return cand if cand.exists() else None
    return None


def normalize_bbox(bbox, w, h):
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return x_center, y_center, bw, bh


def copy_image(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.write_bytes(src.read_bytes())


def write_split(split, grouped):
    image_list = []
    bbox_count = 0
    for rel_path, info in tqdm(grouped.items(), desc=f"write {split}", leave=False):
        rp = Path(rel_path)
        base = DATASET_BASES.get(info["dataset"])
        rel_sub = Path(info["dataset"]) / rp 
        if base:
            base_rel = base.relative_to(REPO_ROOT)
            if rp.parts[: len(base_rel.parts)] == base_rel.parts:
                rel_sub = Path(info["dataset"]) / rp.relative_to(base_rel)

        src = resolve_image_path(rel_path, info["dataset"])
        if src is None:
            print(f"Missing image for {rel_path} (dataset {info['dataset']}), skipped.")
            continue

        dest_img = OUTPUT_ROOT / "images" / split / rel_sub
        copy_image(src, dest_img)

        dest_lbl = OUTPUT_ROOT / "labels" / split / rel_sub.with_suffix(".txt")
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)
        with dest_lbl.open("w", encoding="utf-8") as f:
            for bbox in info["bboxes"]:
                x_center, y_center, bw, bh = normalize_bbox(bbox, info["width"], info["height"])
                f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
                bbox_count += 1

        image_list.append(dest_img.resolve())

    list_path = OUTPUT_ROOT / f"{split}.txt"
    with list_path.open("w", encoding="utf-8") as f:
        for p in image_list:
            f.write(str(p) + "\n")
    print(f"{split}: wrote {len(image_list)} images, {bbox_count} boxes; list -> {list_path}")


def write_data_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"
    train_txt = (OUTPUT_ROOT / "train.txt").resolve()
    val_txt = (OUTPUT_ROOT / "val.txt").resolve()
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_ROOT.resolve()}",
                f"train: {train_txt}",
                f"val: {val_txt}",
                "names:",
                "  0: hand",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote data.yaml -> {yaml_path}")


def main():
    ensure_dirs()

    for split in SPLITS:
        all_records = []
        for csv_path in CSV_MAP.get(split, []):
            all_records.extend(load_csv_records(csv_path))
        grouped = gather_split(all_records)
        write_split(split, grouped)
        per_dataset_counts = defaultdict(lambda: {"images": set(), "boxes": 0, "sources": set()})
        for rec in all_records:
            per_dataset_counts[rec["dataset"]]["images"].add(rec["relative_path"])
            per_dataset_counts[rec["dataset"]]["boxes"] += 1
            per_dataset_counts[rec["dataset"]]["sources"].add(rec.get("source_csv", ""))
        if per_dataset_counts:
            print(f"Summary for {split}:")
            for ds, vals in per_dataset_counts.items():
                print(
                    f"  {ds}: images={len(vals['images'])}, boxes={vals['boxes']}, sources={','.join(sorted(vals['sources']))}"
                )

    write_data_yaml()
    print("Done building YOLO hand dataset.")


if __name__ == "__main__":
    main()
