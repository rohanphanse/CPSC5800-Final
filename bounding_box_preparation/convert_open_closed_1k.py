import csv
import json
import os
import random

import cv2
from tqdm import tqdm


BASE_DIR = "data/open-closed-1k"
OUTPUT_DIR = "bounding_boxes"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
# (directory name in dataset, split name to write in CSV)
SPLITS = [("train", "train"), ("valid", "val"), ("test", "test")]
SAMPLES_PER_SPLIT = 50
SEED = 42

# COCO category_id -> label (open=0, closed=1)
LABEL_MAP = {2: 0, 1: 1}
LABEL_NAMES = {0: "open", 1: "closed"}


def load_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = {}

    for ann in data.get("annotations", []):
        cid = ann.get("category_id")
        if cid not in LABEL_MAP:
            continue
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            continue
        img_id = ann.get("image_id")
        anns_by_image.setdefault(img_id, []).append(
            {
                "label": LABEL_MAP[cid],
                "bbox": (x, y, x + w, y + h),
            }
        )

    return images, anns_by_image


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "relative_path", "label", "width", "height", "xmin", "ymin", "xmax", "ymax"]
        )
        writer.writerows(rows)


def visualize(samples, anns_by_image, images_by_id, dir_split, csv_split):
    if not samples:
        return
    out_dir = os.path.join(VIS_DIR, f"open_closed_1k_{csv_split}")
    os.makedirs(out_dir, exist_ok=True)
    for img_id in tqdm(samples, desc=f"Visualizing {dir_split}", leave=False):
        img_info = images_by_id[img_id]
        img_path = os.path.join(BASE_DIR, dir_split, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            continue
        for ann in anns_by_image.get(img_id, []):
            xmin, ymin, xmax, ymax = ann["bbox"]
            color = (0, 255, 0) if ann["label"] == 0 else (0, 0, 255)
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(
                image,
                LABEL_NAMES[ann["label"]],
                (int(xmin), int(max(ymin - 5, 0))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        out_path = os.path.join(out_dir, img_info["file_name"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, image)


def main():
    random.seed(SEED)
    summary = []
    combined_rows = []

    for dir_split, csv_split in SPLITS:
        ann_path = os.path.join(BASE_DIR, dir_split, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            print(f"Missing annotations for {dir_split}, skipping.")
            continue

        images_by_id, anns_by_image = load_annotations(ann_path)

        rows = []
        img_ids = list(images_by_id.keys())
        annotated_ids = list(anns_by_image.keys())
        for img_id in tqdm(img_ids, desc=f"Processing {dir_split}"):
            if img_id not in anns_by_image:
                continue
            img_info = images_by_id[img_id]
            fname = img_info["file_name"]
            width = img_info.get("width")
            height = img_info.get("height")
            image_field = f"{BASE_DIR}/{dir_split}/{fname}"

            for ann in anns_by_image[img_id]:
                xmin, ymin, xmax, ymax = ann["bbox"]
                rows.append(
                    [
                        "open_closed_1k",
                        image_field,
                        ann["label"],
                        width,
                        height,
                        int(xmin),
                        int(ymin),
                        int(xmax),
                        int(ymax),
                    ]
                )

        csv_path = os.path.join(OUTPUT_DIR, f"open_closed_1k_{csv_split}_bounding_boxes.csv")
        write_csv(rows, csv_path)
        combined_rows.extend(rows)

        sample_count = min(SAMPLES_PER_SPLIT, len(annotated_ids))
        sample_ids = random.sample(annotated_ids, sample_count) if sample_count else []
        visualize(sample_ids, anns_by_image, images_by_id, dir_split, csv_split)

        summary.append((csv_split, len(img_ids), len(rows)))

    if combined_rows:
        combined_csv = os.path.join(OUTPUT_DIR, "open_closed_1k_all_bounding_boxes.csv")
        write_csv(combined_rows, combined_csv)

    for split, img_count, box_count in summary:
        print(f"{split}: {img_count} images, {box_count} boxes written")


if __name__ == "__main__":
    main()
