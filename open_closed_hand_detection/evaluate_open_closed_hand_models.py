import csv
import os
import shutil
from collections import defaultdict

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# Edit these paths if your files live elsewhere.
MODEL_PATHS = {
    "train8": "runs/detect/train8/weights/best.pt",
    "train9": "runs/detect/train9/weights/best.pt",
}

DATASETS = [
    {
        "name": "hagridv2_test",
        "csv": "bounding_boxes/hagridv2_subset_test_bounding_boxes.csv",
    },
    {
        "name": "open_closed_1k_all",
        "csv": "bounding_boxes/open_closed_1k_all_bounding_boxes.csv",
    },
]

STAGING_ROOT = "eval_datasets_open_closed_hand"
IMGSZ = 640
BATCH = 32


def ensure_clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def label_to_class(dataset, label_value):
    if dataset == "hagridv2":
        # hagridv2 fist=1 (closed); palm/stop/stop_inverted=0 (open)
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


def load_csv(csv_path):
    by_image = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get("relative_path", "")
            if not img_path:
                continue
            try:
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
                width = float(row["width"])
                height = float(row["height"])
            except (KeyError, ValueError):
                continue
            by_image[img_path].append(
                {
                    "dataset": row.get("dataset", "unknown"),
                    "label": row.get("label", 0),
                    "bbox": (xmin, ymin, xmax, ymax),
                    "width": width,
                    "height": height,
                }
            )
    return by_image


def stage_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    csv_path = dataset_cfg["csv"]
    staged_root = os.path.join(STAGING_ROOT, name)
    images_dir = os.path.join(staged_root, "images")
    labels_dir = os.path.join(staged_root, "labels")

    ensure_clean_dir(staged_root)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    samples = load_csv(csv_path)
    copied = 0

    for img_rel, boxes in tqdm(samples.items(), desc=f"Staging {name}"):
        src = img_rel if os.path.isabs(img_rel) else os.path.join(os.getcwd(), img_rel)
        if not os.path.exists(src):
            continue

        rel_no_root = img_rel.lstrip("/").replace("\\", "/")
        dst_img = os.path.join(images_dir, rel_no_root)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy2(src, dst_img)
        copied += 1

        label_lines = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box["bbox"]
            w = box["width"]
            h = box["height"]
            if w <= 0 or h <= 0:
                continue
            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            cls_id = label_to_class(box["dataset"], box["label"])
            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        if not label_lines:
            continue

        lbl_rel = os.path.splitext(rel_no_root)[0] + ".txt"
        dst_lbl = os.path.join(labels_dir, lbl_rel)
        os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(label_lines))

    yaml_path = os.path.join(staged_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            "\n".join(
                [
                    f"path: {staged_root}",
                    "train: images",
                    "val: images",
                    "names: {0: open, 1: closed}",
                    "nc: 2",
                ]
            )
        )

    return staged_root, yaml_path, copied


def evaluate_model(model_path, data_yaml):
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=IMGSZ, batch=BATCH, verbose=False)
    return {
        "map50": results.box.map50,
        "map50_95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }


def main():
    os.makedirs(STAGING_ROOT, exist_ok=True)

    staged = []
    for ds in DATASETS:
        csv_path = ds["csv"]
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.getcwd(), csv_path)
        if not os.path.exists(csv_path):
            print(f"Missing CSV for dataset {ds['name']}: {csv_path}, skipping.")
            continue
        ds_copy = dict(ds)
        ds_copy["csv"] = csv_path
        root, yaml_path, copied = stage_dataset(ds_copy)
        staged.append((ds["name"], root, yaml_path, copied))
        print(f"Staged {ds['name']} from {csv_path} -> {root} ({copied} images copied)")

    if not staged:
        print("No datasets staged; exiting.")
        return

    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skipping.")
            continue
        print(f"\nEvaluating model {model_name} ({model_path})")
        model_metrics = []
        for ds_name, _, yaml_path, _ in staged:
            metrics = evaluate_model(model_path, yaml_path)
            model_metrics.append(metrics)
            print(
                f"  {ds_name}: mAP50={metrics['map50']:.4f}, "
                f"mAP50-95={metrics['map50_95']:.4f}, "
                f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}"
            )

        if model_metrics:
            avg_map50 = sum(m["map50"] for m in model_metrics) / len(model_metrics)
            avg_map95 = sum(m["map50_95"] for m in model_metrics) / len(model_metrics)
            print(f"  Average (all datasets): mAP50={avg_map50:.4f}, mAP50-95={avg_map95:.4f}")


if __name__ == "__main__":
    main()
