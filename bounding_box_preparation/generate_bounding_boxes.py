import csv
import glob
import os
import random
import cv2
import mediapipe as mp
from tqdm import tqdm

DATASETS = [
    ("asl_1_dataset_labeled", "data/asl_1_dataset_labeled"),
    ("asl_1_dataset_raw", "data/asl_1_dataset_raw"),
    ("asl_2_dataset_labeled", "data/asl_2_dataset_labeled"),
    ("asl_2_dataset_raw", "data/asl_2_dataset_raw"),
    ("rps_dataset_labeled", "data/rps_dataset_labeled"),
    ("rps_dataset_raw", "data/rps_dataset_raw"),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
OUTPUT_DIR = "bounding_boxes"
SAMPLE_VIS_COUNT = 50

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
    for name, root in DATASETS:
        if not os.path.exists(root):
            print(f"Skip missing {root}")
            continue
        vis_dir = os.path.join(OUTPUT_DIR, "visualizations", name)
        os.makedirs(vis_dir, exist_ok=True)
        records = []
        sample_candidates = []
        pattern = os.path.join(root, "**", "*")
        files = [
            path
            for path in sorted(glob.glob(pattern, recursive=True))
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS
        ]
        for path in tqdm(files, desc=name, leave=False):
            image = cv2.imread(path)
            if image is None:
                continue
            h, w = image.shape[:2]
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            boxes = []
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    xs = [p.x for p in lm.landmark]
                    ys = [p.y for p in lm.landmark]
                    boxes.append({
                        "xmin": int(min(xs) * w),
                        "ymin": int(min(ys) * h),
                        "xmax": int(max(xs) * w),
                        "ymax": int(max(ys) * h),
                    })
            if not boxes:
                continue
            rel_path = os.path.relpath(path, root)
            records.append({
                "dataset": name,
                "relative_path": rel_path,
                "label": os.path.basename(os.path.dirname(path)),
                "width": w,
                "height": h,
                "bounding_boxes": boxes,
            })
            sample_candidates.append((path, rel_path, boxes))
        if not records:
            continue
        take = min(len(sample_candidates), SAMPLE_VIS_COUNT)
        for path, rel_path, boxes in random.sample(sample_candidates, take):
            image = cv2.imread(path)
            if image is None:
                continue
            annotated = image.copy()
            for b in boxes:
                cv2.rectangle(annotated, (b["xmin"], b["ymin"]), (b["xmax"], b["ymax"]), (0, 255, 0), 2)
            out_path = os.path.join(vis_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, annotated)
        csv_path = os.path.join(OUTPUT_DIR, f"{name}_bounding_boxes.csv")
        fields = ["dataset", "relative_path", "label", "width", "height", "xmin", "ymin", "xmax", "ymax"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fields)
            csv_writer.writeheader()
            for r in records:
                base = {
                    "dataset": r["dataset"],
                    "relative_path": r["relative_path"],
                    "label": r["label"],
                    "width": r["width"],
                    "height": r["height"],
                }
                for b in r["bounding_boxes"]:
                    csv_writer.writerow({
                        **base,
                        "xmin": b.get("xmin", ""),
                        "ymin": b.get("ymin", ""),
                        "xmax": b.get("xmax", ""),
                        "ymax": b.get("ymax", ""),
                    })
        print(f"{name}: {len(records)} images -> {csv_path}")

if __name__ == "__main__":
    main()
