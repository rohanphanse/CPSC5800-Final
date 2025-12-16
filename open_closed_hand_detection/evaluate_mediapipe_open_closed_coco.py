import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

DEFAULT_DATASETS = [
    {
        "name": "hagridv2_test",
        "csv": "bounding_boxes/hagridv2_subset_test_bounding_boxes.csv",
    },
    {
        "name": "open_closed_1k_all",
        "csv": "bounding_boxes/open_closed_1k_all_bounding_boxes.csv",
    },
]

MAX_HANDS = 2
MIN_DET_CONF = 0.15
BOX_EXPAND = 0.1
OPEN_THRESHOLD = 1.1


def map_label(dataset: str, lbl: int) -> int:
    if dataset == "hagridv2":
        return 1 if lbl == 1 else 0
    return 1 if lbl == 1 else 0


def load_gt(csv_path: Path) -> Tuple[List[Dict], List[Dict]]:
    images = []
    annotations = []
    img_id_map: Dict[str, int] = {}
    ann_id = 1

    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        cols = {name: idx for idx, name in enumerate(header)}
        for line in f:
            parts = line.strip().split(",")
            try:
                img_rel = parts[cols["relative_path"]]
                w = float(parts[cols["width"]])
                h = float(parts[cols["height"]])
                xmin = float(parts[cols["xmin"]])
                ymin = float(parts[cols["ymin"]])
                xmax = float(parts[cols["xmax"]])
                ymax = float(parts[cols["ymax"]])
                label = int(parts[cols["label"]])
                dataset = parts[cols.get("dataset")] if "dataset" in cols else ""
            except Exception:
                continue

            if img_rel not in img_id_map:
                img_id_map[img_rel] = len(img_id_map) + 1
                images.append(
                    {
                        "id": img_id_map[img_rel],
                        "file_name": img_rel,
                        "width": int(w),
                        "height": int(h),
                    }
                )

            bbox_w = max(0.0, xmax - xmin)
            bbox_h = max(0.0, ymax - ymin)
            if bbox_w <= 0 or bbox_h <= 0:
                continue
            mapped = map_label(dataset, label)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id_map[img_rel],
                    "category_id": 0 if mapped == 0 else 1,  # 0=open, 1=closed
                    "bbox": [xmin, ymin, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return images, annotations


def mediapipe_boxes_and_class(image, width: int, height: int) -> List[Tuple[Tuple[float, float, float, float], int, float]]:
    results = HANDS.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes = []
    if results.multi_hand_landmarks:
        handed = results.multi_handedness or []
        for idx, lm in enumerate(results.multi_hand_landmarks):
            xs = np.array([pt.x for pt in lm.landmark])
            ys = np.array([pt.y for pt in lm.landmark])
            score = 1.0
            if idx < len(handed) and handed[idx].classification:
                score = handed[idx].classification[0].score

            xmin = xs.min() * width
            ymin = ys.min() * height
            xmax = xs.max() * width
            ymax = ys.max() * height
            dx = (xmax - xmin) * BOX_EXPAND
            dy = (ymax - ymin) * BOX_EXPAND
            xmin = max(0.0, xmin - dx)
            ymin = max(0.0, ymin - dy)
            xmax = min(float(width - 1), xmax + dx)
            ymax = min(float(height - 1), ymax + dy)

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
                continue
            open_score = float(sum(ratios) / len(ratios))
            cls_id = 0 if open_score >= OPEN_THRESHOLD else 1  # 0=open, 1=closed

            boxes.append(((xmin, ymin, xmax, ymax), cls_id, float(score)))
    return boxes


def run_eval(name: str, csv_path: Path, output_dir: Path):
    images, annotations = load_gt(csv_path)
    if not images:
        print(f"{name}: no GT loaded, skipping.")
        return

    img_id_lookup = {img["file_name"]: img["id"] for img in images}
    preds = []
    missing = 0
    for img in tqdm(images, desc=f"MediaPipe {name}"):
        img_rel = img["file_name"]
        src = img_rel if os.path.isabs(img_rel) else str(Path(os.getcwd()) / img_rel)
        if not os.path.exists(src):
            missing += 1
            continue
        image = cv2.imread(src)
        if image is None:
            missing += 1
            continue
        h, w = image.shape[:2]
        dets = mediapipe_boxes_and_class(image, w, h)
        for (xmin, ymin, xmax, ymax), cls_id, score in dets:
            bw = max(0.0, xmax - xmin)
            bh = max(0.0, ymax - ymin)
            preds.append(
                {
                    "image_id": img_id_lookup[img_rel],
                    "category_id": cls_id,
                    "bbox": [float(xmin), float(ymin), float(bw), float(bh)],
                    "score": float(score),
                }
            )

    coco_gt = {
        "info": {"description": f"{name} open/closed GT"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": "open"},
            {"id": 1, "name": "closed"},
        ],
    }
    gt_path = output_dir / f"{name}_gt.json"
    dt_path = output_dir / f"{name}_dt.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(coco_gt), encoding="utf-8")
    dt_path.write_text(json.dumps(preds), encoding="utf-8")

    coco = COCO(str(gt_path))
    coco_dt = coco.loadRes(str(dt_path))
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(
        f"{name}: mAP50-95={coco_eval.stats[0]:.4f}, "
        f"mAP50={coco_eval.stats[1]:.4f}, missing images={missing}"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate MediaPipe open/closed with COCO mAP.")
    ap.add_argument("--output", type=Path, default=Path("mediapipe_coco_eval_open_closed"), help="Where to write GT/pred JSONs.")
    return ap.parse_args()


def main():
    args = parse_args()
    for ds in DEFAULT_DATASETS:
        csv_path = Path(ds["csv"])
        if not csv_path.is_absolute():
            csv_path = Path(os.getcwd()) / csv_path
        if not csv_path.exists():
            print(f"{ds['name']}: missing CSV at {csv_path}, skipping.")
            continue
        run_eval(ds["name"], csv_path, args.output)


if __name__ == "__main__":
    HANDS = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DET_CONF,
    )
    main()
    HANDS.close()
