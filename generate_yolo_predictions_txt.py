#!/usr/bin/env python3
"""
Run inference using multiple YOLO models on multiple test datasets.
Predictions are saved per-image as TXT files.
"""

import os
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


# =========================================================
# MODEL CONFIGS
# =========================================================
MODEL_CONFIGS = [
    {
        "name": "YOLOv8n",
        "path": "Ensemble Models/yolov8_ensemble_different_size/yolov8n.pt",
    },
    {
        "name": "YOLOv8s",
        "path": "Ensemble Models/yolov8_ensemble_different_size/yolov8s.pt",
    },
    {
        "name": "YOLOv8m",
        "path": "Ensemble Models/yolov8_ensemble_different_size/yolov8m.pt",
    },
    {
        "name": "YOLOv8l",
        "path": "Ensemble Models/yolov8_ensemble_different_size/yolov8l.pt",
    },
    {
        "name": "YOLOv8x",
        "path": "Ensemble Models/yolov8_ensemble_different_size/yolov8x.pt",
    },
    {
        "name": "YOLOv9t",
        "path": "Ensemble Models/yolo_ensemble_different_version/yolov9t.pt",
    },
    {
        "name": "YOLOv10n",
        "path": "Ensemble Models/yolo_ensemble_different_version/yolov10n.pt",
    },
    {
        "name": "YOLOv11n",
        "path": "Ensemble Models/yolo_ensemble_different_version/yolov11n.pt",
    },
    {
        "name": "YOLOv8-fold1",
        "path": "Ensemble Models/yolov8_ensemble_5_fold/yolov8n_fold1.pt",
    },
    {
        "name": "YOLOv8-fold2",
        "path": "Ensemble Models/yolov8_ensemble_5_fold/yolov8n_fold2.pt",
    },
    {
        "name": "YOLOv8-fold3",
        "path": "Ensemble Models/yolov8_ensemble_5_fold/yolov8n_fold3.pt",
    },
    {
        "name": "YOLOv8-fold4",
        "path": "Ensemble Models/yolov8_ensemble_5_fold/yolov8n_fold4.pt",
    },
    {
        "name": "YOLOv8-fold5",
        "path": "Ensemble Models/yolov8_ensemble_5_fold/yolov8n_fold5.pt",
    },
]

# =========================================================
# DATASETS
# =========================================================
TEST_DATASETS = [
    {
        "name": "test",
        "path": "Secret1.v19-experiment-1-balance.yolov8/test/images",
        "suffix": "-test",
    },
    {
        "name": "test-horizontal-mirror",
        "path": "Secret1.v19-experiment-1-balance.yolov8/test-horizontal-mirror/images",
        "suffix": "-test-horizontal-mirror",
    },
    {
        "name": "test-vertical-mirror",
        "path": "Secret1.v19-experiment-1-balance.yolov8/test-vertical-mirror/images",
        "suffix": "-test-vertical-mirror",
    },
    {
        "name": "test-rotate-90-left",
        "path": "Secret1.v19-experiment-1-balance.yolov8/test-rotate-90-left/images",
        "suffix": "-test-rotate-90-left",
    },
    {
        "name": "test-rotate-90-right",
        "path": "Secret1.v19-experiment-1-balance.yolov8/test-rotate-90-right/images",
        "suffix": "-test-rotate-90-right",
    },
]


# =========================================================
# UTILS
# =========================================================


def normalize(x1, y1, x2, y2, w, h):
    return x1 / w, y1 / h, x2 / w, y2 / h


def inverse_h_mirror(x1, y1, x2, y2, w):
    return w - x2, y1, w - x1, y2


def inverse_v_mirror(x1, y1, x2, y2, h):
    return x1, h - y2, x2, h - y1


def inverse_rot_left(x1, y1, x2, y2, w):
    return w - y2, x1, w - y1, x2


def inverse_rot_right(x1, y1, x2, y2, h):
    return y1, h - x2, y2, h - x1


def save_predictions(preds, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for img_name, dets in preds.items():
        if not dets:
            continue

        lines = []
        for det in dets:
            # Enforce EXACT format: 6 values
            if len(det) != 6:
                continue

            cls, conf, x, y, w, h = det

            # Clamp values (important for strict evaluators)
            conf = min(max(conf, 0.0), 1.0)
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            if w <= 0 or h <= 0:
                continue

            lines.append(
                f"{cls} {conf:.6f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            )

        if not lines:
            continue

        # NO trailing newline (strict app requirement)
        with open(os.path.join(out_dir, f"{img_name}.txt"), "w") as f:
            f.write("\n".join(lines))
            
# =========================================================
# INFERENCE
# =========================================================
def run_model(model_path, image_dir, conf=0.25):
    preds = {}

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return preds

    model = YOLO(model_path)

    images = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    for img_name in tqdm(images, desc="Inference"):
        img_path = os.path.join(image_dir, img_name)

        transform = "none"
        original = img_name

        if img_name.startswith("h_mirror_"):
            original = img_name[9:]
            transform = "h"
        elif img_name.startswith("v_mirror_"):
            original = img_name[9:]
            transform = "v"
        elif img_name.startswith("rot90left_"):
            original = img_name[10:]
            transform = "l"
        elif img_name.startswith("rot90right_"):
            original = img_name[11:]
            transform = "r"

        image_key = os.path.splitext(original)[0]  # filename without extension

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        results = model(img_path, conf=conf, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box, conf_, cls in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box

                if transform == "h":
                    x1, y1, x2, y2 = inverse_h_mirror(x1, y1, x2, y2, w)
                elif transform == "v":
                    x1, y1, x2, y2 = inverse_v_mirror(x1, y1, x2, y2, h)
                elif transform == "l":
                    x1, y1, x2, y2 = inverse_rot_left(x1, y1, x2, y2, w)
                elif transform == "r":
                    x1, y1, x2, y2 = inverse_rot_right(x1, y1, x2, y2, h)

                x_c, y_c, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)

                preds.setdefault(image_key, []).append(
                    (
                        int(cls),  # class_id (0-based)
                        float(conf_),  # confidence
                        float(x_c),
                        float(y_c),
                        float(bw),
                        float(bh),
                    )
                )
    return preds


# =========================================================
# MAIN
# =========================================================
def main():

    output_root = "model_predictions"
    os.makedirs(output_root, exist_ok=True)

    for model_cfg in MODEL_CONFIGS:
        for dataset_cfg in TEST_DATASETS:
            print(f"\n▶ {model_cfg['name']} | {dataset_cfg['name']}")

            preds = run_model(
                model_cfg["path"],
                dataset_cfg["path"],
            )

            out_dir = os.path.join(
                output_root, f"{model_cfg['name']}{dataset_cfg['suffix']}"
            )
            save_predictions(preds, out_dir)

            print(f"✔ Saved {len(preds)} images → {out_dir}")

    print("\n✅ ALL INFERENCE COMPLETE")


if __name__ == "__main__":
    main()
