import pandas as pd
from pycocotools.coco import COCO


# --------------------------------------------------
# IoU computation
# --------------------------------------------------
def box_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------
# Load GT
# --------------------------------------------------
def load_gt_boxes(coco_json):
    coco = COCO(coco_json)
    gt = {}

    for img_id in coco.getImgIds():
        gt[img_id] = []
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        for a in anns:
            x, y, w, h = a["bbox"]
            gt[img_id].append(
                {
                    "box": [x, y, x + w, y + h],
                    "label": str(a["category_id"]),
                    "used": False,
                }
            )
    return coco, gt


# --------------------------------------------------
# YOLO-style PR/F1
# --------------------------------------------------
def yolo_pr_f1(pred_csv, coco_json, conf_thr=0.25, iou_thr=0.7):
    coco, gt_boxes = load_gt_boxes(coco_json)
    preds = pd.read_csv(pred_csv)

    preds["img_id"] = preds["img_id"].astype(int)
    preds["label"] = preds["label"].astype(str)

    TP = FP = FN = 0

    for img_id, img_preds in preds.groupby("img_id"):
        gts = gt_boxes.get(img_id, [])
        for g in gts:
            g["used"] = False

        img_info = coco.loadImgs(img_id)[0]
        W, H = img_info["width"], img_info["height"]

        img_preds = img_preds[img_preds["score"] >= conf_thr]

        for _, p in img_preds.iterrows():
            pred_box = [
                p["x1"] * W,
                p["y1"] * H,
                p["x2"] * W,
                p["y2"] * H,
            ]
            pred_label = p["label"]

            best_iou, best_gt = 0, None
            for gt in gts:
                if gt["used"] or gt["label"] != pred_label:
                    continue
                iou = box_iou(pred_box, gt["box"])
                if iou > best_iou:
                    best_iou, best_gt = iou, gt

            if best_iou >= iou_thr:
                TP += 1
                best_gt["used"] = True
            else:
                FP += 1

        FN += sum(not g["used"] for g in gts)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1


# --------------------------------------------------
# Markdown writer
# --------------------------------------------------
def write_md_section(f, title, rows):
    f.write(f"## {title}\n\n")
    f.write("| Model | Precision | Recall | F1 |\n")
    f.write("|------|-----------|--------|----|\n")
    for r in rows:
        f.write(f"| {r['name']} | {r['p']:.3f} | {r['r']:.3f} | {r['f1']:.3f} |\n")
    f.write("\n")


# --------------------------------------------------
# Run & export .md
# --------------------------------------------------
if __name__ == "__main__":

    groups = {
        "WBF - 5 Fold Ensemble": [
            "ensemble_wbf_5_fold-test.csv",
        ],
        "WBF - Different Model Sizes": [
            "ensemble_wbf_diff_size-test.csv",
        ],
        "WBF - Different Model Version": [
            "ensemble_wbf_diff_version-test.csv",
        ],
        "NMS - 5 Fold Ensemble": [
            "ensemble_nms_5_fold-test.csv",
            "ensemble_soft-nms_5_fold-test.csv",
        ],
        "NMS - Different Model Sizes": [
            "ensemble_nms_diff_size-test.csv",
            "ensemble_soft-nms_diff_size-test.csv",
        ],
        "NMS - Different Model Version": [
            "ensemble_nms_diff_version-test.csv",
            "ensemble_soft-nms_diff_version-test.csv",
        ],
        "NMW - 5 Fold Ensemble": [
            "ensemble_nmw_5_fold-test.csv",
        ],
        "NMW - Different Model Sizes": [
            "ensemble_nmw_diff_size-test.csv",
        ],
        "NMW - Different Model Sizes": [
            "ensemble_nmw_diff_version-test.csv",
        ],
        "Single Models - 5 Fold": [
            "YOLOv8-fold1-test.csv",
            "YOLOv8-fold2-test.csv",
            "YOLOv8-fold3-test.csv",
            "YOLOv8-fold4-test.csv",
            "YOLOv8-fold5-test.csv",
        ],
        "Single Models - Different Sizes (YOLOv8n,YOLOv8s,YOLOv8m,YOLOv8l,YOLOv8x)": [
            "YOLOv8n-test.csv",
            "YOLOv8s-test.csv",
            "YOLOv8m-test.csv",
            "YOLOv8l-test.csv",
            "YOLOv8x-test.csv",
        ],
        "Single YOLO - Different Version (YOLOv8n,YOLOv9t,YOLOv10n,YOLOv11n)": [
            "YOLOv8n-test.csv",
            "YOLOv9t-test.csv",
            "YOLOv10n-test.csv",
            "YOLOv11n-test.csv",
        ],
    }

    md_path = "result_precision_recall_f1.md"

    with open(md_path, "w") as f:
        f.write("# YOLO-style Precision / Recall / F1\n\n")
        f.write("- Confidence threshold: **0.25**\n")
        f.write("- IoU threshold: **0.7**\n\n")

        for group_name, models in groups.items():
            rows = []
            for m in models:
                p, r, f1 = yolo_pr_f1(
                    f"./model_predictions/{m}",
                    coco_json="instances_test.json",
                    conf_thr=0.25,
                    iou_thr=0.7,
                )
                rows.append({"name": m, "p": p, "r": r, "f1": f1})

            write_md_section(f, group_name, rows)

    print(f"Markdown saved to: {md_path}")
