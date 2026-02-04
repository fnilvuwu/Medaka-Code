# coding: utf-8
__author__ = "ZFTurbo: https://kaggle.com/zfturbo"

import numpy as np
import pandas as pd
import json
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from multiprocessing import Process, cpu_count, Manager
from ensemble_boxes import *
import os


def get_coco_annotations_data():
    file_in = "instances_test.json"
    images = dict()
    with open(file_in) as json_file:
        data = json.load(json_file)
        for i in range(len(data["images"])):
            image_id = data["images"][i]["id"]
            images[image_id] = data["images"][i]

    return images


def get_coco_score(csv_path):
    images = get_coco_annotations_data()
    s = pd.read_csv(csv_path, dtype={"img_id": str, "label": str})

    out = np.zeros((len(s), 7), dtype=np.float64)
    out[:, 0] = s["img_id"]
    ids = s["img_id"].astype(np.int32).values
    x1 = s["x1"].values
    x2 = s["x2"].values
    y1 = s["y1"].values
    y2 = s["y2"].values
    for i in range(len(s)):
        width = images[ids[i]]["width"]
        height = images[ids[i]]["height"]
        out[i, 1] = x1[i] * width
        out[i, 2] = y1[i] * height
        out[i, 3] = (x2[i] - x1[i]) * width
        out[i, 4] = (y2[i] - y1[i]) * height
    out[:, 5] = s["score"].values
    out[:, 6] = s["label"].values

    filename = "instances_test.json"
    coco_gt = COCO(filename)
    detections = out
    print(detections.shape)
    print(detections[:5])
    image_ids = list(set(detections[:, 0]))
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats
    print(coco_metrics)
    return coco_metrics, detections


def process_single_id(id, res_boxes, weights, params):
    run_type = params["run_type"]
    verbose = params["verbose"]

    # print('Go for ID: {}'.format(id))
    boxes_list = []
    scores_list = []
    labels_list = []
    labels_to_use_forward = dict()
    labels_to_use_backward = dict()

    for i in range(len(res_boxes[id])):
        boxes = []
        scores = []
        labels = []

        dt = res_boxes[id][i]

        for j in range(0, len(dt)):
            lbl = dt[j][5]
            scr = float(dt[j][4])
            box_x1 = float(dt[j][0])
            box_y1 = float(dt[j][1])
            box_x2 = float(dt[j][2])
            box_y2 = float(dt[j][3])

            if box_x1 >= box_x2:
                if verbose:
                    print("Problem with box x1 and x2: {}. Skip it".format(dt[j]))
                continue
            if box_y1 >= box_y2:
                if verbose:
                    print("Problem with box y1 and y2: {}. Skip it".format(dt[j]))
                continue
            if scr <= 0:
                if verbose:
                    print("Problem with box score: {}. Skip it".format(dt[j]))
                continue

            boxes.append([box_x1, box_y1, box_x2, box_y2])
            scores.append(scr)
            if lbl not in labels_to_use_forward:
                cur_point = len(labels_to_use_forward)
                labels_to_use_forward[lbl] = cur_point
                labels_to_use_backward[cur_point] = lbl
            labels.append(labels_to_use_forward[lbl])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # Empty predictions for all models
    if len(boxes_list) == 0:
        return np.array([]), np.array([]), np.array([])

    if run_type == "wbf":
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=params["intersection_thr"],
            skip_box_thr=params["skip_box_thr"],
            conf_type=params["conf_type"],
        )
    elif run_type == "wbf_exp":
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion_experimental(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=params["intersection_thr"],
            skip_box_thr=params["skip_box_thr"],
            conf_type=params["conf_type"],
        )
    elif run_type == "nms":
        iou_thr = params["iou_thr"]
        merged_boxes, merged_scores, merged_labels = nms(
            boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr
        )
    elif run_type == "soft-nms":
        iou_thr = params["iou_thr"]
        sigma = params["sigma"]
        thresh = params["thresh"]
        merged_boxes, merged_scores, merged_labels = soft_nms(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            sigma=sigma,
            thresh=thresh,
        )
    elif run_type == "nmw":
        merged_boxes, merged_scores, merged_labels = non_maximum_weighted(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=params["intersection_thr"],
            skip_box_thr=params["skip_box_thr"],
        )

    # print(len(boxes_list), len(merged_boxes))
    if "limit_boxes" in params:
        limit_boxes = params["limit_boxes"]
        if len(merged_boxes) > limit_boxes:
            merged_boxes = merged_boxes[:limit_boxes]
            merged_scores = merged_scores[:limit_boxes]
            merged_labels = merged_labels[:limit_boxes]

    # Rename labels back
    merged_labels_string = []
    for m in merged_labels:
        merged_labels_string.append(labels_to_use_backward[m])
    merged_labels = np.array(merged_labels_string, dtype=str)

    # Create IDs array
    ids_list = [id] * len(merged_labels)

    return (
        merged_boxes.copy(),
        merged_scores.copy(),
        merged_labels.copy(),
        ids_list.copy(),
    )


def process_part_of_data(
    proc_number, return_dict, ids_to_use, res_boxes, weights, params
):
    print("Start process: {} IDs to proc: {}".format(proc_number, len(ids_to_use)))
    result = []
    for id in ids_to_use:
        merged_boxes, merged_scores, merged_labels, ids_list = process_single_id(
            id, res_boxes, weights, params
        )
        # print(merged_boxes.shape, merged_scores.shape, merged_labels.shape, len(ids_list))
        result.append((merged_boxes, merged_scores, merged_labels, ids_list))
    return_dict[proc_number] = result.copy()


def ensemble_predictions(pred_filenames, weights, params):
    verbose = False
    if "verbose" in params:
        verbose = params["verbose"]

    start_time = time.time()
    procs_to_use = max(cpu_count() // 2, 1)
    # procs_to_use = 6
    print("Use processes: {}".format(procs_to_use))
    weights = np.array(weights)

    res_boxes = dict()
    ref_ids = None
    for j in range(len(pred_filenames)):
        if weights[j] == 0:
            continue
        print("Read {}...".format(pred_filenames[j]))
        s = pd.read_csv(pred_filenames[j], dtype={"img_id": str, "label": str})
        s.sort_values("img_id", inplace=True)
        s.reset_index(drop=True, inplace=True)
        ids = s["img_id"].values
        unique_ids = sorted(s["img_id"].unique())
        if ref_ids is None:
            ref_ids = tuple(unique_ids)
        else:
            if ref_ids != tuple(unique_ids):
                print(
                    "Different IDs in ensembled CSVs! {} != {}".format(
                        len(ref_ids), len(unique_ids)
                    )
                )
                s = s[s["img_id"].isin(ref_ids)]
                s.sort_values("img_id", inplace=True)
                s.reset_index(drop=True, inplace=True)
                ids = s["img_id"].values
        preds = s[["x1", "y1", "x2", "y2", "score", "label"]].values
        single_res = dict()
        for i in range(len(ids)):
            id = ids[i]
            if id not in single_res:
                single_res[id] = []
            single_res[id].append(preds[i])
        for el in single_res:
            if el not in res_boxes:
                res_boxes[el] = []
            res_boxes[el].append(single_res[el])

    # Reduce weights if needed
    weights = weights[weights != 0]

    ids_to_use = sorted(list(res_boxes.keys()))
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(procs_to_use):
        start = i * len(ids_to_use) // procs_to_use
        end = (i + 1) * len(ids_to_use) // procs_to_use
        if i == procs_to_use - 1:
            end = len(ids_to_use)
        p = Process(
            target=process_part_of_data,
            args=(i, return_dict, ids_to_use[start:end], res_boxes, weights, params),
        )
        jobs.append(p)
        p.start()

    for i in range(len(jobs)):
        jobs[i].join()

    results = []
    for i in range(len(jobs)):
        results += return_dict[i]

    # p = Pool(processes=procs_to_use)
    # results = p.starmap(process_single_id, zip(ids_to_use, repeat(weights), repeat(params)))

    all_ids = []
    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, labels, ids_list in results:
        if boxes is None:
            continue
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_ids.append(ids_list)

    all_ids = np.concatenate(all_ids)
    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    if verbose:
        print(all_ids.shape, all_boxes.shape, all_scores.shape, all_labels.shape)

    res = pd.DataFrame(all_ids, columns=["img_id"])
    res["label"] = all_labels
    res["score"] = all_scores
    res["x1"] = all_boxes[:, 0]
    res["x2"] = all_boxes[:, 2]
    res["y1"] = all_boxes[:, 1]
    res["y2"] = all_boxes[:, 3]
    print("Run time: {:.2f}".format(time.time() - start_time))
    return res



def save_ensemble_txt(
    df,
    output_dir,
    instances_json_path="instances_test.json",
    conf_thr=0.25,
):
    """
    Save YOLO-style TXT predictions for DETECTIONS (with confidence):
    Format: <class_id> <confidence> <x_center> <y_center> <width> <height>
    
    IMPORTANT: Coordinates MUST be normalized (0-1 range) for text2bb to work
    with CoordinatesType.RELATIVE and BBType.DETECTED
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO data
    with open(instances_json_path, "r") as f:
        coco = json.load(f)

    # Create mappings
    id_to_name = {}
    id_to_dims = {}
    for img in coco["images"]:
        img_id = int(img["id"])
        # Remove extension for filename matching
        img_name = os.path.splitext(img["file_name"])[0]
        id_to_name[img_id] = img_name
        id_to_dims[img_id] = (img["width"], img["height"])
    
    # Create COCO category_id to YOLO class mapping (0-indexed)
    coco_cats = sorted(coco["categories"], key=lambda x: x["id"])
    coco_id_to_yolo = {cat["id"]: idx for idx, cat in enumerate(coco_cats)}
    
    print(f"Found {len(coco_cats)} categories: {coco_cats}")
    print(f"COCO to YOLO mapping: {coco_id_to_yolo}")
    print(f"Image ID to name mapping (first 5): {dict(list(id_to_name.items())[:5])}")
    
    saved_count = 0
    skipped_count = 0
    total_boxes = 0
    
    for img_id_raw, g in df.groupby("img_id"):
        try:
            img_id = int(img_id_raw)
        except ValueError:
            print(f"Warning: Invalid img_id: {img_id_raw}")
            skipped_count += 1
            continue

        if img_id not in id_to_name:
            print(f"Warning: img_id {img_id} not found in COCO JSON")
            skipped_count += 1
            continue

        image_name = id_to_name[img_id]
        width, height = id_to_dims[img_id]
        lines = []

        for _, r in g.iterrows():
            score = float(r["score"])
            if score < conf_thr:
                continue

            # Convert label to YOLO class ID
            try:
                coco_cat_id = int(r["label"])
                if coco_cat_id not in coco_id_to_yolo:
                    print(f"Warning: Unknown category_id {coco_cat_id}")
                    continue
                yolo_cls = coco_id_to_yolo[coco_cat_id]
            except (ValueError, TypeError):
                print(f"Warning: Non-integer label: {r['label']}")
                continue

            x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
            
            # Determine if coordinates are normalized or absolute
            if x2 > 1.0 or y2 > 1.0 or x1 > 1.0 or y1 > 1.0:
                # Absolute (pixel) coordinates - normalize them
                print(f"Normalizing pixel coords: ({x1}, {y1}, {x2}, {y2})")
                x1, x2 = x1 / width, x2 / width
                y1, y2 = y1 / height, y2 / height

            # Calculate YOLO format (center_x, center_y, width, height)
            x_c = (x1 + x2) / 2.0
            y_c = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            # Validate coordinates are in valid range [0, 1]
            if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print(f"Warning: Invalid normalized coords for {image_name}: "
                      f"x_c={x_c:.4f}, y_c={y_c:.4f}, w={w:.4f}, h={h:.4f}")
                skipped_count += 1
                continue

            # Format: class_id confidence x_center y_center width height
            # All values must be space-separated, coordinates in [0,1]
            lines.append(
                f"{yolo_cls} {score:.10f} "
                f"{x_c:.10f} {y_c:.10f} {w:.10f} {h:.10f}"
            )
            total_boxes += 1

        if lines:
            output_path = os.path.join(output_dir, f"{image_name}.txt")
            with open(output_path, "w") as f:
                f.write("\n".join(lines))
            saved_count += 1

    print(f"\n{'='*60}")
    print(f"Saved {saved_count} prediction files to {output_dir}")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Skipped {skipped_count} invalid boxes/images")
    print(f"{'='*60}")
    
    # Verify a sample file
    if saved_count > 0:
        sample_files = os.listdir(output_dir)[:3]
        print(f"\nSample predictions (first 3 files):")
        for fname in sample_files:
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "r") as f:
                lines = f.readlines()
                print(f"\n{fname} ({len(lines)} boxes):")
                for line in lines[:2]:  # Show first 2 boxes
                    print(f"  {line.strip()}")
    
    return saved_count

def ensemble(benchmark_csv, weights, params, get_score_init=True):
    if get_score_init:
        for bcsv in benchmark_csv:
            print("Go for {}".format(bcsv))
            get_coco_score(bcsv)

    ensemble_preds = ensemble_predictions(benchmark_csv, weights, params)

    # 🔥 FIX score semantics for NMS / Soft-NMS
    if params["run_type"] in ["nms", "soft-nms"]:
        ensemble_preds["score"] *= len(weights)
        ensemble_preds["score"] = ensemble_preds["score"].clip(0, 1)

    ensemble_preds.to_csv("ensemble.csv", index=False)

    # NEW: also save YOLO-style TXT predictions
    save_ensemble_txt(
        ensemble_preds,
        output_dir="./ensemble_txt",
        instances_json_path="instances_test.json",
        conf_thr=0.25,  # ← YOU CONTROL THIS
    )

    get_coco_score("ensemble.csv")


if __name__ == "__main__":
    if 0:
        params = {
            "run_type": "nms",
            "iou_thr": 0.7,
            "conf_type": "avg",
            "verbose": True,
        }
    if 0:
        params = {
            "run_type": "soft-nms",
            "iou_thr": 0.7,
            "thresh": 0.0001,
            "sigma": 0.1,
            "verbose": True,
        }
    if 0:
        params = {
            "run_type": "nmw",
            "skip_box_thr": 0.000000001,
            "intersection_thr": 0.7,
            "limit_boxes": 30000,
            "verbose": True,
        }

    if 0:
        params = {
            "run_type": "wbf",
            "skip_box_thr": 0.001,
            "intersection_thr": 0.7,
            "conf_type": "avg",
            "limit_boxes": 30000,
            "verbose": False,
        }

    if 1:
        params = {
            "run_type": "wbf_exp",
            "skip_box_thr": 0.001,
            "intersection_thr": 0.7,
            "conf_type": "avg",
            "limit_boxes": 30000,
            "verbose": False,
        }

    in_dir = "./model_predictions/"
    benchmark_csv = [
        # 5 Fold
        in_dir + "YOLOv8-fold1-test.csv",
        in_dir + "YOLOv8-fold2-test.csv",
        in_dir + "YOLOv8-fold3-test.csv",
        in_dir + "YOLOv8-fold4-test.csv",
        in_dir + "YOLOv8-fold5-test.csv",
        # Different Size
        # in_dir + 'YOLOv8n-test.csv',
        # in_dir + 'YOLOv8s-test.csv',
        # in_dir + 'YOLOv8m-test.csv',
        # in_dir + 'YOLOv8l-test.csv',
        # in_dir + 'YOLOv8x-test.csv',
        # Different Version
        # in_dir + 'YOLOv8n-test.csv',
        # in_dir + 'YOLOv9t-test.csv',
        # in_dir + 'YOLOv10n-test.csv',
        # in_dir + 'YOLOv11n-test.csv',
    ]

    # Weight 5 Fold/Different Size
    weights = [1, 1, 1, 1, 1]

    # Weight Different Version
    # weights = [1, 1, 1, 1]

    assert len(benchmark_csv) == len(weights)
    ensemble(benchmark_csv, weights, params, get_score_init=False)

    # single evaluation
    # get_coco_score("./model_predictions/YOLOv8-fold1-test.csv")
    # get_coco_score("./model_predictions/YOLOv8-fold2-test.csv")
    # get_coco_score("./model_predictions/YOLOv8-fold3-test.csv")
    # get_coco_score("./model_predictions/YOLOv8-fold4-test.csv")
    # get_coco_score("./model_predictions/YOLOv8-fold5-test.csv")

    # get_coco_score("./model_predictions/YOLOv8n-test.csv")
    # get_coco_score("./model_predictions/YOLOv8s-test.csv")
    # get_coco_score("./model_predictions/YOLOv8m-test.csv")
    # get_coco_score("./model_predictions/YOLOv8l-test.csv")
    # get_coco_score("./model_predictions/YOLOv8x-test.csv")

    # get_coco_score("./model_predictions/YOLOv8n-test.csv")
    # get_coco_score("./model_predictions/YOLOv9t-test.csv")
    # get_coco_score("./model_predictions/YOLOv10n-test.csv")
    # get_coco_score("./model_predictions/YOLOv11n-test.csv")
