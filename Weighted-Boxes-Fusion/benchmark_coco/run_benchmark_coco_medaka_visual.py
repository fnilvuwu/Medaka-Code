# coding: utf-8
__author__ = "ZFTurbo: https://kaggle.com/zfturbo"

import numpy as np
import pandas as pd
import json
import time
import os
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from multiprocessing import Process, cpu_count, Manager
from ensemble_boxes import *


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


def visualize_results(
    pred_csv, output_dir="./output_images", confidence_threshold=0.25
):
    """
    Visualize detection results and save images with bounding boxes
    Shows class name and uses different colors per class
    """

    os.makedirs(output_dir, exist_ok=True)

    # Class mapping
    CLASS_NAMES = {
        1: "o.celebensis",
        2: "o.javanicus",
    }

    # BGR colors (OpenCV uses BGR, not RGB)
    CLASS_COLORS = {
        1: (0, 255, 0),  # Green -> o.celebensis
        2: (255, 0, 0),  # Blue  -> o.javanicus
    }

    # Get image metadata from COCO annotations
    images_data = get_coco_annotations_data()

    # Read predictions
    s = pd.read_csv(pred_csv, dtype={"img_id": str})

    for img_id in s["img_id"].unique():
        img_preds = s[s["img_id"] == img_id]

        img_info = images_data[int(img_id)]
        img_file = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        img_path = os.path.join("images", img_file)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image {img_path}")
            continue

        for _, row in img_preds.iterrows():
            score = float(row["score"])
            if score < confidence_threshold:
                continue

            # Convert label safely to int
            label_id = int(row["label"])
            class_name = CLASS_NAMES[label_id]
            color = CLASS_COLORS[label_id]

            # Convert normalized coords → pixel coords
            x1 = int(row["x1"] * width)
            y1 = int(row["y1"] * height)
            x2 = int(row["x2"] * width)
            y2 = int(row["y2"] * height)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            # Label text
            text = f"{class_name}: {score:.2f}"

            # Text background for readability
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                image,
                (x1, y1 - th - 6),
                (x1 + tw + 4, y1),
                color,
                -1,
            )

            cv2.putText(
                image,
                text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, image)
        print(f"Saved annotated image: {output_path}")


def evaluate_single_model(
    csv_path, visualize=False, output_dir="./output_images_single"
):
    """
    Evaluate a single model prediction file and optionally visualize the results

    Args:
        csv_path: Path to the CSV file with predictions
        visualize: Whether to visualize detection results
        output_dir: Directory to save visualized images

    Returns:
        metrics: COCO evaluation metrics
    """
    # Evaluate model
    print(f"Evaluating {csv_path}")
    metrics, _ = get_coco_score(csv_path)

    # Visualize if requested
    if visualize:
        model_name = os.path.basename(csv_path).split(".")[0]
        vis_output_dir = os.path.join(output_dir, model_name)
        visualize_results(csv_path, vis_output_dir, confidence_threshold=0.25)
        print(f"Visualization saved to {vis_output_dir}")

    return metrics


def ensemble(
    benchmark_csv,
    weights,
    params,
    get_score_init=True,
    visualize=True,
    output_dir="./output_images",
):
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
    metrics, _ = get_coco_score("ensemble.csv")

    # Visualize and save annotated images with 0.25 confidence threshold
    if visualize:
        visualize_results("ensemble.csv", output_dir, confidence_threshold=0.25)

    return metrics


if __name__ == "__main__":
    if 0:
        params = {
            "run_type": "nms",
            "iou_thr": 0.7,
            "verbose": True,
        }
    if 1:
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

    if 0:
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
        # in_dir + "YOLOv8-fold1-test.csv",
        # in_dir + "YOLOv8-fold2-test.csv",
        # in_dir + "YOLOv8-fold3-test.csv",
        # in_dir + "YOLOv8-fold4-test.csv",
        # in_dir + "YOLOv8-fold5-test.csv",

        # Different Size
        in_dir + 'YOLOv8n-test.csv',
        in_dir + 'YOLOv8s-test.csv',
        in_dir + 'YOLOv8m-test.csv',
        in_dir + 'YOLOv8l-test.csv',
        in_dir + 'YOLOv8x-test.csv',

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
    ensemble(
        benchmark_csv,
        weights,
        params,
        get_score_init=False,
        visualize=True,
        output_dir="./output_images_ensemble_soft_nms_different_size",  # Changed output directory name for clarity
    )

    # Single model evaluation with visualization
    # Uncomment and modify the models you want to visualize individually
    # evaluate_single_model("./model_predictions/YOLOv8-fold1-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv8-fold2-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv8-fold3-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv8-fold4-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv8-fold5-test.csv", visualize=True)

    # You can also evaluate and visualize other models
    # evaluate_single_model("./model_predictions/YOLOv8n-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv8s-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv8m-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv8l-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv8x-test.csv", visualize=True)

    # evaluate_single_model("./model_predictions/YOLOv9t-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv10n-test.csv", visualize=True)
    # evaluate_single_model("./model_predictions/YOLOv11n-test.csv", visualize=True)
