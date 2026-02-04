#!/usr/bin/env python3
"""
Script to generate predictions from 14 different YOLO models on two test datasets.
Generates 28 CSV files with predictions in the required format.
"""

import os
import json
import csv
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Define model paths and names
MODEL_CONFIGS = [
    # Different YOLO versions
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
    # YOLOv8 5-fold models
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
    # Main YOLOv8 model
    # {"name": "YOLOv8", "path": "yolov8_best_medaka_detector_v4.pt"},
]

# Define test datasets
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


def load_image_id_mapping(instances_json_path):
    """Load mapping from filename to image ID from instances_test.json"""
    with open(instances_json_path, "r") as f:
        data = json.load(f)

    filename_to_id = {}
    for image in data["images"]:
        filename_to_id[image["file_name"]] = image["id"]

    return filename_to_id


def normalize_coordinates(x1, y1, x2, y2, img_width, img_height):
    """Normalize bounding box coordinates to [0,1] range"""
    x1_norm = x1 / img_width
    y1_norm = y1 / img_height
    x2_norm = x2 / img_width
    y2_norm = y2 / img_height
    return x1_norm, y1_norm, x2_norm, y2_norm


def inverse_transform_horizontal_mirror(x1, y1, x2, y2, img_width, img_height):
    """
    Inverse transform bounding box coordinates from mirrored image back to original coordinate system.
    For horizontal mirroring: x_original = img_width - x_mirrored
    """
    # Inverse transform x coordinates (flip horizontally)
    x1_original = img_width - x2
    x2_original = img_width - x1

    # y coordinates remain the same
    y1_original = y1
    y2_original = y2

    # Ensure x1 < x2 for consistency
    if x1_original > x2_original:
        x1_original, x2_original = x2_original, x1_original

    return x1_original, y1_original, x2_original, y2_original


def inverse_transform_vertical_mirror(x1, y1, x2, y2, img_width, img_height):
    """
    Inverse transform bounding box coordinates from vertically mirrored image back to original coordinate system.
    For vertical mirroring: y_original = img_height - y_mirrored
    """
    # x coordinates remain the same
    x1_original = x1
    x2_original = x2

    # Inverse transform y coordinates (flip vertically)
    y1_original = img_height - y2
    y2_original = img_height - y1

    # Ensure y1 < y2 for consistency
    if y1_original > y2_original:
        y1_original, y2_original = y2_original, y1_original

    return x1_original, y1_original, x2_original, y2_original


def inverse_transform_rotate_90_left(x1, y1, x2, y2, img_width, img_height):
    """
    Inverse transform bounding box coordinates from 90° left rotated image back to original coordinate system.
    Original transform: (x, y) -> (y, img_height - x)
    Inverse transform: (x, y) -> (img_width - y, x)
    """
    # For 90° left rotation inverse: (x, y) -> (img_width - y, x)
    x1_original = img_width - y2
    y1_original = x1
    x2_original = img_width - y1
    y2_original = x2

    # Ensure x1 < x2 and y1 < y2 for consistency
    if x1_original > x2_original:
        x1_original, x2_original = x2_original, x1_original
    if y1_original > y2_original:
        y1_original, y2_original = y2_original, y1_original

    return x1_original, y1_original, x2_original, y2_original


def inverse_transform_rotate_90_right(x1, y1, x2, y2, img_width, img_height):
    """
    Inverse transform bounding box coordinates from 90° right rotated image back to original coordinate system.
    Original transform: (x, y) -> (img_height - y, x)
    Inverse transform: (x, y) -> (y, img_height - x)
    """
    # For 90° right rotation inverse: (x, y) -> (y, img_height - x)
    x1_original = y1
    y1_original = img_height - x2
    x2_original = y2
    y2_original = img_height - x1

    # Ensure x1 < x2 and y1 < y2 for consistency
    if x1_original > x2_original:
        x1_original, x2_original = x2_original, x1_original
    if y1_original > y2_original:
        y1_original, y2_original = y2_original, y1_original

    return x1_original, y1_original, x2_original, y2_original


def run_model_inference(
    model_path, image_dir, filename_to_id, dataset_suffix, conf_threshold=0.001
):
    """Run inference on a single model and return predictions"""
    print(f"Loading model: {model_path}")

    # Check if model file exists
    full_model_path = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(full_model_path):
        print(f"Warning: Model file not found at {full_model_path}")
        return []

    try:
        model = YOLO(full_model_path)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return []

    predictions = []
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Processing {len(image_files)} images...")

    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_dir, img_file)

        # Determine transformation type and original filename
        original_filename = img_file
        transform_type = "none"

        if img_file.startswith("h_mirror_"):
            original_filename = img_file[9:]  # Remove 'h_mirror_' prefix
            transform_type = "horizontal_mirror"
        elif img_file.startswith("v_mirror_"):
            original_filename = img_file[9:]  # Remove 'v_mirror_' prefix
            transform_type = "vertical_mirror"
        elif img_file.startswith("rot90left_"):
            original_filename = img_file[10:]  # Remove 'rot90left_' prefix
            transform_type = "rotate_90_left"
        elif img_file.startswith("rot90right_"):
            original_filename = img_file[11:]  # Remove 'rot90right_' prefix
            transform_type = "rotate_90_right"

        if original_filename not in filename_to_id:
            print(
                f"Warning: Image {original_filename} not found in instances_test.json"
            )
            continue

        img_id = filename_to_id[original_filename]

        try:
            # Run inference
            results = model(img_path, conf=conf_threshold, verbose=False)

            # Get image dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue

            img_height, img_width = img.shape[:2]

            # Process detections
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i]
                        confidence = confidences[i]
                        class_id = (
                            int(class_ids[i]) + 1
                        )  # Convert 0-based to 1-based class ID

                        # Apply inverse transformation based on the augmentation type
                        if transform_type == "horizontal_mirror":
                            x1, y1, x2, y2 = inverse_transform_horizontal_mirror(
                                x1, y1, x2, y2, img_width, img_height
                            )
                        elif transform_type == "vertical_mirror":
                            x1, y1, x2, y2 = inverse_transform_vertical_mirror(
                                x1, y1, x2, y2, img_width, img_height
                            )
                        elif transform_type == "rotate_90_left":
                            x1, y1, x2, y2 = inverse_transform_rotate_90_left(
                                x1, y1, x2, y2, img_width, img_height
                            )
                        elif transform_type == "rotate_90_right":
                            x1, y1, x2, y2 = inverse_transform_rotate_90_right(
                                x1, y1, x2, y2, img_width, img_height
                            )

                        # Normalize coordinates
                        x1_norm, y1_norm, x2_norm, y2_norm = normalize_coordinates(
                            x1, y1, x2, y2, img_width, img_height
                        )

                        predictions.append(
                            [
                                img_id,
                                class_id,
                                confidence,
                                x1_norm,
                                y1_norm,
                                x2_norm,
                                y2_norm,
                            ]
                        )

        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            continue

    print(f"Generated {len(predictions)} predictions")
    return predictions


def save_predictions_to_csv(predictions, output_file):
    """Save predictions to CSV file"""
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(["img_id", "label", "score", "x1", "x2", "y1", "y2"])
        for pred in predictions:
            # Format: img_id,label,score,x1,x2,y1,y2
            writer.writerow(
                [
                    pred[0],  # img_id
                    pred[1],  # label (class)
                    pred[2],  # score (confidence)
                    pred[3],  # x1
                    pred[5],  # x2
                    pred[4],  # y1
                    pred[6],  # y2
                ]
            )


def main():
    """Main function to run inference on all models and datasets"""

    # Load image ID mapping
    instances_json_path = r"Secret1.v19-experiment-1-balance.yolov8\instances_test.json"
    if not os.path.exists(instances_json_path):
        print(f"Error: {instances_json_path} not found!")
        return

    filename_to_id = load_image_id_mapping(instances_json_path)
    print(f"Loaded {len(filename_to_id)} image ID mappings")

    # Create output directory
    output_dir = "model_predictions"
    os.makedirs(output_dir, exist_ok=True)

    total_combinations = len(MODEL_CONFIGS) * len(TEST_DATASETS)
    current_combination = 0

    # Run inference for each model and dataset combination
    for model_config in MODEL_CONFIGS:
        model_name = model_config["name"]
        model_path = model_config["path"]

        for dataset_config in TEST_DATASETS:
            dataset_name = dataset_config["name"]
            dataset_path = dataset_config["path"]
            dataset_suffix = dataset_config["suffix"]

            current_combination += 1
            print(f"\n{'='*60}")
            print(f"Progress: {current_combination}/{total_combinations}")
            print(f"Model: {model_name}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")

            # Check if dataset directory exists
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset directory not found: {dataset_path}")
                continue

            # Run inference
            predictions = run_model_inference(
                model_path, dataset_path, filename_to_id, dataset_suffix
            )

            # Save predictions
            output_filename = f"{model_name}{dataset_suffix}.csv"
            output_path = os.path.join(output_dir, output_filename)
            save_predictions_to_csv(predictions, output_path)

            print(f"Saved predictions to: {output_path}")
            print(f"Total predictions: {len(predictions)}")

    print(f"\n{'='*60}")
    print("All inference completed!")
    print(f"Output files saved in: {output_dir}")
    print(f"Generated {len(MODEL_CONFIGS) * len(TEST_DATASETS)} CSV files")

    # List all generated files
    print("\nGenerated files:")
    for model_config in MODEL_CONFIGS:
        for dataset_config in TEST_DATASETS:
            filename = f"{model_config['name']}{dataset_config['suffix']}.csv"
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
