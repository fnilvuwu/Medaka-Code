#!/usr/bin/env python3
"""
YOLOv8 to COCO Format Converter
Converts YOLOv8 test dataset to COCO format JSON file.

Author: GitHub Copilot
Date: 2025-10-03
"""

import os
import json
import yaml
from PIL import Image
from datetime import datetime
import argparse


def load_yaml_config(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLOv8 normalized bbox to COCO format.

    Args:
        yolo_bbox: [x_center, y_center, width, height] (all normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        [x_min, y_min, width, height] in absolute pixels (COCO format)
    """
    x_center, y_center, width, height = yolo_bbox

    # Convert normalized coordinates to absolute pixels
    abs_x_center = x_center * img_width
    abs_y_center = y_center * img_height
    abs_width = width * img_width
    abs_height = height * img_height

    # Convert center coordinates to top-left corner (COCO format)
    x_min = abs_x_center - (abs_width / 2)
    y_min = abs_y_center - (abs_height / 2)

    return [x_min, y_min, abs_width, abs_height]


def convert_yolo_to_coco(dataset_path, output_path="instances_test.json"):
    """
    Convert YOLOv8 test dataset to COCO format.

    Args:
        dataset_path: Path to the YOLOv8 dataset root directory
        output_path: Output path for the COCO JSON file
    """

    # Load dataset configuration
    yaml_path = os.path.join(dataset_path, "data.yaml")
    config = load_yaml_config(yaml_path)

    # Get test images and labels paths
    test_images_path = os.path.join(dataset_path, "test", "images")
    test_labels_path = os.path.join(dataset_path, "test", "labels")

    if not os.path.exists(test_images_path):
        raise FileNotFoundError(f"Test images directory not found: {test_images_path}")

    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"Test labels directory not found: {test_labels_path}")

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Medaka Fish Dataset - Test Split (Converted from YOLOv8)",
            "url": config.get("roboflow", {}).get("url", ""),
            "version": "1.0",
            "year": 2025,
            "contributor": "Medaka Research Team",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [
            {
                "id": 1,
                "name": config.get("roboflow", {}).get("license", "CC BY 4.0"),
                "url": "",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Create categories from YAML config
    class_names = config.get("names", [])
    for i, class_name in enumerate(class_names):
        coco_data["categories"].append(
            {
                "id": i + 1,  # COCO category IDs start from 1
                "name": class_name,
                "supercategory": "fish",
            }
        )

    # Get all image files
    image_files = [
        f
        for f in os.listdir(test_images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]

    print(f"Found {len(image_files)} images in test set")

    image_id = 1
    annotation_id = 1

    for image_file in image_files:
        image_path = os.path.join(test_images_path, image_file)

        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(test_labels_path, label_file)

        # Load image to get dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue

        # Add image info to COCO
        coco_data["images"].append(
            {
                "id": image_id,
                "license": 1,
                "file_name": image_file,
                "height": img_height,
                "width": img_width,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "coco_url": "",
                "flickr_url": "",
            }
        )

        # Process annotations if label file exists
        if os.path.exists(label_path):
            try:
                with open(label_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse YOLO format: class_id x_center y_center width height
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Invalid annotation format in {label_file}: {line}")
                        continue

                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError as e:
                        print(f"Error parsing annotation in {label_file}: {line} - {e}")
                        continue

                    # Convert YOLO bbox to COCO bbox
                    coco_bbox = yolo_to_coco_bbox(
                        [x_center, y_center, width, height], img_width, img_height
                    )

                    # Calculate area
                    area = coco_bbox[2] * coco_bbox[3]  # width * height

                    # Add annotation to COCO
                    coco_data["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id
                            + 1,  # COCO category IDs start from 1
                            "bbox": coco_bbox,
                            "area": area,
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )

                    annotation_id += 1

            except Exception as e:
                print(f"Error processing label file {label_file}: {e}")
        else:
            print(f"Warning: No label file found for {image_file}")

        image_id += 1

    # Save COCO JSON file
    output_file = os.path.join(dataset_path, output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print(f"\nConversion completed!")
    print(f"Output saved to: {output_file}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {len(coco_data['categories'])}")

    # Print category mapping
    print("\nCategory mapping:")
    for category in coco_data["categories"]:
        print(f"  ID {category['id']}: {category['name']}")

    return output_file


def main():
    """Main function to run the converter."""
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 test dataset to COCO format"
    )
    parser.add_argument(
        "--dataset_path",
        "-d",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path to YOLOv8 dataset root directory (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="instances_test.json",
        help="Output filename for COCO JSON file (default: instances_test.json)",
    )

    args = parser.parse_args()

    try:
        output_file = convert_yolo_to_coco(args.dataset_path, args.output)
        print(f"\n✅ Successfully converted YOLOv8 test dataset to COCO format!")
        print(f"📁 Output file: {output_file}")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
