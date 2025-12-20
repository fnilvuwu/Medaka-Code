import os
import cv2
import shutil
from PIL import Image
import numpy as np


def augment_and_save(
    base_dir,
    images_dir,
    labels_dir,
    out_folder,
    image_transform_fn,
    label_transform_fn,
    image_prefix,
    label_prefix,
    summary_name,
    transform_desc,
):
    """
    Apply transformations to images and labels, then save to output folder
    """
    # Create output directories
    out_dir = os.path.join(base_dir, out_folder)
    out_images_dir = os.path.join(out_dir, "images")
    out_labels_dir = os.path.join(out_dir, "labels")

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    processed_count = 0

    # Process all images
    for image_file in os.listdir(images_dir):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file)

            # Load and transform image
            img = cv2.imread(image_path)
            if img is not None:
                transformed_img = image_transform_fn(img)

                # Save transformed image
                out_image_name = image_prefix + image_file
                out_image_path = os.path.join(out_images_dir, out_image_name)
                cv2.imwrite(out_image_path, transformed_img)

                # Transform and save labels if they exist
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        lines = f.readlines()

                    out_label_name = label_prefix + label_file
                    out_label_path = os.path.join(out_labels_dir, out_label_name)

                    with open(out_label_path, "w") as f:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = parts[0]
                                x, y, w, h = map(float, parts[1:5])
                                new_x, new_y, new_w, new_h = label_transform_fn(
                                    x, y, w, h
                                )
                                f.write(
                                    f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n"
                                )

                processed_count += 1

    # Write summary
    summary_path = os.path.join(out_dir, summary_name)
    with open(summary_path, "w") as f:
        f.write(f"Transformation Summary\n")
        f.write(f"=====================\n")
        f.write(f"Processed images: {processed_count}\n")
        f.write(f"Output folder: {out_folder}\n\n")
        f.write("Transformations applied:\n")
        for desc in transform_desc:
            f.write(f"- {desc}\n")

    print(f"Completed {out_folder}: {processed_count} images processed")


def create_all_augmentations():
    """
    Create horizontal mirror, vertical mirror, rotate 90 left, rotate 90 right datasets for YOLO format
    """
    base_dir = (
        r"d:\Penelitian Medaka (Code) (ONGOING)\Secret1.v19-experiment-1-balance.yolov8"
    )
    test_images_dir = os.path.join(base_dir, "test", "images")
    test_labels_dir = os.path.join(base_dir, "test", "labels")

    # Horizontal mirror
    augment_and_save(
        base_dir,
        test_images_dir,
        test_labels_dir,
        out_folder="test-horizontal-mirror",
        image_transform_fn=lambda img: cv2.flip(img, 1),
        label_transform_fn=lambda x, y, w, h: (1.0 - x, y, w, h),
        image_prefix="h_mirror_",
        label_prefix="h_mirror_",
        summary_name="h_mirror_summary.txt",
        transform_desc=[
            "Images: Horizontal flip (cv2.flip with flag=1)",
            "Labels: X-coordinate mirrored (new_x = 1 - original_x)",
            "Y-coordinate, width, height: unchanged",
            "Class IDs: unchanged",
        ],
    )

    # Vertical mirror
    augment_and_save(
        base_dir,
        test_images_dir,
        test_labels_dir,
        out_folder="test-vertical-mirror",
        image_transform_fn=lambda img: cv2.flip(img, 0),
        label_transform_fn=lambda x, y, w, h: (x, 1.0 - y, w, h),
        image_prefix="v_mirror_",
        label_prefix="v_mirror_",
        summary_name="v_mirror_summary.txt",
        transform_desc=[
            "Images: Vertical flip (cv2.flip with flag=0)",
            "Labels: Y-coordinate mirrored (new_y = 1 - original_y)",
            "X-coordinate, width, height: unchanged",
            "Class IDs: unchanged",
        ],
    )

    # Rotate 90 left (counterclockwise)
    def rotate90left_img(img):
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def rotate90left_label(x, y, w, h):
        # (x, y) -> (y, 1-x)
        return (y, 1.0 - x, h, w)

    augment_and_save(
        base_dir,
        test_images_dir,
        test_labels_dir,
        out_folder="test-rotate-90-left",
        image_transform_fn=rotate90left_img,
        label_transform_fn=rotate90left_label,
        image_prefix="rot90left_",
        label_prefix="rot90left_",
        summary_name="rot90left_summary.txt",
        transform_desc=[
            "Images: Rotated 90 degrees left (cv2.ROTATE_90_COUNTERCLOCKWISE)",
            "Labels: (x, y, w, h) -> (y, 1-x, h, w)",
            "Class IDs: unchanged",
        ],
    )

    # Rotate 90 right (clockwise)
    def rotate90right_img(img):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    def rotate90right_label(x, y, w, h):
        # (x, y) -> (1-y, x)
        return (1.0 - y, x, h, w)

    augment_and_save(
        base_dir,
        test_images_dir,
        test_labels_dir,
        out_folder="test-rotate-90-right",
        image_transform_fn=rotate90right_img,
        label_transform_fn=rotate90right_label,
        image_prefix="rot90right_",
        label_prefix="rot90right_",
        summary_name="rot90right_summary.txt",
        transform_desc=[
            "Images: Rotated 90 degrees right (cv2.ROTATE_90_CLOCKWISE)",
            "Labels: (x, y, w, h) -> (1-y, x, h, w)",
            "Class IDs: unchanged",
        ],
    )


def verify_mirror_transformation(
    original_image_name,
):
    """
    Verify the mirror transformation by comparing original and mirrored annotations
    """
    base_dir = (
        r"d:\Penelitian Medaka (Code) (ONGOING)\Secret1.v19-experiment-1-balance.yolov8"
    )

    # Original files
    original_label = os.path.join(
        base_dir, "test", "labels", os.path.splitext(original_image_name)[0] + ".txt"
    )

    # Mirrored files
    mirrored_label = os.path.join(
        base_dir,
        "test-horizontal-mirror",
        "labels",
        f"h_mirror_{os.path.splitext(original_image_name)[0]}.txt",
    )

    print(f"\nVerification for {original_image_name}:")
    print("=" * 50)

    if os.path.exists(original_label):
        print("Original annotations:")
        with open(original_label, "r") as f:
            original_lines = f.readlines()
        for i, line in enumerate(original_lines):
            print(f"  {i+1}: {line.strip()}")
    else:
        print("Original label file not found!")
        return

    if os.path.exists(mirrored_label):
        print("\nMirrored annotations:")
        with open(mirrored_label, "r") as f:
            mirrored_lines = f.readlines()
        for i, line in enumerate(mirrored_lines):
            print(f"  {i+1}: {line.strip()}")

        print("\nTransformation verification:")
        for i, (orig, mirror) in enumerate(zip(original_lines, mirrored_lines)):
            orig_parts = orig.strip().split()
            mirror_parts = mirror.strip().split()

            if len(orig_parts) == 5 and len(mirror_parts) == 5:
                orig_x = float(orig_parts[1])
                mirror_x = float(mirror_parts[1])
                expected_x = 1.0 - orig_x

                print(
                    f"  Box {i+1}: Original X={orig_x:.6f}, Mirrored X={mirror_x:.6f}, Expected={expected_x:.6f}"
                )
                print(f"          Difference: {abs(mirror_x - expected_x):.8f}")
    else:
        print("Mirrored label file not found!")


if __name__ == "__main__":
    # Create all augmentations
    create_all_augmentations()
    # Optionally verify one transformation - provide a sample image name
    # Uncomment and provide actual image name to test:
    # verify_mirror_transformation("sample_image.jpg")
    print("\nScript completed successfully!")
