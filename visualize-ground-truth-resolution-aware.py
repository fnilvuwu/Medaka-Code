import os
import cv2

def visualize_yolo_ground_truth(
    images_dir="images",
    labels_dir="labels",
    output_dir="output_images_gt"
):
    """
    Visualize YOLO ground-truth annotations with resolution-aware
    bounding box thickness and font size so that appearance is
    consistent across different image sizes.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Class mapping (YOLO class IDs)
    CLASS_NAMES = {
        0: "o.celebensis",
        1: "o.javanicus",
    }

    # BGR colors
    CLASS_COLORS = {
        0: (0, 255, 0),   # Green
        1: (255, 0, 0),   # Blue
    }

    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, f"{img_id}.txt")

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        # -------------------------------
        # Resolution-aware scaling
        # -------------------------------
        # Reference: ~1000px image
        scale = min(h, w) / 1000.0

        box_thickness = max(1, int(2 * scale))
        font_scale = max(0.4, 0.5 * scale)
        font_thickness = max(1, int(2 * scale))

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_c, y_c, bw, bh = map(float, parts[1:])

                # YOLO normalized → pixel coordinates
                x1 = int((x_c - bw / 2) * w)
                y1 = int((y_c - bh / 2) * h)
                x2 = int((x_c + bw / 2) * w)
                y2 = int((y_c + bh / 2) * h)

                # Clip to image boundaries
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                class_name = CLASS_NAMES.get(class_id, str(class_id))
                color = CLASS_COLORS.get(class_id, (0, 255, 255))

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

                # Prepare label
                text = class_name
                (tw, th), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_thickness
                )

                # Background rectangle for text
                cv2.rectangle(
                    image,
                    (x1, y1 - th - 8),
                    (x1 + tw + 6, y1),
                    color,
                    -1
                )

                # Put text
                cv2.putText(
                    image,
                    text,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )

        out_path = os.path.join(output_dir, img_file)
        cv2.imwrite(out_path, image)

    print(f"Ground-truth visualization saved to: {output_dir}")


# Example usage
visualize_yolo_ground_truth(
    images_dir=r"Medaka-Object-Detection-Dataset.v23-dataset-balance-raw.yolov8\test\images",
    labels_dir=r"Medaka-Object-Detection-Dataset.v23-dataset-balance-raw.yolov8\test\labels",
    output_dir=r"Medaka-Object-Detection-Dataset.v23-dataset-balance-raw.yolov8\test-ground-truth"
)
