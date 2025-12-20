import cv2
import os

# File paths
img_path = r"test-horizontal-mirror\images\mirror_339703-oryzias-javanicus-2_jpg.rf.6bd583a7804755f493cbd77b3d1c6585.jpg"
label_path = r"test-horizontal-mirror\labels\mirror_339703-oryzias-javanicus-2_jpg.rf.6bd583a7804755f493cbd77b3d1c6585.txt"

# Load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

h, w = img.shape[:2]

# Read label file
with open(label_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) != 5:
        continue
    cls, x_center, y_center, bw, bh = map(float, parts)
    # Convert YOLO format to pixel coordinates
    x_center *= w
    y_center *= h
    bw *= w
    bh *= h
    x1 = int(x_center - bw / 2)
    y1 = int(y_center - bh / 2)
    x2 = int(x_center + bw / 2)
    y2 = int(y_center + bh / 2)
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img, str(int(cls)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

# Show image
cv2.imshow("Labeled Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
