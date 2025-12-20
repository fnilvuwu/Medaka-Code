from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import glob
import os

# Load trained model
model = YOLO(r"Ensemble Models\yolo_ensemble_different_version\yolov8n.pt")

# Path to test image
image_path = r"test-samples\javanicus-sample2.jpeg"

# Run prediction
results = model.predict(image_path, conf=0.25)

# Get the first result (assumes single image)
result = results[0]

# 1. Save image with bounding boxes
output_folder = "prediction_results"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "detection_result.jpg")

# Save the plotted image with bounding boxes
result.save(filename=output_path)
print(f"Detection image saved to: {output_path}")

# 2. Save results to a text file (optional)
txt_path = os.path.join(output_folder, "detection_data.txt")
with open(txt_path, "w") as f:
    # Write header
    f.write("Class\tConfidence\tBox Coordinates (x1, y1, x2, y2)\n")

    # Extract boxes, confidence scores, and class ids
    boxes = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()

    # Write detection results
    for i in range(len(boxes)):
        box = boxes[i]
        f.write(
            f"{int(cls[i])}\t{conf[i]:.4f}\t{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}\n"
        )

print(f"Detection data saved to: {txt_path}")

# 3. Print detection summary
print(f"Detected {len(result.boxes)} objects")

# 4. Display the results (optional - uncomment to use)
# plt.figure(figsize=(10, 10))
# plt.imshow(result.plot()[:, :, ::-1])  # Convert BGR to RGB
# plt.axis('off')
# plt.show()
