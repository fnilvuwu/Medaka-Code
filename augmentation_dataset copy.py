import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------------
# HSV augmentation
# ----------------------------
def augment_hsv(img, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hsv_h * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + hsv_s), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + hsv_v), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ----------------------------
# Translation
# ----------------------------
def translate(img, frac=0.1):
    h, w = img.shape[:2]
    tx = int(frac * w)
    ty = int(frac * h)
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))

# ----------------------------
# Scaling
# ----------------------------
def scale(img, gain=0.5):
    h, w = img.shape[:2]
    scaled = cv2.resize(img, None, fx=gain, fy=gain)
    canvas = np.zeros_like(img)
    sh, sw = scaled.shape[:2]
    canvas[:sh, :sw] = scaled
    return canvas

# ----------------------------
# Flip Left–Right
# ----------------------------
def fliplr(img):
    return cv2.flip(img, 1)

# ----------------------------
# Mosaic (reuse same image 4×)
# ----------------------------
def mosaic_single_image(img, size=256):
    img = cv2.resize(img, (size, size))
    top = np.hstack((img, img))
    bottom = np.hstack((img, img))
    return np.vstack((top, bottom))

# ----------------------------
# Random Erasing
# ----------------------------
def random_erasing(img, erase_ratio=0.4):
    h, w = img.shape[:2]
    eh, ew = int(h * erase_ratio), int(w * erase_ratio)
    x = random.randint(0, w - ew)
    y = random.randint(0, h - eh)
    erased = img.copy()
    erased[y:y+eh, x:x+ew] = np.random.randint(0, 255, (eh, ew, 3))
    return erased

# ----------------------------
# Convert BGR → RGB for plotting
# ----------------------------
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------------------------
# MAIN (aesthetic figure)
# ----------------------------
if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    img = cv2.resize(img, (256, 256))

    images = [
        img,
        augment_hsv(img),
        translate(img, 0.1),
        scale(img, 0.5),
        fliplr(img),
        mosaic_single_image(img, 128),
        random_erasing(img, 0.4)
    ]

    titles = [
        "Original",
        "HSV Augmentation",
        "Translation",
        "Scaling",
        "Flip (Left–Right)",
        "Mosaic (2×2)",
        "Random Erasing"
    ]

    # Create a clean grid layout
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(bgr2rgb(images[i]))
            ax.set_title(titles[i], fontsize=12, fontweight="bold")
        ax.axis("off")

    # Global title
    fig.suptitle("Visualization of Preprocessing and Data Augmentation Steps",
                 fontsize=18, fontweight="bold")

    # Tight layout for paper-quality aesthetics
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("preprocessing_overview.png", dpi=300, bbox_inches="tight")
    plt.show()
