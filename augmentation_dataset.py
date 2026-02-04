import cv2
import numpy as np
import os
import random

# ----------------------------
# Utility
# ----------------------------
def save(img, name, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    cv2.imwrite(path, img)
    print(f"Saved: {path}")

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
# Scaling (centered)
# ----------------------------
def scale(img, gain=0.5):
    h, w = img.shape[:2]
    scaled = cv2.resize(img, None, fx=gain, fy=gain)

    canvas = np.zeros_like(img)
    sh, sw = scaled.shape[:2]

    # Center offsets
    y_offset = (h - sh) // 2
    x_offset = (w - sw) // 2

    canvas[y_offset:y_offset+sh, x_offset:x_offset+sw] = scaled
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
# MAIN
# ----------------------------
if __name__ == "__main__":
    # Only ONE input image
    img = cv2.imread("image.jpg")

    save(img, "original")

    # 1. HSV (h, s, v combined)
    save(augment_hsv(img), "hsv")

    # 2. Translation
    save(translate(img, 0.1), "translate_0.1")

    # 3. Scaling (centered)
    save(scale(img, 0.5), "scale_0.5_centered")

    # 4. Flip Left–Right
    save(fliplr(img), "fliplr")

    # 5. Mosaic (same image tiled 2×2)
    save(mosaic_single_image(img), "mosaic_single_image")

    # 6. Random Erasing
    save(random_erasing(img, 0.4), "erasing_0.4")
