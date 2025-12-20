import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# Case NMW wins
# boxes = np.array([
#     [25, 25, 135, 135],   # Model 1 (dominant but shifted) - HIGH score
#     [40, 40, 140, 140],   # Model 2 (over-shifted)
#     [30, 35, 130, 135],   # Model 3 (close to GT)
#     [35, 30, 135, 130],   # Model 4 (close to GT)
#     [50, 50, 150, 150],   # Model 5 (outlier, low quality)
# ])

# scores = np.array([0.95, 0.70, 0.88, 0.86, 0.55])
# gt_box = np.array([32, 32, 132, 132])

# Case WBF wins
boxes = np.array([
    [30, 34, 130, 134],   # Model 1 (close to GT)
    [34, 30, 134, 130],   # Model 2 (close to GT)
    [32, 32, 132, 132],   # Model 3 (almost perfect)
    [36, 36, 136, 136],   # Model 4 (slightly large)
    [50, 20, 150, 120],   # Model 5 (confident but biased)
])

scores = np.array([
    0.85,  # Model 1
    0.86,  # Model 2
    0.88,  # Model 3 (best consensus)
    0.83,  # Model 4
    0.90,  # Model 5 (highest score but WRONG)
])
gt_box = np.array([32, 32, 132, 132])



def draw_boxes(boxes, scores, title, colors=None, gt_box=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.invert_yaxis()

    # Draw predicted boxes
    for i, box in enumerate(boxes):
        color = colors[i] if colors else "blue"
        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            box[0],
            box[1] - 5,
            f"{scores[i]:.2f}",
            color=color,
            fontsize=9,
            weight="bold",
        )

    # Draw ground truth (red)
    if gt_box is not None:
        gt_rect = Rectangle(
            (gt_box[0], gt_box[1]),
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
            fill=False,
            edgecolor="red",
            linewidth=3,
            linestyle="--",
        )
        ax.add_patch(gt_rect)
        ax.text(
            gt_box[0],
            gt_box[1] - 5,
            "GT",
            color="red",
            fontsize=10,
            weight="bold",
        )

    ax.set_title(title)
    ax.axis("off")
    plt.show()


def nms(boxes, scores, iou_thresh=0.5):
    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        idxs = idxs[1:]
        idxs = np.array([i for i in idxs if iou(boxes[current], boxes[i]) < iou_thresh])

    return boxes[keep], scores[keep]


def soft_nms(boxes, scores, iou_thresh=0.5, sigma=0.5):
    boxes = boxes.copy()
    scores = scores.copy()

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ov = iou(boxes[i], boxes[j])
            if ov > iou_thresh:
                scores[j] *= np.exp(-(ov**2) / sigma)

    keep = scores > 0.05
    return boxes[keep], scores[keep]


def nmw(boxes, scores, iou_thresh=0.5):
    idx = np.argmax(scores)
    ref_box = boxes[idx]

    weights = []
    selected = []

    for i, box in enumerate(boxes):
        ov = iou(ref_box, box)
        if ov >= iou_thresh:
            w = scores[i] * ov
            weights.append(w)
            selected.append(box)

    weights = np.array(weights)
    selected = np.array(selected)

    fused = np.sum(selected * weights[:, None], axis=0) / weights.sum()
    fused_score = scores[idx]

    return np.array([fused]), np.array([fused_score])


def wbf(boxes, scores, iou_thresh=0.5):
    weights = scores
    fused = np.sum(boxes * weights[:, None], axis=0) / weights.sum()
    fused_score = scores.mean()
    return np.array([fused]), np.array([fused_score])


def run_all_methods(boxes, scores, gt_box, title_prefix=""):
    draw_boxes(boxes, scores, f"{title_prefix}Original", gt_box=gt_box)

    b_nms, s_nms = nms(boxes, scores)
    draw_boxes(b_nms, s_nms, f"{title_prefix}NMS", gt_box=gt_box)

    b_soft, s_soft = soft_nms(boxes, scores)
    draw_boxes(b_soft, s_soft, f"{title_prefix}Soft-NMS", gt_box=gt_box)

    b_nmw, s_nmw = nmw(boxes, scores)
    draw_boxes(b_nmw, s_nmw, f"{title_prefix}NMW", gt_box=gt_box)

    b_wbf, s_wbf = wbf(boxes, scores)
    draw_boxes(b_wbf, s_wbf, f"{title_prefix}WBF", gt_box=gt_box)


boxes_case1 = np.array([
    [30, 30, 130, 130],   # dominant & correct
    [45, 30, 145, 130],   # noisy
    [28, 45, 128, 145],   # noisy
    [60, 20, 160, 120],   # outlier
    [20, 60, 120, 160],   # outlier
])

scores_case1 = np.array([0.95, 0.75, 0.72, 0.60, 0.58])
gt_case1 = np.array([32, 32, 132, 132])

run_all_methods(boxes_case1, scores_case1, gt_case1, title_prefix="Case 1: ")


boxes_case2 = np.array([
    [30, 34, 130, 134],
    [34, 30, 134, 130],
    [32, 32, 132, 132],
    [36, 36, 136, 136],
    [50, 20, 150, 120],  # biased but confident
])

scores_case2 = np.array([0.85, 0.86, 0.88, 0.83, 0.90])
gt_case2 = np.array([32, 32, 132, 132])

run_all_methods(boxes_case2, scores_case2, gt_case2, title_prefix="Case 2: ")
