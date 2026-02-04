import matplotlib.pyplot as plt
import numpy as np

# ===================== Data =====================
fusion_methods = [
    "Different Sizes\nWBF",
    "Different Versions\nWBF",
    "5-Fold (Proposed)\nWBF"
]

# Added mAP@50
metrics = [
    "Precision",
    "Recall",
    "F1",
    "mAP@0.50",
    "mAP\n@0.50:0.95",
    "mAR\n@0.50:0.95"
]

# From your table
precision = [0.891, 0.920, 0.951]
recall    = [0.927, 0.935, 0.951]
f1_score  = [0.908, 0.927, 0.951]
map_50    = [0.972, 0.981, 0.993]
map_5095  = [0.790, 0.791, 0.805]
ar_100    = [0.836, 0.843, 0.849]

data = {
    fusion_methods[0]: [precision[0], recall[0], f1_score[0], map_50[0], map_5095[0], ar_100[0]],
    fusion_methods[1]: [precision[1], recall[1], f1_score[1], map_50[1], map_5095[1], ar_100[1]],
    fusion_methods[2]: [precision[2], recall[2], f1_score[2], map_50[2], map_5095[2], ar_100[2]],
}

# ===================== Radar Geometry =====================
N = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Orientation: top is first metric, clockwise direction
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Remove circular frame and default grid
ax.spines["polar"].set_visible(False)
ax.grid(False)

# ===================== Polygon Grid =====================
grid_levels = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

# Draw polygon rings
for r in grid_levels:
    ax.plot(
        angles, [r] * len(angles),
        linestyle="--", linewidth=0.8,
        color="grey", alpha=0.6
    )

# Draw radial spokes
for a in angles[:-1]:
    ax.plot(
        [a, a], [0.75, 1.00],
        linestyle="--", linewidth=0.8,
        color="grey", alpha=0.6
    )

# ===================== Plot Data =====================
for method in fusion_methods:
    values = data[method] + data[method][:1]
    ax.plot(angles, values, linewidth=2, marker="o", markersize=4, label=method)
    ax.fill(angles, values, alpha=0.10)

# ===================== Axis Labels =====================
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=14,  fontweight='bold')
ax.tick_params(pad=16)

# ===================== Radial Labels =====================
ax.set_ylim(0.75, 1.0)

# Remove default radial tick labels
ax.set_yticks(grid_levels)
ax.set_yticklabels([])

# Place labels on the top vertical axis
label_angle = 0.0
for r in grid_levels:
    ax.text(
        label_angle,
        r,
        f"{r:.2f}",
        ha="center",
        va="bottom",
        fontsize=14,
        color="black",
    )

# ===================== Legend =====================
plt.legend(
    loc="upper right",
    bbox_to_anchor=(1.25, 1.25),
    frameon=True,
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "radar_fusion_sources.png",
    dpi=300,
    bbox_inches="tight",
    transparent=False
)

plt.show()
