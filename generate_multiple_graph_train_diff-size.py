import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG: Different YOLOv8 sizes
# --------------------------------------------------
models = {
    "YOLOv8n": r"Training 13 Models\yolo8n\exp1\results.csv",
    "YOLOv8s": r"Training 13 Models\yolo8s\exp1\results.csv",
    "YOLOv8m": r"Training 13 Models\yolo8m\exp1\results.csv",
    "YOLOv8l": r"Training 13 Models\yolo8l\exp1\results.csv",
    "YOLOv8x": r"Training 13 Models\yolo8x\exp1\results.csv",
}

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# --------------------------------------------------
# FIGURE
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0))

for i, (label, path) in enumerate(models.items()):
    df = pd.read_csv(path)
    epochs = df["epoch"]
    c = colors[i % len(colors)]

    axes[0].plot(
        epochs,
        df["metrics/mAP50(B)"],
        color=c,
        marker="o",
        markersize=4,
        linewidth=1.2,
        alpha=0.85,
        label=label
    )

    axes[1].plot(
        epochs,
        df["metrics/mAP50-95(B)"],
        color=c,
        marker="o",
        markersize=4,
        linewidth=1.2,
        alpha=0.85
    )

# --------------------------------------------------
# AXIS FORMATTING
# --------------------------------------------------
axes[0].set_title("metrics/mAP50(B)")
axes[1].set_title("metrics/mAP50-95(B)")

for ax in axes:
    ax.set_xlabel("Epoch")

# --------------------------------------------------
# LEGEND (centered, bar-chart style)
# --------------------------------------------------
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
    handlelength=2.0,
    handletextpad=0.6,
    columnspacing=1.2,
    markerscale=1.0
)

# --------------------------------------------------
# LAYOUT CONTROL (CRITICAL PART)
# --------------------------------------------------
plt.tight_layout()
plt.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.90,
    bottom=0.22,
    wspace=0.12
)

plt.show()
