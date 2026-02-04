import pandas as pd
import matplotlib.pyplot as plt

models = {
    "YOLOv8n": r"Training 13 Models\yolo8n\exp1\results.csv",
    "YOLOv8s": r"Training 13 Models\yolo8s\exp1\results.csv",
    "YOLOv8m": r"Training 13 Models\yolo8m\exp1\results.csv",
    "YOLOv8l": r"Training 13 Models\yolo8l\exp1\results.csv",
    "YOLOv8x": r"Training 13 Models\yolo8x\exp1\results.csv",
}

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Bigger canvas to avoid overflow
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

for i, (label, path) in enumerate(models.items()):
    df = pd.read_csv(path)
    epochs = df["epoch"]
    c = colors[i % len(colors)]

    axes[0].plot(
        epochs, df["train/box_loss"],
        color=c, marker="o", markersize=3,
        linewidth=1.2, alpha=0.85, label=label
    )

    axes[1].plot(
        epochs, df["val/box_loss"],
        color=c, marker="o", markersize=3,
        linewidth=1.2, alpha=0.85
    )

# Titles and labels
axes[0].set_title("Training Box Loss")
axes[1].set_title("Validation Box Loss")

for ax in axes:
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

# Legend outside, centered at bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.02)
)

# Manual layout (NO tight_layout)
fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.90,
    bottom=0.22,   # reserve space for legend
    wspace=0.12
)

plt.show()
