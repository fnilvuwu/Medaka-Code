import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG: Different YOLOv8 sizes
# --------------------------------------------------
models = {
    "YOLOv8n":  r"Training 13 Models\yolo8n\exp1\results.csv",
    "YOLOv9t":  r"Training 13 Models\yolo9t\exp1\results.csv",
    "YOLOv10n": r"Training 13 Models\yolo10n\exp1\results.csv",
    "YOLOv11n": r"Training 13 Models\yolo11n\exp2\results.csv",
}

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# --------------------------------------------------
# FIGURE (bigger canvas to avoid overflow)
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

for i, (label, path) in enumerate(models.items()):
    df = pd.read_csv(path)
    epochs = df["epoch"]
    c = colors[i % len(colors)]

    # ---- Training box loss ----
    axes[0].plot(
        epochs,
        df["train/box_loss"],
        color=c,
        marker="o",
        markersize=3,
        linewidth=1.2,
        alpha=0.85,
        label=label
    )

    # ---- Validation box loss ----
    axes[1].plot(
        epochs,
        df["val/box_loss"],
        color=c,
        marker="o",
        markersize=3,
        linewidth=1.2,
        alpha=0.85
    )

# --------------------------------------------------
# AXIS FORMATTING
# --------------------------------------------------
axes[0].set_title("Training Box Loss")
axes[1].set_title("Validation Box Loss")

for ax in axes:
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

# --------------------------------------------------
# LEGEND (inside figure, bottom center)
# --------------------------------------------------
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,                      # 4 models → cleaner layout
    frameon=False,
    bbox_to_anchor=(0.5, 0.02),  # keep legend inside the figure
    handlelength=2.0,
    handletextpad=0.6,
    columnspacing=1.2,
    markerscale=1.0
)

# --------------------------------------------------
# LAYOUT CONTROL
#   - NO tight_layout()
#   - Reserve space manually for legend
# --------------------------------------------------
fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.90,
    bottom=0.22,   # space for legend
    wspace=0.12
)

plt.show()
