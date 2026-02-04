import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# CONFIG: 5 models (5-fold)
# --------------------------------------------------
models = {
    "Fold 1": r"Training 13 Models\yolo8n-fold1\exp1\results.csv",
    "Fold 2": r"Training 13 Models\yolo8n-fold2\exp1\results.csv",
    "Fold 3": r"Training 13 Models\yolo8n-fold3\exp1\results.csv",
    "Fold 4": r"Training 13 Models\yolo8n-fold4\exp1\results.csv",
    "Fold 5": r"Training 13 Models\yolo8n-fold5\exp1\results.csv",
}

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# --------------------------------------------------
# FIGURE (bigger canvas to avoid overflow)
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

# Containers to compute the averages
all_train_box = []
all_val_box = []
epochs = None

for i, (label, path) in enumerate(models.items()):
    df = pd.read_csv(path)
    epochs = df["epoch"]  # assume same epochs for all folds
    c = colors[i % len(colors)]

    train_box = df["train/box_loss"]
    val_box = df["val/box_loss"]

    # Store values for averaging
    all_train_box.append(train_box.values)
    all_val_box.append(val_box.values)

    # ---- Training box loss ----
    axes[0].plot(
        epochs,
        train_box,
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
        val_box,
        color=c,
        marker="o",
        markersize=3,
        linewidth=1.2,
        alpha=0.85
    )

# --------------------------------------------------
# COMPUTE AVERAGE (ONE SINGLE LINE PER METRIC)
# --------------------------------------------------
avg_train_box = np.mean(all_train_box, axis=0)
avg_val_box = np.mean(all_val_box, axis=0)

# --------------------------------------------------
# PLOT AVERAGE AS DOTTED LINE
# --------------------------------------------------
axes[0].plot(
    epochs,
    avg_train_box,
    color="black",
    linestyle="--",
    linewidth=2.0,
    label="Average"
)

axes[1].plot(
    epochs,
    avg_val_box,
    color="black",
    linestyle="--",
    linewidth=2.0
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
    ncol=6,                 # 5 folds + Average
    frameon=False,
    bbox_to_anchor=(0.5, 0.02),
    handlelength=2.0,
    handletextpad=0.6,
    columnspacing=1.2,
    markerscale=1.0
)

# --------------------------------------------------
# LAYOUT CONTROL
#   - DO NOT use tight_layout()
#   - Manually reserve space for the legend
# --------------------------------------------------
fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.90,
    bottom=0.22,
    wspace=0.12
)

plt.show()
