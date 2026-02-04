import matplotlib.pyplot as plt
import numpy as np

fusion_methods = ["NMS", "Soft-NMS", "NMW", "WBF"]

# Precision, Recall, F1 (5-Fold Ensemble)
precision = [0.766, 0.779, 0.779, 0.915]
recall    = [0.959, 0.976, 0.976, 0.959]
f1_score  = [0.852, 0.866, 0.866, 0.937]

x = np.arange(len(fusion_methods))
width = 0.25  # smaller width since we now have 3 bars

plt.figure(figsize=(6, 4))

# Bars
bars_p  = plt.bar(x - width, precision, width, label="Precision")
bars_r  = plt.bar(x,         recall,    width, label="Recall")
bars_f1 = plt.bar(x + width, f1_score,  width, label="F1-score")

# Value labels on bars
for bars in [bars_p, bars_r, bars_f1]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False
        )

plt.xticks(x, fusion_methods)
plt.ylabel("Score")

# Set y-axis from 0 to 1 with headroom for labels
plt.ylim(0, 1.05)

# Legend at bottom right
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
