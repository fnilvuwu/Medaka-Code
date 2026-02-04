import matplotlib.pyplot as plt
import numpy as np

# Fusion sources (simplified)
fusion_sources = [
    "Different Sizes\nWBF",
    "Different Versions\nWBF",
    "5-Fold (Proposed)\nWBF"
]

# Metrics
precision = [0.891, 0.920, 0.915]
recall    = [0.927, 0.935, 0.959]
f1_score  = [0.908, 0.927, 0.937]

x = np.arange(len(fusion_sources))
width = 0.25

plt.figure(figsize=(6, 4))

# Bars
bars_p  = plt.bar(x - width, precision, width, label="Precision")
bars_r  = plt.bar(x,         recall,    width, label="Recall")
bars_f1 = plt.bar(x + width, f1_score,  width, label="F1-score")

# Add value labels (with headroom and no clipping)
for bars in [bars_p, bars_r, bars_f1]:
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False
        )

plt.xticks(x, fusion_sources)
plt.ylabel("Score")

# Y-axis from 0 to 1 with small headroom
plt.ylim(0, 1.05)

# Legend at bottom right
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
