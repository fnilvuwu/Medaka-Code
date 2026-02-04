import matplotlib.pyplot as plt
import numpy as np

# Data
fusion_sources = [
    "Different Sizes\nWBF",
    "Different Versions\nWBF",
    "5-Fold (Proposed)\nWBF"
]
map_5095 = [0.790, 0.791, 0.805]
ar_100 = [0.834, 0.844, 0.849]

x = np.arange(len(fusion_sources))
width = 0.35

# Figure
plt.figure(figsize=(6, 4))

# Bars
bars_map = plt.bar(x - width / 2, map_5095, width, label="mAP@0.5:0.95")
bars_ar  = plt.bar(x + width / 2, ar_100,  width, label="AR@100")

# Value labels (with headroom and no clipping)
for bars in [bars_map, bars_ar]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False
        )

# Axes & labels
plt.xticks(x, fusion_sources)
plt.ylabel("Score")

# Y-axis from 0 to 1 with small headroom for labels
plt.ylim(0, 1.05)

# Grid & legend (bottom right)
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
