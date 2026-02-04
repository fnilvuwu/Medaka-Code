import matplotlib.pyplot as plt
import numpy as np

fusion_methods = ["NMS", "Soft-NMS", "NMW", "WBF"]
map_5095 = [0.785, 0.783, 0.799, 0.805]
ar_100 = [0.839, 0.842, 0.849, 0.849]

x = np.arange(len(fusion_methods))
width = 0.35

plt.figure(figsize=(6, 4))

# Bars
bars_map = plt.bar(x - width/2, map_5095, width, label="mAP@0.5:0.95")
bars_ar  = plt.bar(x + width/2, ar_100,  width, label="AR@100")

# Value labels on bars (with headroom and no clipping)
for bars in [bars_map, bars_ar]:
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
