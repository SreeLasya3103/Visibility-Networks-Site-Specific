"""
make_fig2_r2dist.py  —  stacked bar of per-site R² tiers, 3 configurations
Run from:  vis_networks/
Output:    ../paper/figures/fig2_r2dist.png
"""
import csv, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.framealpha": 0.88,
    "axes.linewidth": 0.8, "patch.linewidth": 0.5,
    "grid.linewidth": 0.5, "grid.alpha": 0.35,
    "figure.constrained_layout.use": True,
})
FULL_WIDTH = 6.30

CSVS = {
    "VisNet FT":     "VisNetFineTuneAndPredictionsResults/persite_results.csv",
    "RMEP FT":       "RMEPFineTuneAndPredictionsResults/persite_results.csv",
    "VisNet-Sim FT": "VisNetSimilarityPersiteFineTuneResults/persite_results.csv",
}
MIN_N = 20

# Tiers ordered top→bottom in stacked bar (plotted bottom→top)
TIERS = [
    ("Poor ($R^2<0$)",           "#d62728"),   # red       — plotted first (bottom)
    ("Fair ($0\\leq R^2<0.5$)",  "#f5c542"),   # yellow
    ("Good ($0.5\\leq R^2<0.89$)","#98df8a"),  # light green
    ("VEIA-level ($R^2\\geq0.89$)","#2ca02c"), # dark green — plotted last (top)
]

def classify(r2):
    if r2 >= 0.89: return 3
    if r2 >= 0.50: return 2
    if r2 >= 0.00: return 1
    return 0

def tier_counts(path):
    counts = [0, 0, 0, 0]
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["n_total"]) < MIN_N:
                continue
            counts[classify(float(row["R2"]))] += 1
    return counts

fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.58, FULL_WIDTH * 0.46))

labels = list(CSVS.keys())
x = np.arange(len(labels))
all_counts = [tier_counts(CSVS[lbl]) for lbl in labels]  # shape (3, 4)
bottoms = np.zeros(len(labels))

for tier_idx, (tier_label, color) in enumerate(TIERS):
    vals = np.array([all_counts[li][tier_idx] for li in range(len(labels))],
                    dtype=float)
    ax.bar(x, vals, width=0.50, bottom=bottoms, color=color,
           edgecolor="white", linewidth=0.5, label=tier_label, zorder=3)
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v >= 5:
            ax.text(xi, b + v / 2, str(int(v)), ha="center", va="center",
                    fontsize=7.5, color="black", fontweight="bold")
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Number of sites")
ax.set_ylim(0, max(bottoms) * 1.04)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

# Legend placed above the plot so it never overlaps the bars
handles, lbls = ax.get_legend_handles_labels()
ax.legend(handles[::-1], lbls[::-1],
          loc="lower center", bbox_to_anchor=(0.5, 1.02),
          ncol=2, frameon=True, fontsize=7.5,
          columnspacing=1.2, handlelength=1.4)

out = os.path.join("..", "paper", "figures", "fig2_r2dist.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")