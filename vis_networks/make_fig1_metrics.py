"""
make_fig1_metrics.py  —  grouped bar chart, all 7 configurations × 4 metrics
Run from:  vis_networks/
Output:    ../paper/figures/fig1_metrics.png
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# ── Publication rcParams ─────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.framealpha": 0.88,
    "axes.linewidth": 0.8, "patch.linewidth": 0.5,
    "grid.linewidth": 0.5, "grid.alpha": 0.35,
    "figure.constrained_layout.use": True,
})
FULL_WIDTH = 6.30  # inches  (≈160 mm, cas-sc text width)

# ── Aggregate results (Table 2 of manuscript) ────────────────────────────────
configs = [
    "VisNet\nGlobal", "RMEP\nGlobal",
    "VisNet\nScratch", "RMEP\nScratch",
    "VisNet\nFine-tune", "RMEP\nFine-tune",
    "VisNet-Sim\nFT",
]
metrics = {
    "$R^2$":         [-0.030,  0.126, -0.581, -0.594,  0.310,  0.396,  0.170],
    "Acc (%)":       [ 28.4,   30.9,   27.8,   28.1,   44.9,   46.3,   37.7],
    "$\\pm$2 Acc (%)":[ 66.4,  69.3,   56.1,   56.9,   75.7,   77.6,   71.6],
    "MAE (mi)":      [  2.047, 1.861,  2.583,  2.588,  1.473,  1.363,  1.720],
}
# Whether higher is better for each metric
higher_better = [True, True, True, False]

# Colorblind-safe palette
C = {"global": "#0072B2", "scratch": "#E69F00",
     "ft": "#009E73", "sim": "#CC79A7"}
bar_colors = [C["global"], C["global"],
              C["scratch"], C["scratch"],
              C["ft"], C["ft"],
              C["sim"]]

ylims  = [(-0.70, 0.52), (0, 62), (38, 92), (0, 4.6)]
yticks = [
    [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4],
    [0, 10, 20, 30, 40, 50, 60],
    [40, 50, 60, 70, 80, 90],
    [0, 1, 2, 3, 4],
]

fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.72))
x = np.arange(len(configs))

for ax, (mname, vals), ylim, yt, hb in zip(
        axes.flatten(), metrics.items(), ylims, yticks, higher_better):

    bars = ax.bar(x, vals, width=0.62, color=bar_colors,
                  edgecolor="white", linewidth=0.5, zorder=3)
    # Bold border on best CNN bar (exclude VEIA-level — all are CNN here)
    best = int(np.argmax(vals) if hb else np.argmin(vals))
    bars[best].set_edgecolor("#111111")
    bars[best].set_linewidth(1.4)

    ax.axhline(0, color="black", linewidth=0.5, zorder=2)
    ax.set_ylabel(mname)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(ylim)
    ax.set_yticks(yt)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

legend_handles = [
    Patch(facecolor=C["global"],  label="Global"),
    Patch(facecolor=C["scratch"], label="Per-site scratch"),
    Patch(facecolor=C["ft"],      label="Per-site fine-tune"),
    Patch(facecolor=C["sim"],     label="VisNet-Similarity FT"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=4, bbox_to_anchor=(0.5, -0.045), frameon=True)

out = os.path.join("..", "paper", "figures", "fig1_metrics.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")