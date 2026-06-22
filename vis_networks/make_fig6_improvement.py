"""
make_fig6_improvement.py  —  4-panel improvement vs sample count (VisNet FT vs global)
Run from:  vis_networks\
Output:    ..\paper\figures\fig6_improvement.png
"""
import csv, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ── Publication rcParams ─────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.framealpha": 0.88,
    "axes.linewidth": 0.8, "lines.linewidth": 1.1,
    "patch.linewidth": 0.5, "grid.linewidth": 0.5, "grid.alpha": 0.35,
    "figure.constrained_layout.use": True,
})
FULL_WIDTH = 6.30

# ── Paths — set GLOBAL_CSV once you confirm the filename ────────────────────
GLOBAL_CSV   = r"VisNetSiteEvaluationResults\site_specific_validation_epoch0.csv"
FINETUNE_CSV = r"VisNetFineTuneAndPredictionsResults\persite_results.csv"
MIN_N = 20

# ── Load ─────────────────────────────────────────────────────────────────────
def load_csv(path):
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n = int(row.get("n_total", row.get("n_samples", 0)))
            if n < MIN_N:
                continue
            data[row["site_id"]] = {
                "n_total":  n,
                "n_train":  int(row["n_train"]) if "n_train" in row else 0,
                "MAE":  float(row["MAE_visibility"]),
                "RMSE": float(row["RMSE_visibility"]),
            }
    return data

glb = load_csv(GLOBAL_CSV)
ft  = load_csv(FINETUNE_CSV)
common = sorted(set(glb) & set(ft))
print(f"Sites in common: {len(common)}")

n_train   = np.array([ft[s]["n_train"]  for s in common])
glb_mae   = np.array([glb[s]["MAE"]     for s in common])
glb_rms   = np.array([glb[s]["RMSE"]    for s in common])
ft_mae    = np.array([ft[s]["MAE"]      for s in common])
ft_rms    = np.array([ft[s]["RMSE"]     for s in common])
delta_mae = ft_mae - glb_mae   # negative = improvement
delta_rms = ft_rms - glb_rms

# Sample-count buckets for boxplot
edges  = [0, 50, 100, 200, 400, 9999]
blabels = ["<50", "50–100", "100–200", "200–400", ">400"]
bucket_drms = [
    delta_rms[(n_train > lo) & (n_train <= hi)]
    for lo, hi in zip(edges[:-1], edges[1:])
]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.85))
ax_dmae, ax_drms, ax_scatter, ax_box = axes.flatten()

norm = Normalize(vmin=n_train.min(), vmax=n_train.max())
cmap = plt.get_cmap("viridis")

def scatter_with_trend(ax, x, y, color, xlabel, ylabel):
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=2)
    ax.scatter(x, y, s=12, alpha=0.65, color=color, edgecolors="none", zorder=3)
    mask = x > 0
    if mask.sum() > 5:
        z = np.polyfit(np.log(x[mask] + 1), y[mask], 1)
        xfit = np.linspace(x[mask].min(), x[mask].max(), 300)
        ax.plot(xfit, np.polyval(z, np.log(xfit + 1)),
                color="#333333", linewidth=1.0, zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

scatter_with_trend(ax_dmae, n_train, delta_mae,
                   "#0072B2",
                   "Training samples per site",
                   "$\\Delta$MAE (fine-tune $-$ global, mi)")

scatter_with_trend(ax_drms, n_train, delta_rms,
                   "#E69F00",
                   "Training samples per site",
                   "$\\Delta$RMSE (fine-tune $-$ global, mi)")

# Bottom-left: global vs fine-tune MAE scatter, colored by n_train
sc = ax_scatter.scatter(glb_mae, ft_mae, c=n_train, cmap=cmap,
                        s=12, alpha=0.75, edgecolors="none", zorder=3)
lo = min(glb_mae.min(), ft_mae.min()) - 0.1
hi = max(glb_mae.max(), ft_mae.max()) + 0.1
ax_scatter.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, zorder=2,
                label="No change")
ax_scatter.set_xlabel("Global VisNet MAE (mi)")
ax_scatter.set_ylabel("Fine-tuned VisNet MAE (mi)")
ax_scatter.yaxis.grid(True, zorder=0)
ax_scatter.set_axisbelow(True)
ax_scatter.legend(fontsize=7.5, frameon=True)
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                  ax=ax_scatter, fraction=0.046, pad=0.04)
cb.set_label("Training samples", fontsize=7.5)
cb.ax.tick_params(labelsize=7)

# Bottom-right: ΔRMSE boxplot by bucket
nonempty = [(l, d) for l, d in zip(blabels, bucket_drms) if len(d) > 0]
ax_box.boxplot(
    [d for _, d in nonempty],
    labels=[l for l, _ in nonempty],
    patch_artist=True, widths=0.5,
    boxprops=dict(facecolor="#98df8a", linewidth=0.8),
    medianprops=dict(color="#333333", linewidth=1.3),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
    flierprops=dict(marker="o", markersize=3, alpha=0.45,
                    markeredgewidth=0.4),
)
ax_box.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=2)
ax_box.set_xlabel("Training samples (bucket)")
ax_box.set_ylabel("$\\Delta$RMSE (fine-tune $-$ global, mi)")
ax_box.yaxis.grid(True, zorder=0)
ax_box.set_axisbelow(True)

out = os.path.join("..", "paper", "figures", "fig6_improvement.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")