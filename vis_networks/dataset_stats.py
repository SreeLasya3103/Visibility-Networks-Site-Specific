"""
dataset_stats.py

Computes dataset statistics for the JASTP manuscript Table 1.

Two modes:
  1. DIRECT (preferred): scans image files under DSET_ROOT
  2. FALLBACK: derives stats from predictions CSVs already in this directory

Usage (from vis_networks/):
    python dataset_stats.py
    python dataset_stats.py --csv   # force fallback CSV mode
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- Dataset root candidates (first one that exists wins) ------------------
DSET_CANDIDATES = [
    r"D:\Research - Lasya\NewWebcams",
    r"D:/Research - Lasya/NewWebcams",
    "/data/NewWebcams",
    os.path.join(os.path.dirname(BASE), "NewWebcams"),
]

# ---- Predictions CSVs to aggregate when dataset root is unavailable --------
PRED_CSVS = [
    "VisNetFineTuneAndPredictionsResults/predictions.csv",
    "RMEPFineTuneAndPredictionsResults/predictions.csv",
    "VisNetSimilarityPersiteFineTuneResults/predictions.csv",
    # Linux proxy names (fallback)
    "finetune_results_with_prediction/predictions.csv",
    "finetune_rmep_results_with_prediction/predictions.csv",
    "persite_similarity_finetune/predictions.csv",
]

# ---- Visibility class definitions (matches cls_10full) ---------------------
CLASS_GROUPS = [
    {0.0, 1.0, 1.25, 1.5, 1.75},
    {2.0, 2.25, 2.5},
    {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0},
]
CLASS_NAMES = ["≤1 mi", "2 mi", "3 mi", "4 mi", "5 mi",
               "6 mi", "7 mi", "8 mi", "9 mi", "≥10 mi"]


def parse_vis_from_filename(fname):
    """Extract float visibility value from SITE{id}_ORNT{deg}_VIS{val}mi.png."""
    try:
        base = os.path.basename(fname)
        seg = base.split('_')[2].split('.')[0]   # e.g. VIS10+mi or VIS4mi
        val_str = seg.split('S')[1].split('m')[0].replace('-', '.')
        return 10.0 if val_str == '10+' else min(float(val_str), 10.0)
    except Exception:
        return None


def parse_site_from_filename(fname):
    """Extract site ID from SITE{id}_ORNT{deg}_VIS{val}mi.png."""
    try:
        return os.path.basename(fname).split('_')[0]
    except Exception:
        return None


def vis_to_class(vis):
    for i, group in enumerate(CLASS_GROUPS):
        if vis in group:
            return i
    return None


def stats_from_images(dset_root):
    print(f"Scanning images under: {dset_root}")
    patterns = [
        os.path.join(dset_root, "**", "*.png"),
        os.path.join(dset_root, "**", "*.jpg"),
        os.path.join(dset_root, "**", "*.jpeg"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    if not files:
        print("  No images found.")
        return None

    rows = []
    for f in files:
        site = parse_site_from_filename(f)
        vis  = parse_vis_from_filename(f)
        if site and vis is not None:
            rows.append({"site": site, "vis": vis})

    return pd.DataFrame(rows)


def stats_from_csvs():
    """Load all predictions CSVs and deduplicate by sample_path."""
    dfs = []
    for rel in PRED_CSVS:
        p = os.path.join(BASE, rel)
        if os.path.exists(p):
            print(f"  Loading {rel}")
            dfs.append(pd.read_csv(p))

    if not dfs:
        print("ERROR: No predictions CSVs found. Cannot compute stats.")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    # Keep unique images by sample_path
    combined = combined.drop_duplicates(subset="sample_path")

    rows = []
    for _, row in combined.iterrows():
        site = row.get("site_id", parse_site_from_filename(str(row["sample_path"])))
        vis  = float(row["true_vis"])
        rows.append({"site": site, "vis": vis})

    return pd.DataFrame(rows)


def print_stats(df):
    total     = len(df)
    n_sites   = df["site"].nunique()
    per_site  = df.groupby("site").size()

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total images   : {total:,}")
    print(f"  Total sites    : {n_sites}")
    print(f"  Per-site mean  : {per_site.mean():.1f}")
    print(f"  Per-site median: {per_site.median():.0f}")
    print(f"  Per-site min   : {per_site.min()}")
    print(f"  Per-site max   : {per_site.max()}")

    # Class distribution
    df["class"] = df["vis"].apply(vis_to_class)
    class_counts = df["class"].value_counts().sort_index()

    print("\n  Visibility class distribution:")
    print(f"  {'Class':<10} {'Count':>8} {'%':>7}")
    print("  " + "-" * 28)
    for i, name in enumerate(CLASS_NAMES):
        count = class_counts.get(i, 0)
        pct   = count / total * 100 if total > 0 else 0
        print(f"  {name:<10} {count:>8,} {pct:>6.1f}%")
    print("  " + "-" * 28)
    print(f"  {'TOTAL':<10} {total:>8,} {'100.0%':>7}")
    print("=" * 60 + "\n")

    # Table 1 copy-paste block for the manuscript
    print("LaTeX snippet for Table 1 (paste into manuscript.tex):")
    print("-" * 60)
    for i, name in enumerate(CLASS_NAMES):
        count = class_counts.get(i, 0)
        pct   = count / total * 100 if total > 0 else 0
        print(f"  {name} & {count:,} & {pct:.1f}\\% \\\\")
    print(f"  \\midrule")
    print(f"  Total & {total:,} & 100.0\\% \\\\")
    print("-" * 60)
    print(f"\nSummary line: {total:,} images across {n_sites} sites "
          f"(mean {per_site.mean():.0f}, range {per_site.min()}–{per_site.max()} per site)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true",
                        help="Force CSV fallback mode (skip image scan)")
    args = parser.parse_args()

    df = None

    if not args.csv:
        for candidate in DSET_CANDIDATES:
            if os.path.isdir(candidate):
                df = stats_from_images(candidate)
                break
        if df is None:
            print("Dataset root not found. Falling back to predictions CSVs.")

    if df is None:
        df = stats_from_csvs()

    if df is None or df.empty:
        print("No data available. Check DSET_CANDIDATES or run an experiment first.")
        return

    print_stats(df)


if __name__ == "__main__":
    main()
