"""Dry-run test for select_references.py — summary across all sites."""
import os, sys
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_references import extract_site_id, extract_visibility

SPLITS_PATH = r"D:\Research - Lasya\NewWebcams"

all_samples = []
for split in ["train.csv", "validation.csv", "test.csv"]:
    fp = os.path.join(SPLITS_PATH, split)
    if os.path.exists(fp):
        with open(fp) as f:
            all_samples.extend([l.rstrip() for l in f if l.strip()])

print(f"Total samples: {len(all_samples)}\n")

# Group by site
sites = defaultdict(list)
for s in all_samples:
    sites[extract_site_id(s)].append(s)

print(f"Total sites: {len(sites)}\n")

# Compute best_vis per site
site_best = {}
for site_id, samples in sites.items():
    best = -1
    best_sample = None
    for s in samples:
        v = extract_visibility(s)
        if v > best:
            best = v
            best_sample = s
    site_best[site_id] = (best, best_sample)

# Histogram of best_vis across sites
hist = Counter(round(v, 2) for v, _ in site_best.values())
print("=== Distribution of best_vis across all sites ===")
for vis in sorted(hist.keys(), reverse=True):
    bar = "#" * hist[vis]
    print(f"  best_vis={vis:5.2f}: {hist[vis]:4d} sites  {bar}")

# Show a few high-vis picks as proof
print("\n=== Sample high-vis picks (best_vis >= 15) ===")
high_vis_sites = [(sid, v, s) for sid, (v, s) in site_best.items() if v >= 15]
high_vis_sites.sort(key=lambda x: -x[1])
for sid, v, s in high_vis_sites[:10]:
    print(f"  {sid}: best_vis={v}  {s}")

if not high_vis_sites:
    print("  (none)")

# Show a few medium-vis picks
print("\n=== Sample sites with best_vis=9 (most common cap) ===")
cap9 = [(sid, s) for sid, (v, s) in site_best.items() if v == 9.0][:5]
for sid, s in cap9:
    print(f"  {sid}: {s}")
