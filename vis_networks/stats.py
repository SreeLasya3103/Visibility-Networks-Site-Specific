import csv, statistics

def load_persite(path, min_n=20):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row['n_total']) >= min_n:
                rows.append(row)
    return rows

def ms(vals):
    return statistics.mean(vals), statistics.stdev(vals)

paths = {
    "VisNet Scratch":       "VisNetPersiteAndPredictionsResults/persite_results.csv",
    "RMEP Scratch":         "RMEPPerSiteAndPredictionsResults/persite_results.csv",
    "VisNet Fine-tune":     "VisNetFineTuneAndPredictionsResults/persite_results.csv",
    "RMEP Fine-tune":       "RMEPFineTuneAndPredictionsResults/persite_results.csv",
    "VisNet-Similarity FT": "VisNetSimilarityPersiteFineTuneResults/persite_results.csv",
}

for name, path in paths.items():
    data = load_persite(path)
    r2_m, r2_s   = ms([float(r['R2'])              for r in data])
    mae_m, mae_s = ms([float(r['MAE_visibility'])  for r in data])
    rms_m, rms_s = ms([float(r['RMSE_visibility']) for r in data])
    print(f"{name}: sites={len(data)}  R2={r2_m:.3f}±{r2_s:.3f}  "
          f"MAE={mae_m:.3f}±{mae_s:.3f}  RMSE={rms_m:.3f}±{rms_s:.3f}")