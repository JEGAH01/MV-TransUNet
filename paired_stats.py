import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from pathlib import Path

ROOT = Path("evaluation_outputs")

NO_TTA_DIRS = {
    "DRIVE_test": "drive_no_tta",
    "CHASE_DB1": "chase_no_tta",
    "HRF": "hrf_no_tta",
    "STARE": "stare_no_tta",
}

METRICS = [
    "dice", "iou", "sensitivity", "specificity", "precision", "roc_auc",
    "cldice", "thin_vessel_dice", "thin_vessel_precision", "thin_vessel_recall",
]

def load(model_dir, dataset, tta):
    if tta:
        p = ROOT / model_dir / "all_datasets_tta" / dataset / "per_image_metrics.csv"
    else:
        p = ROOT / model_dir / NO_TTA_DIRS[dataset] / dataset / "per_image_metrics.csv"
    df = pd.read_csv(p)
    return df

rows = []
for tta in (False, True):
    for dataset in ["DRIVE_test", "STARE", "CHASE_DB1", "HRF"]:
        base = load("matched_baseline", dataset, tta)
        topo = load("matched_topology", dataset, tta)
        merged = base[["sample"] + METRICS].merge(
            topo[["sample"] + METRICS], on="sample", suffixes=("_base", "_topo")
        )
        n = len(merged)
        for metric in METRICS:
            b = merged[f"{metric}_base"].to_numpy()
            t = merged[f"{metric}_topo"].to_numpy()
            diff = t - b
            mean_diff = diff.mean()
            median_diff = np.median(diff)
            if np.allclose(diff, 0):
                w_stat, p_wil = np.nan, 1.0
            else:
                try:
                    w_stat, p_wil = wilcoxon(t, b, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    w_stat, p_wil = np.nan, np.nan
            t_stat, p_t = ttest_rel(t, b)
            rows.append({
                "tta": tta,
                "dataset": dataset,
                "metric": metric,
                "n": n,
                "mean_diff_topo_minus_base": mean_diff,
                "median_diff_topo_minus_base": median_diff,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": p_wil,
                "paired_t_stat": t_stat,
                "paired_t_p": p_t,
                "sig_wilcoxon_0.05": bool(p_wil < 0.05) if not np.isnan(p_wil) else False,
            })

out = pd.DataFrame(rows)
out.to_csv("analysis_outputs/matched_paired_significance.csv", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

for tta in (False, True):
    print("=" * 100)
    print("TTA =", tta)
    print("=" * 100)
    sub = out[out.tta == tta]
    for dataset in ["DRIVE_test", "STARE", "CHASE_DB1", "HRF"]:
        print(f"\n--- {dataset} ---")
        cols = ["metric", "n", "mean_diff_topo_minus_base", "wilcoxon_p", "sig_wilcoxon_0.05"]
        print(sub[sub.dataset == dataset][cols].to_string(index=False))

print("\nSaved full table to analysis_outputs/matched_paired_significance.csv")
