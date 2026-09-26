"""
Full ablation ladder analysis: A0 (matched baseline) -> A1 (TFFM branch-only,
structural losses zeroed) -> A2 (TFFM + clDice-only) -> A3 (full topology:
clDice + skeleton + endpoint + junction).

For each dataset x TTA setting, loads per-image metrics for all four models,
merges them on `sample`, and reports:
  1. Per-model descriptive means (dataset-level summary).
  2. Paired significance tests (Wilcoxon signed-rank + paired t-test) for
     three adjacent-step comparisons that isolate specific effects, plus the
     total effect:
       A1 - A0  : effect of adding TFFM architecture/capacity alone
                  (structural loss weights = 0, so no structural supervision)
       A2 - A1  : effect of adding clDice supervision on top of the TFFM
                  branch (branch already present in both)
       A3 - A2  : effect of adding skeleton + endpoint + junction supervision
                  on top of clDice (both already have clDice)
       A3 - A0  : total effect of the full topology model over the matched
                  baseline (already computed previously; recomputed here for
                  a single consistent table)
  3. For the three topology-branch-only metrics (skeleton_f1, endpoint_f1,
     junction_f1) that A0 does not produce, the same tests are run only for
     the comparisons where both sides have that head active: A2-A1, A3-A2,
     A3-A1.

All numbers come directly from the existing evaluation_outputs/ CSVs -- no
metric is recomputed, averaged across runs, or altered.
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from pathlib import Path

ROOT = Path("evaluation_outputs")
DATASETS = ["DRIVE_test", "STARE", "CHASE_DB1", "HRF"]

COMMON_METRICS = [
    "dice", "iou", "sensitivity", "specificity", "precision", "roc_auc",
    "cldice", "thin_vessel_dice", "thin_vessel_precision", "thin_vessel_recall",
]

BRANCH_METRICS = ["skeleton_f1", "endpoint_f1", "junction_f1"]

MATCHED_BASELINE_NO_TTA_DIRS = {
    "DRIVE_test": "drive_no_tta",
    "CHASE_DB1": "chase_no_tta",
    "HRF": "hrf_no_tta",
    "STARE": "stare_no_tta",
}


def path_for(model, dataset, tta):
    if model == "A0_matched_baseline":
        sub = "all_datasets_tta" if tta else MATCHED_BASELINE_NO_TTA_DIRS[dataset]
        return ROOT / "matched_baseline" / sub / dataset / "per_image_metrics.csv"
    if model == "A3_matched_topology":
        sub = "all_datasets_tta" if tta else MATCHED_BASELINE_NO_TTA_DIRS[dataset]
        return ROOT / "matched_topology" / sub / dataset / "per_image_metrics.csv"
    if model == "A1_branch_only":
        sub = "ablation_A1_branch_only_tta" if tta else "ablation_A1_branch_only_no_tta"
        return ROOT / sub / dataset / "per_image_metrics.csv"
    if model == "A2_cldice_only":
        sub = "ablation_A2_cldice_only_tta" if tta else "ablation_A2_cldice_only_no_tta"
        return ROOT / sub / dataset / "per_image_metrics.csv"
    raise ValueError(model)


MODELS = ["A0_matched_baseline", "A1_branch_only", "A2_cldice_only", "A3_matched_topology"]


def load(model, dataset, tta):
    p = path_for(model, dataset, tta)
    if not p.is_file():
        raise FileNotFoundError(f"{model}/{dataset}/tta={tta}: missing {p}")
    return pd.read_csv(p)


def paired_test(a, b):
    diff = a - b
    mean_diff = diff.mean()
    median_diff = np.median(diff)
    if np.allclose(diff, 0):
        w_stat, p_wil = np.nan, 1.0
    else:
        try:
            w_stat, p_wil = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, p_wil = np.nan, np.nan
    t_stat, p_t = ttest_rel(a, b)
    return {
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": p_wil,
        "paired_t_stat": t_stat,
        "paired_t_p": p_t,
        "sig_wilcoxon_0.05": bool(p_wil < 0.05) if not np.isnan(p_wil) else False,
    }


COMPARISONS = [
    ("A1_minus_A0", "A1_branch_only", "A0_matched_baseline"),
    ("A2_minus_A1", "A2_cldice_only", "A1_branch_only"),
    ("A3_minus_A2", "A3_matched_topology", "A2_cldice_only"),
    ("A3_minus_A0", "A3_matched_topology", "A0_matched_baseline"),
]

BRANCH_COMPARISONS = [
    ("A2_minus_A1", "A2_cldice_only", "A1_branch_only"),
    ("A3_minus_A2", "A3_matched_topology", "A2_cldice_only"),
    ("A3_minus_A1", "A3_matched_topology", "A1_branch_only"),
]

summary_rows = []
sig_rows = []
branch_sig_rows = []

for tta in (False, True):
    for dataset in DATASETS:
        loaded = {m: load(m, dataset, tta) for m in MODELS}

        for m in MODELS:
            df = loaded[m]
            row = {"tta": tta, "dataset": dataset, "model": m, "n": len(df)}
            for metric in COMMON_METRICS:
                row[metric] = df[metric].mean()
            for metric in BRANCH_METRICS:
                row[metric] = df[metric].mean() if metric in df.columns else np.nan
            summary_rows.append(row)

        for label, m_a, m_b in COMPARISONS:
            merged = loaded[m_a][["sample"] + COMMON_METRICS].merge(
                loaded[m_b][["sample"] + COMMON_METRICS],
                on="sample", suffixes=("_a", "_b"),
            )
            n = len(merged)
            for metric in COMMON_METRICS:
                a = merged[f"{metric}_a"].to_numpy()
                b = merged[f"{metric}_b"].to_numpy()
                res = paired_test(a, b)
                sig_rows.append({
                    "tta": tta, "dataset": dataset, "comparison": label,
                    "model_a": m_a, "model_b": m_b, "metric": metric, "n": n,
                    **res,
                })

        for label, m_a, m_b in BRANCH_COMPARISONS:
            merged = loaded[m_a][["sample"] + BRANCH_METRICS].merge(
                loaded[m_b][["sample"] + BRANCH_METRICS],
                on="sample", suffixes=("_a", "_b"),
            )
            n = len(merged)
            for metric in BRANCH_METRICS:
                a = merged[f"{metric}_a"].to_numpy()
                b = merged[f"{metric}_b"].to_numpy()
                res = paired_test(a, b)
                branch_sig_rows.append({
                    "tta": tta, "dataset": dataset, "comparison": label,
                    "model_a": m_a, "model_b": m_b, "metric": metric, "n": n,
                    **res,
                })

summary_df = pd.DataFrame(summary_rows)
sig_df = pd.DataFrame(sig_rows)
branch_sig_df = pd.DataFrame(branch_sig_rows)

Path("analysis_outputs").mkdir(exist_ok=True)
summary_df.to_csv("analysis_outputs/ablation_ladder_summary_means.csv", index=False)
sig_df.to_csv("analysis_outputs/ablation_ladder_paired_significance.csv", index=False)
branch_sig_df.to_csv("analysis_outputs/ablation_ladder_branch_paired_significance.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

print("=" * 110)
print("PER-MODEL DATASET-LEVEL MEANS (dice, cldice, thin_vessel_dice only, for brevity)")
print("=" * 110)
for tta in (False, True):
    print(f"\n### TTA = {tta} ###")
    for dataset in DATASETS:
        sub = summary_df[(summary_df.tta == tta) & (summary_df.dataset == dataset)]
        print(f"\n-- {dataset} --")
        print(sub[["model", "n", "dice", "cldice", "thin_vessel_dice", "skeleton_f1", "endpoint_f1", "junction_f1"]].to_string(index=False))

print("\n" + "=" * 110)
print("PAIRED SIGNIFICANCE -- COMMON METRICS (dice, cldice, thin_vessel_dice shown)")
print("=" * 110)
for tta in (False, True):
    print(f"\n### TTA = {tta} ###")
    for label, _, _ in COMPARISONS:
        print(f"\n-- {label} --")
        sub = sig_df[(sig_df.tta == tta) & (sig_df.comparison == label) &
                     (sig_df.metric.isin(["dice", "cldice", "thin_vessel_dice"]))]
        print(sub[["dataset", "metric", "n", "mean_diff", "wilcoxon_p", "sig_wilcoxon_0.05"]].to_string(index=False))

print("\n" + "=" * 110)
print("PAIRED SIGNIFICANCE -- BRANCH METRICS (skeleton_f1, endpoint_f1, junction_f1)")
print("=" * 110)
for tta in (False, True):
    print(f"\n### TTA = {tta} ###")
    for label, _, _ in BRANCH_COMPARISONS:
        print(f"\n-- {label} --")
        sub = branch_sig_df[(branch_sig_df.tta == tta) & (branch_sig_df.comparison == label)]
        print(sub[["dataset", "metric", "n", "mean_diff", "wilcoxon_p", "sig_wilcoxon_0.05"]].to_string(index=False))

print("\nSaved:")
print("  analysis_outputs/ablation_ladder_summary_means.csv")
print("  analysis_outputs/ablation_ladder_paired_significance.csv")
print("  analysis_outputs/ablation_ladder_branch_paired_significance.csv")
