#!/usr/bin/env python3
"""Same methodology as paired_significance_baseline_v3_vs_a8.py, scoped to
seed 123 only, writing to a separate output file so the original seed42/seed7
CSV is untouched. Not a new statistical method -- identical Wilcoxon
signed-rank procedure already used for Table 4."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent
DATASETS = ["DRIVE_test", "STARE", "CHASE_DB1", "HRF"]
RUN_PAIRS = [
    (123, "no_tta",
     "evaluation_outputs/baseline_scvam_seed123_no_tta",
     "evaluation_outputs/topology_a8_scvam_seed123_no_tta"),
    (123, "tta",
     "evaluation_outputs/baseline_scvam_seed123_tta",
     "evaluation_outputs/topology_a8_scvam_seed123_tta"),
]

def load_dice(output_dir, dataset):
    csv_path = REPO_ROOT / output_dir / dataset / "per_image_metrics.csv"
    if not csv_path.is_file():
        print(f"  [skip] missing: {csv_path}")
        return None
    frame = pd.read_csv(csv_path)
    return frame[["sample", "dice"]].rename(columns={"dice": "dice"})

def main():
    results: List[Dict[str, object]] = []
    for seed, tta_label, baseline_dir, topology_dir in RUN_PAIRS:
        print(f"\n=== seed={seed} tta={tta_label} ===")
        for dataset in DATASETS:
            baseline_frame = load_dice(baseline_dir, dataset)
            topology_frame = load_dice(topology_dir, dataset)
            if baseline_frame is None or topology_frame is None:
                continue
            merged = baseline_frame.merge(topology_frame, on="sample", suffixes=("_baseline", "_topology"), how="inner")
            if len(merged) != len(baseline_frame) or len(merged) != len(topology_frame):
                print(f"  WARNING [{dataset}]: image-count mismatch (baseline={len(baseline_frame)}, topology={len(topology_frame)}, matched={len(merged)})")
            diff = merged["dice_topology"] - merged["dice_baseline"]
            n = len(diff)
            favor_topology = int((diff > 0).sum())
            favor_baseline = int((diff < 0).sum())
            ties = int((diff == 0).sum())
            if n < 1 or diff.abs().sum() == 0:
                print(f"  {dataset}: n={n}, all differences zero -- skipping Wilcoxon")
                continue
            try:
                statistic, p_value = wilcoxon(diff, zero_method="wilcox")
            except ValueError as error:
                statistic, p_value = float("nan"), float("nan")
                print(f"  {dataset}: Wilcoxon failed ({error})")
            row = {
                "seed": seed, "tta": tta_label, "dataset": dataset, "n_images": n,
                "mean_diff_topology_minus_baseline": float(diff.mean()),
                "median_diff_topology_minus_baseline": float(diff.median()),
                "n_favor_topology": favor_topology, "n_favor_baseline": favor_baseline, "n_ties": ties,
                "wilcoxon_statistic": float(statistic), "p_value": float(p_value),
                "significant_0_05": bool(p_value < 0.05) if p_value == p_value else False,
            }
            results.append(row)
            print(f"  {dataset}: n={n}, mean_diff={row['mean_diff_topology_minus_baseline']:+.4f}, favor_A8={favor_topology}, favor_baseline={favor_baseline}, p={p_value:.4f}{' *' if row['significant_0_05'] else ''}")
    output_path = REPO_ROOT / "paired_significance_seed123.csv"
    if results:
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {output_path.resolve()} ({len(results)} rows)")
    else:
        print("\nNo results computed")

if __name__ == "__main__":
    main()
