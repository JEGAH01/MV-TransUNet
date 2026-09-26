#!/usr/bin/env python3
"""
Paired (per-image) Wilcoxon signed-rank test: baseline_v3 vs A8, on Dice,
matched by image ("sample" column) within each dataset.

This is a real statistical test computable RIGHT NOW from files already on
disk -- it does not require a third seed. It pairs each image's Dice score
between the two matched-protocol checkpoints (same images, same
reconstruction settings, only the topology branch + SC-VAM differ) and asks:
is the sign of the per-image difference consistent enough across the
dataset's images to call the aggregate difference real, rather than noise?

This is a different (and complementary) question to seed variance: seed
variance asks "does the AGGREGATE difference reproduce across training
seeds", this asks "within one trained pair, is the per-image pattern
consistent enough to be statistically distinguishable from zero". Both
matter; neither substitutes for the other.

Usage (run from the repo root, after all 8 evaluation runs -- 2 seeds x
{no_tta, tta} x {baseline_v3, A8} -- have produced their
evaluation_outputs/.../<DATASET>/per_image_metrics.csv files):

    python paired_significance_baseline_v3_vs_a8.py

Prints one row per (seed, tta_setting, dataset): n images, mean diff
(A8 - baseline_v3), median diff, how many images favor A8 vs baseline_v3,
Wilcoxon statistic and p-value (two-sided). Also writes a combined CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent

DATASETS = ["DRIVE_test", "STARE", "CHASE_DB1", "HRF"]

# (seed, tta_label, baseline_output_dir, topology_output_dir)
RUN_PAIRS = [
    (42, "no_tta",
     "evaluation_outputs/baseline_scvam_seed42_no_tta",
     "evaluation_outputs/topology_a8_scvam_seed42_no_tta"),
    (42, "tta",
     "evaluation_outputs/baseline_scvam_seed42_tta",
     "evaluation_outputs/topology_a8_scvam_seed42_tta"),
    (7, "no_tta",
     "evaluation_outputs/baseline_scvam_seed7_no_tta",
     "evaluation_outputs/topology_a8_scvam_seed7_no_tta"),
    (7, "tta",
     "evaluation_outputs/baseline_scvam_seed7_tta",
     "evaluation_outputs/topology_a8_scvam_seed7_tta"),
]


def load_dice(output_dir: str, dataset: str) -> Optional[pd.DataFrame]:
    csv_path = REPO_ROOT / output_dir / dataset / "per_image_metrics.csv"
    if not csv_path.is_file():
        print(f"  [skip] missing: {csv_path}")
        return None
    frame = pd.read_csv(csv_path)
    if "sample" not in frame.columns or "dice" not in frame.columns:
        raise KeyError(
            f"{csv_path}: expected 'sample' and 'dice' columns, found "
            f"{list(frame.columns)}"
        )
    return frame[["sample", "dice"]].rename(columns={"dice": "dice"})


def main() -> None:
    results: List[Dict[str, object]] = []

    for seed, tta_label, baseline_dir, topology_dir in RUN_PAIRS:
        print(f"\n=== seed={seed} tta={tta_label} ===")
        for dataset in DATASETS:
            baseline_frame = load_dice(baseline_dir, dataset)
            topology_frame = load_dice(topology_dir, dataset)
            if baseline_frame is None or topology_frame is None:
                continue

            merged = baseline_frame.merge(
                topology_frame,
                on="sample",
                suffixes=("_baseline", "_topology"),
                how="inner",
            )
            if len(merged) != len(baseline_frame) or len(merged) != len(topology_frame):
                print(
                    f"  WARNING [{dataset}]: image-count mismatch "
                    f"(baseline={len(baseline_frame)}, "
                    f"topology={len(topology_frame)}, "
                    f"matched={len(merged)}) -- check the two runs used the "
                    "same dataset/image set."
                )

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
                "seed": seed,
                "tta": tta_label,
                "dataset": dataset,
                "n_images": n,
                "mean_diff_topology_minus_baseline": float(diff.mean()),
                "median_diff_topology_minus_baseline": float(diff.median()),
                "n_favor_topology": favor_topology,
                "n_favor_baseline": favor_baseline,
                "n_ties": ties,
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
                "significant_0_05": bool(p_value < 0.05) if p_value == p_value else False,
            }
            results.append(row)

            print(
                f"  {dataset}: n={n}, mean_diff={row['mean_diff_topology_minus_baseline']:+.4f}, "
                f"favor_A8={favor_topology}, favor_baseline_v3={favor_baseline}, "
                f"p={p_value:.4f}{' *' if row['significant_0_05'] else ''}"
            )

    output_path = REPO_ROOT / "paired_significance_baseline_v3_vs_a8.csv"
    if results:
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {output_path.resolve()} ({len(results)} rows)")
    else:
        print("\nNo results computed -- check that the evaluation_outputs paths exist.")


if __name__ == "__main__":
    main()
