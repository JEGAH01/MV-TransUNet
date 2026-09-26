#!/usr/bin/env python3
"""
Precompute per-image scale_ratio values for SC-VAM (scale-conditioned
Vessel Attention Module) inference.

At training time, RandomPatchTopologyDataset / RandomPatchRetinalDataset
already know the exact scale_factor applied to each patch (it's the random
draw made by the existing scale-augmentation code). At test time, on
STARE/CHASE_DB1/HRF (zero-shot, no augmentation), there is no such ground
truth -- so this script estimates each test image's own vessel scale using
the existing VSDI Frangi-based estimator (src/vessel_scale_estimator.py),
the same tool that already produced the paper's cited DRIVE/STARE/CHASE_DB1/
HRF Frangi-sigma gap figures, and expresses it as a ratio to the DRIVE
source scale already calibrated in drive_source_reference.yaml (see
src/calibrate_vessel_scale.py).

    scale_ratio(image) = weighted_median_sigma(image) / weighted_median_sigma(DRIVE reference)

This is a one-time, per-image preprocessing cost (not per training step,
and not repeated across evaluation runs once cached): run this once per
dataset, then pass --scale-ratio-cache <output_json> to
evaluate_topology_v2.py / evaluate_baseline_v2.py.

Usage: run once per evaluation dataset (DRIVE_test, STARE, CHASE_DB1, HRF),
pointing --output-json at the SAME combined cache file each time -- results
are merged in (loaded, updated, rewritten), not overwritten, since the
evaluation scripts load one shared --scale-ratio-cache file covering every
dataset in a single run. Each entry is keyed "<dataset-name>/<filename>",
NOT bare filename -- these datasets reuse generic sequential names like
"002.jpg" across datasets (confirmed: STARE/CHASE_DB1/HRF collide), so a
bare-filename key would silently mix up different images' scale_ratio:

    python -m src.precompute_scale_ratios \\
        --dataset-name DRIVE_test \\
        --image-dir datasets_processed/DRIVE_test/images \\
        --output-json cache/scale_ratios/all_scale_ratios.json \\
        --diagnostics-dir cache/vessel_scale/drive_test_scale

    python -m src.precompute_scale_ratios \\
        --dataset-name STARE \\
        --image-dir datasets_processed/STARE/images \\
        --output-json cache/scale_ratios/all_scale_ratios.json \\
        --diagnostics-dir cache/vessel_scale/stare_test_scale

    (... repeat for CHASE_DB1 and HRF, with --dataset-name CHASE_DB1 / HRF ...)

--dataset-name must exactly match the dataset name evaluate_topology_v2.py /
evaluate_baseline_v2.py uses (DatasetSpecification.name -- "DRIVE_test",
"STARE", "CHASE_DB1", "HRF" by default; see resolve_datasets()), since that
is the other half of the lookup key at evaluation time.

Uses EstimatorConfig() defaults throughout, matching the settings that
produced drive_source_reference.yaml, so the ratio is apples-to-apples.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import yaml

from src.vessel_scale_estimator import (
    EstimatorConfig,
    VesselScaleResult,
    estimate_vessel_scale,
)

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".bmp"
}


def discover_images(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory was not found: {image_dir.resolve()}"
        )
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(
            f"No supported images were found in {image_dir.resolve()}."
        )
    return images


def load_drive_reference_median(reference_path: Path) -> float:
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"DRIVE source reference not found: {reference_path.resolve()}. "
            "Run src/calibrate_vessel_scale.py first (see "
            "cache/vessel_scale/drive_source_calibration/ for the existing one)."
        )
    with reference_path.open("r", encoding="utf-8") as stream:
        reference = yaml.safe_load(stream)
    try:
        return float(
            reference["scalar_statistics"]["weighted_median_sigma"]["mean"]
        )
    except (KeyError, TypeError) as error:
        raise KeyError(
            "drive_source_reference.yaml is missing "
            "scalar_statistics.weighted_median_sigma.mean."
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute per-image scale_ratio (relative to the DRIVE "
            "source scale) for SC-VAM inference-time conditioning."
        )
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help=(
            "Must match the dataset name used by evaluate_topology_v2.py / "
            "evaluate_baseline_v2.py (e.g. DRIVE_test, STARE, CHASE_DB1, "
            "HRF) -- entries are cached as '<dataset-name>/<filename>' so "
            "identically-named files in different datasets don't collide."
        ),
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        required=True,
        help="Where estimate_vessel_scale() writes its per-image diagnostics.",
    )
    parser.add_argument(
        "--drive-reference",
        type=Path,
        default=Path(
            "cache/vessel_scale/drive_source_calibration/drive_source_reference.yaml"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional audit CSV with per-image weighted_median, "
            "confidence, and the resulting scale_ratio. Defaults to "
            "<output-json>.audit.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    drive_reference_median = load_drive_reference_median(
        arguments.drive_reference
    )
    print(
        "DRIVE reference weighted-median sigma:",
        f"{drive_reference_median:.4f}",
        f"(from {arguments.drive_reference.resolve()})",
    )

    images = discover_images(arguments.image_dir)
    print(f"Found {len(images)} images in {arguments.image_dir.resolve()}")

    arguments.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    config = EstimatorConfig()

    scale_ratios: Dict[str, float] = {}
    if arguments.output_json.is_file():
        with arguments.output_json.open("r", encoding="utf-8") as stream:
            scale_ratios = {str(k): float(v) for k, v in json.load(stream).items()}
        print(
            f"Merging into existing cache ({len(scale_ratios)} images already "
            f"present in {arguments.output_json.resolve()})"
        )

    audit_rows: List[Dict[str, object]] = []

    for index, image_path in enumerate(images, start=1):
        image_output_dir = arguments.diagnostics_dir / image_path.stem
        cache_key = f"{arguments.dataset_name}/{image_path.name}"
        print(f"[{index}/{len(images)}] {cache_key}")

        result: VesselScaleResult = estimate_vessel_scale(
            image_path=image_path,
            output_dir=image_output_dir,
            config=config,
            save_arrays=False,
        )

        scale_ratio = result.weighted_median / drive_reference_median

        if cache_key in scale_ratios:
            print(
                f"    WARNING: '{cache_key}' already had a cached "
                f"scale_ratio ({scale_ratios[cache_key]:.4f}); "
                f"overwriting with {scale_ratio:.4f}. This means the same "
                "--dataset-name/--image-dir pair was run twice -- check you "
                "didn't mean a different --dataset-name."
            )

        scale_ratios[cache_key] = float(scale_ratio)

        audit_rows.append(
            {
                "dataset": arguments.dataset_name,
                "image": image_path.name,
                "cache_key": cache_key,
                "weighted_median_sigma": result.weighted_median,
                "drive_reference_weighted_median_sigma": drive_reference_median,
                "scale_ratio": scale_ratio,
                "confidence": result.confidence,
                "status": result.status,
            }
        )

        print(
            f"    weighted_median={result.weighted_median:.4f}  "
            f"scale_ratio={scale_ratio:.4f}  "
            f"confidence={result.confidence:.3f}  "
            f"status={result.status}"
        )

    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output_json.open("w", encoding="utf-8") as stream:
        json.dump(scale_ratios, stream, indent=2, sort_keys=True)
    print(f"\nWrote {arguments.output_json.resolve()} ({len(scale_ratios)} images)")

    # Default audit CSV name includes the dataset name so repeated
    # invocations (one per dataset, merging into the same --output-json)
    # don't overwrite each other's audit trail.
    output_csv = arguments.output_csv or (
        arguments.output_json.parent
        / f"{arguments.output_json.stem}.{arguments.dataset_name}.audit.csv"
    )
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"Wrote {output_csv.resolve()}")

    non_finite_or_low_confidence = [
        row for row in audit_rows if row["confidence"] < 0.5
    ]
    if non_finite_or_low_confidence:
        print(
            f"\nWARNING: {len(non_finite_or_low_confidence)} image(s) had "
            "estimator confidence below 0.5 -- inspect the audit CSV before "
            "trusting their scale_ratio:"
        )
        for row in non_finite_or_low_confidence:
            print(f"    {row['image']}: confidence={row['confidence']:.3f}")


if __name__ == "__main__":
    main()
