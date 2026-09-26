#!/usr/bin/env python
"""
Stage B: DRIVE source vessel-scale calibration.

Run:
python -m src.calibrate_vessel_scale ^
  --image-dir "datasets/DRIVE/training/image" ^
  --output-dir "cache/vessel_scale/drive_source_calibration"
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.vessel_scale_estimator import (
    DEFAULT_SIGMAS,
    EstimatorConfig,
    ScaleGroupStatistics,
    VesselScaleResult,
    estimate_vessel_scale,
)

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".bmp"
}


def parse_sigma_list(text: str) -> Tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in text.split(",") if x.strip()}))
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("Sigmas must be positive.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a DRIVE source-domain vessel-scale reference."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sigmas", type=parse_sigma_list, default=DEFAULT_SIGMAS)
    parser.add_argument("--response-quantile", type=float, default=0.90)
    parser.add_argument("--minimum-valid-pixels", type=int, default=500)
    parser.add_argument("--minimum-component-area", type=int, default=8)
    parser.add_argument(
        "--fov-border-exclusion-fraction", type=float, default=0.06
    )
    parser.add_argument("--fine-scale-max-sigma", type=float, default=2.50)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-grid-size", type=int, default=8)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=0.70)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def natural_key(path: Path) -> List[object]:
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", path.name)]


def discover_images(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory was not found: {image_dir.resolve()}"
        )
    images = sorted(
        [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=natural_key,
    )
    if not images:
        raise FileNotFoundError(
            f"No supported images were found in {image_dir.resolve()}."
        )
    return images


def load_cached_result(path: Path) -> VesselScaleResult | None:
    if not path.exists():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = dict(payload["result"])
        data["fine_scale"] = ScaleGroupStatistics(**data["fine_scale"])
        data["coarse_scale"] = ScaleGroupStatistics(**data["coarse_scale"])
        return VesselScaleResult(**data)
    except Exception as exc:
        print(f"  Cache ignored because it could not be read: {exc}")
        return None


def finite_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def summarize(values: Sequence[float]) -> Dict[str, float]:
    array = finite_array(values)
    if array.size == 0:
        return {k: float("nan") for k in (
            "mean", "std", "median", "q025", "q25",
            "q75", "q975", "minimum", "maximum"
        )}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q975": float(np.quantile(array, 0.975)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def profile_matrix(
    results: Sequence[VesselScaleResult],
    sigmas: Sequence[float],
) -> np.ndarray:
    return np.asarray(
        [
            [
                result.scale_profile[f"sigma_{float(sigma):g}"]
                for sigma in sigmas
            ]
            for result in results
        ],
        dtype=np.float64,
    )


def build_reference(
    results: Sequence[VesselScaleResult],
    config: EstimatorConfig,
    image_dir: Path,
) -> Dict[str, object]:
    accepted = [
        result for result in results
        if result.status not in {"failed_no_valid_vessel_pixels", "low_support"}
        and np.isfinite(result.weighted_median)
    ]
    if not accepted:
        raise RuntimeError("No valid DRIVE calibration results were produced.")

    matrix = profile_matrix(accepted, config.sigmas)
    mean_profile = np.mean(matrix, axis=0)
    mean_profile /= max(float(mean_profile.sum()), 1e-12)
    covariance = (
        np.cov(matrix, rowvar=False, ddof=1)
        if len(accepted) > 1
        else np.zeros((len(config.sigmas), len(config.sigmas)))
    )

    status_counts: Dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    scalar_stats = {
        "weighted_median_sigma": summarize(
            [r.weighted_median for r in accepted]
        ),
        "weighted_mean_sigma": summarize(
            [r.weighted_mean for r in accepted]
        ),
        "weighted_std_sigma": summarize(
            [r.weighted_std for r in accepted]
        ),
        "fine_scale_weight_fraction": summarize(
            [r.fine_scale.vesselness_weight_fraction for r in accepted]
        ),
        "coarse_scale_weight_fraction": summarize(
            [r.coarse_scale.vesselness_weight_fraction for r in accepted]
        ),
        "normalized_entropy": summarize(
            [r.normalized_entropy for r in accepted]
        ),
        "histogram_concentration": summarize(
            [r.histogram_concentration for r in accepted]
        ),
        "confidence": summarize(
            [r.confidence for r in accepted]
        ),
        "boundary_response_fraction": summarize(
            [r.boundary_response_fraction for r in accepted]
        ),
        "valid_vessel_fraction_of_fov": summarize(
            [r.valid_vessel_fraction_of_fov for r in accepted]
        ),
    }

    profile_stats = {}
    for index, sigma in enumerate(config.sigmas):
        key = f"sigma_{float(sigma):g}"
        profile_stats[key] = summarize(matrix[:, index])

    return {
        "method": "VSDI_v2_1_source_calibration",
        "source_dataset": "DRIVE",
        "source_image_directory": str(image_dir.resolve()),
        "number_of_discovered_images": len(results),
        "number_of_accepted_images": len(accepted),
        "status_counts": status_counts,
        "configuration": {
            **asdict(config),
            "sigmas": [float(x) for x in config.sigmas],
        },
        "scalar_statistics": scalar_stats,
        "source_profile": {
            f"sigma_{float(s):g}": float(v)
            for s, v in zip(config.sigmas, mean_profile)
        },
        "profile_statistics": profile_stats,
        "profile_covariance": covariance.tolist(),
        "per_image_summary": [
            {
                "image_name": Path(r.image_path).name,
                "status": r.status,
                "weighted_median_sigma": float(r.weighted_median),
                "weighted_mean_sigma": float(r.weighted_mean),
                "fine_scale_weight_fraction": float(
                    r.fine_scale.vesselness_weight_fraction
                ),
                "coarse_scale_weight_fraction": float(
                    r.coarse_scale.vesselness_weight_fraction
                ),
                "normalized_entropy": float(r.normalized_entropy),
                "confidence": float(r.confidence),
                "boundary_response_fraction": float(
                    r.boundary_response_fraction
                ),
                "scale_profile": {
                    k: float(v) for k, v in r.scale_profile.items()
                },
            }
            for r in accepted
        ],
    }


def write_csv(
    path: Path,
    results: Sequence[VesselScaleResult],
    sigmas: Sequence[float],
) -> None:
    profile_fields = [f"sigma_{float(s):g}" for s in sigmas]
    fields = [
        "image_name", "status", "weighted_median_sigma",
        "weighted_mean_sigma", "weighted_q25", "weighted_q75",
        "weighted_std", "fine_scale_median",
        "fine_scale_weight_fraction", "coarse_scale_median",
        "coarse_scale_weight_fraction", "normalized_entropy",
        "histogram_concentration", "bootstrap_median_std",
        "bootstrap_median_q025", "bootstrap_median_q975",
        "valid_vessel_pixels", "valid_vessel_fraction_of_fov",
        "boundary_response_fraction", "support_score",
        "bootstrap_stability_score", "boundary_cleanliness_score",
        "confidence", *profile_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {
                "image_name": Path(r.image_path).name,
                "status": r.status,
                "weighted_median_sigma": r.weighted_median,
                "weighted_mean_sigma": r.weighted_mean,
                "weighted_q25": r.weighted_q25,
                "weighted_q75": r.weighted_q75,
                "weighted_std": r.weighted_std,
                "fine_scale_median": r.fine_scale.weighted_median,
                "fine_scale_weight_fraction":
                    r.fine_scale.vesselness_weight_fraction,
                "coarse_scale_median": r.coarse_scale.weighted_median,
                "coarse_scale_weight_fraction":
                    r.coarse_scale.vesselness_weight_fraction,
                "normalized_entropy": r.normalized_entropy,
                "histogram_concentration": r.histogram_concentration,
                "bootstrap_median_std": r.bootstrap_median_std,
                "bootstrap_median_q025": r.bootstrap_median_q025,
                "bootstrap_median_q975": r.bootstrap_median_q975,
                "valid_vessel_pixels": r.valid_vessel_pixels,
                "valid_vessel_fraction_of_fov":
                    r.valid_vessel_fraction_of_fov,
                "boundary_response_fraction": r.boundary_response_fraction,
                "support_score": r.support_score,
                "bootstrap_stability_score": r.bootstrap_stability_score,
                "boundary_cleanliness_score": r.boundary_cleanliness_score,
                "confidence": r.confidence,
            }
            row.update(r.scale_profile)
            writer.writerow(row)


def save_plots(
    output_dir: Path,
    results: Sequence[VesselScaleResult],
    sigmas: Sequence[float],
) -> None:
    matrix = profile_matrix(results, sigmas)
    positions = np.arange(len(sigmas))
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0, ddof=1) if len(results) > 1 else np.zeros(len(sigmas))

    fig, ax = plt.subplots(figsize=(10, 5))
    for row in matrix:
        ax.plot(positions, row, alpha=0.20, linewidth=1.0)
    ax.errorbar(
        positions, mean, yerr=std, marker="o", capsize=4,
        label="DRIVE mean ± standard deviation"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(x) for x in sigmas])
    ax.set_xlabel("Dominant sigma")
    ax.set_ylabel("Normalized vesselness weight")
    ax.set_title("DRIVE Source-Domain Vessel-Scale Profile")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "source_scale_profile.png", dpi=250)
    plt.close(fig)

    labels = [Path(r.image_path).stem for r in results]
    medians = [r.weighted_median for r in results]
    x = np.arange(len(results))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, medians)
    ax.axhline(
        float(np.mean(medians)), linestyle="--",
        label=f"Mean = {np.mean(medians):.3f}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("DRIVE training image")
    ax.set_ylabel("Weighted median sigma")
    ax.set_title("Per-Image Source Vessel-Scale Median")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_image_median_sigma.png", dpi=250)
    plt.close(fig)

    fine = np.asarray(
        [r.fine_scale.vesselness_weight_fraction for r in results]
    )
    coarse = np.asarray(
        [r.coarse_scale.vesselness_weight_fraction for r in results]
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, fine, label="Fine-scale fraction")
    ax.bar(x, coarse, bottom=fine, label="Coarse-scale fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("DRIVE training image")
    ax.set_ylabel("Vesselness weight fraction")
    ax.set_title("Fine/Coarse Vessel-Scale Composition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fine_coarse_fraction.png", dpi=250)
    plt.close(fig)

    confidence = np.asarray([r.confidence for r in results])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidence, bins=min(10, max(5, len(confidence) // 2)))
    ax.axvline(
        float(np.mean(confidence)), linestyle="--",
        label=f"Mean = {np.mean(confidence):.3f}"
    )
    ax.set_xlabel("Estimator confidence")
    ax.set_ylabel("Number of images")
    ax.set_title("DRIVE Calibration Confidence Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_distribution.png", dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    images = discover_images(args.image_dir)

    config = EstimatorConfig(
        sigmas=tuple(args.sigmas),
        response_quantile=args.response_quantile,
        minimum_valid_pixels=args.minimum_valid_pixels,
        minimum_component_area=args.minimum_component_area,
        fov_border_exclusion_fraction=args.fov_border_exclusion_fraction,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_grid_size=args.clahe_grid_size,
        fine_scale_max_sigma=args.fine_scale_max_sigma,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_sample_fraction=args.bootstrap_sample_fraction,
        bootstrap_seed=args.bootstrap_seed,
    )

    output_dir = args.output_dir
    per_image_dir = output_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    print("\nStage B — DRIVE Source Vessel-Scale Calibration")
    print("=" * 78)
    print(f"Images found:     {len(images)}")
    print(f"Image directory:  {args.image_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print("=" * 78)

    results: List[VesselScaleResult] = []

    for index, image_path in enumerate(images, start=1):
        image_output = per_image_dir / image_path.stem
        diagnostics = image_output / "diagnostics.yaml"

        print(f"\n[{index:02d}/{len(images):02d}] {image_path.name}")

        result = None
        if not args.overwrite:
            result = load_cached_result(diagnostics)
            if result is not None:
                print("  Using cached diagnostics.")

        if result is None:
            try:
                result = estimate_vessel_scale(
                    image_path=image_path,
                    output_dir=image_output,
                    config=config,
                    save_arrays=args.save_arrays,
                )
            except Exception as exc:
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                continue

        results.append(result)
        print(
            f"  status={result.status}, "
            f"median={result.weighted_median:.3f}, "
            f"mean={result.weighted_mean:.3f}, "
            f"fine={result.fine_scale.vesselness_weight_fraction:.3f}, "
            f"confidence={result.confidence:.3f}"
        )

    if not results:
        raise RuntimeError("Calibration failed for every image.")

    reference = build_reference(results, config, args.image_dir)

    (output_dir / "drive_source_reference.yaml").write_text(
        yaml.safe_dump(reference, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "drive_source_calibration.csv",
        results,
        config.sigmas,
    )

    accepted = [
        r for r in results
        if r.status not in {"failed_no_valid_vessel_pixels", "low_support"}
        and np.isfinite(r.weighted_median)
    ]
    save_plots(output_dir, accepted, config.sigmas)

    scalar = reference["scalar_statistics"]
    median_stats = scalar["weighted_median_sigma"]
    mean_stats = scalar["weighted_mean_sigma"]
    fine_stats = scalar["fine_scale_weight_fraction"]
    confidence_stats = scalar["confidence"]

    print("\n" + "=" * 78)
    print("DRIVE SOURCE CALIBRATION COMPLETE")
    print("=" * 78)
    print(
        f"Images accepted:                 "
        f"{reference['number_of_accepted_images']}/{len(results)}"
    )
    print(f"Status counts:                   {reference['status_counts']}")
    print("-" * 78)
    print(
        f"Weighted median sigma:           "
        f"{median_stats['mean']:.4f} ± {median_stats['std']:.4f}"
    )
    print(
        f"Weighted median 95% interval:    "
        f"{median_stats['q025']:.4f}-{median_stats['q975']:.4f}"
    )
    print(
        f"Weighted mean sigma:             "
        f"{mean_stats['mean']:.4f} ± {mean_stats['std']:.4f}"
    )
    print(
        f"Fine-scale weight fraction:      "
        f"{fine_stats['mean']:.4f} ± {fine_stats['std']:.4f}"
    )
    print(
        f"Estimator confidence:            "
        f"{confidence_stats['mean']:.4f} ± {confidence_stats['std']:.4f}"
    )
    print("-" * 78)
    print("Source profile:")
    for key, value in reference["source_profile"].items():
        print(f"  {key:<16} {value:.6f}")
    print("=" * 78)
    print("\nOutputs saved to:")
    print(f"  {output_dir.resolve()}")


if __name__ == "__main__":
    main()