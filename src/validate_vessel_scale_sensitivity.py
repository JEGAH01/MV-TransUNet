#!/usr/bin/env python
"""
Stage B.5 — Vessel-Scale Sensitivity Validation
===============================================

Purpose
-------
Validate whether VSDI Version 2.1 responds predictably and monotonically
to controlled image resizing before using it for adaptive profile matching.

For each selected DRIVE image, this script:

1. Resizes the original image over a controlled scale range.
2. Runs the existing vessel-scale estimator on each resized image.
3. Records scalar statistics and the full vessel-scale profile.
4. Measures monotonic association between image scale and estimated scale.
5. Saves per-image and aggregate publication-style figures.
6. Produces CSV and YAML summaries for later paper analysis.

Recommended first run
---------------------
python -m src.validate_vessel_scale_sensitivity ^
    --image-dir "datasets/DRIVE/training/image" ^
    --image-names "21.png,29.png,34.png" ^
    --output-dir "cache/vessel_scale/scale_sensitivity_validation"

Default resize factors
----------------------
0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30

Outputs
-------
output_dir/
    resized_images/
        21/
        29/
        34/
    estimator_outputs/
        21/
        29/
        34/
    scale_sensitivity_results.csv
    scale_sensitivity_summary.yaml
    aggregate_median_vs_scale.png
    aggregate_mean_vs_scale.png
    aggregate_fine_fraction_vs_scale.png
    aggregate_profile_heatmap.png
    per_image/
        21_scale_response.png
        21_profile_evolution.png
        ...
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
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


SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".ppm",
    ".bmp",
)


def parse_float_list(text: str) -> Tuple[float, ...]:
    values: List[float] = []

    for item in text.split(","):
        item = item.strip()
        if not item:
            continue

        value = float(item)

        if value <= 0:
            raise argparse.ArgumentTypeError(
                "All resize factors must be greater than zero."
            )

        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one resize factor is required."
        )

    return tuple(sorted(set(values)))


def parse_sigma_list(text: str) -> Tuple[float, ...]:
    values: List[float] = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        value = float(item)

        if value <= 0:
            raise argparse.ArgumentTypeError(
                "All sigma values must be greater than zero."
            )

        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one sigma value is required."
        )

    return tuple(sorted(set(values)))


def parse_image_names(text: str) -> Tuple[str, ...]:
    names = tuple(
        item.strip()
        for item in text.split(",")
        if item.strip()
    )

    if not names:
        raise argparse.ArgumentTypeError(
            "At least one image name is required."
        )

    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate VSDI scale sensitivity under controlled image resizing."
        )
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing DRIVE source images.",
    )
    parser.add_argument(
        "--image-names",
        type=parse_image_names,
        default=("21.png", "29.png", "34.png"),
        help=(
            "Comma-separated image names. "
            "Default: 21.png,29.png,34.png"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for all validation outputs.",
    )
    parser.add_argument(
        "--resize-factors",
        type=parse_float_list,
        default=(0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30),
        help=(
            "Comma-separated resize factors. "
            "Default: 0.70,0.80,0.90,1.00,1.10,1.20,1.30"
        ),
    )
    parser.add_argument(
        "--sigmas",
        type=parse_sigma_list,
        default=DEFAULT_SIGMAS,
    )
    parser.add_argument(
        "--response-quantile",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--minimum-valid-pixels",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--minimum-component-area",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--fov-border-exclusion-fraction",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--fine-scale-max-sigma",
        type=float,
        default=2.50,
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--clahe-grid-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--bootstrap-sample-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--interpolation",
        choices=(
            "area",
            "linear",
            "cubic",
            "lanczos",
        ),
        default="area",
        help=(
            "Interpolation used for resizing. "
            "AREA is recommended for downscaling."
        ),
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Save estimator NumPy arrays for each scaled image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even when cached diagnostics exist.",
    )

    return parser.parse_args()


def interpolation_flag(name: str, scale_factor: float) -> int:
    if name == "area":
        if scale_factor < 1.0:
            return cv2.INTER_AREA
        return cv2.INTER_CUBIC

    mapping = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    return mapping[name]


def scale_tag(scale_factor: float) -> str:
    return f"{scale_factor:.2f}".replace(".", "p")


def resolve_image_path(
    image_dir: Path,
    image_name: str,
) -> Path:
    direct_path = image_dir / image_name

    if direct_path.exists():
        return direct_path

    requested_stem = Path(image_name).stem.lower()

    matches = [
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.stem.lower() == requested_stem
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        matches.sort()
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate image '{image_name}' in "
        f"{image_dir.resolve()}."
    )


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not read image: {path.resolve()}"
        )

    return image


def resize_image(
    image: np.ndarray,
    scale_factor: float,
    interpolation_name: str,
) -> np.ndarray:
    original_height, original_width = image.shape[:2]

    new_width = max(
        16,
        int(round(original_width * scale_factor)),
    )
    new_height = max(
        16,
        int(round(original_height * scale_factor)),
    )

    interpolation = interpolation_flag(
        interpolation_name,
        scale_factor,
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )

    return resized


def save_resized_image(
    image: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(path),
        image,
    )

    if not success:
        raise IOError(
            f"Failed to save resized image: {path.resolve()}"
        )


def load_cached_result(
    diagnostics_path: Path,
) -> VesselScaleResult | None:
    if not diagnostics_path.exists():
        return None

    try:
        payload = yaml.safe_load(
            diagnostics_path.read_text(
                encoding="utf-8",
            )
        )

        data = dict(payload["result"])
        data["fine_scale"] = ScaleGroupStatistics(
            **data["fine_scale"]
        )
        data["coarse_scale"] = ScaleGroupStatistics(
            **data["coarse_scale"]
        )

        return VesselScaleResult(**data)

    except Exception as error:
        print(
            "  Cached diagnostics ignored because they "
            f"could not be loaded: {error}"
        )
        return None


def finite_array(
    values: Sequence[float],
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return array[np.isfinite(array)]


def rank_values(
    values: Sequence[float],
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        array,
        kind="mergesort",
    )

    ranks = np.empty(
        array.shape,
        dtype=np.float64,
    )

    sorted_values = array[order]

    start = 0

    while start < len(sorted_values):
        end = start + 1

        while (
            end < len(sorted_values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        average_rank = 0.5 * (
            start + end - 1
        ) + 1.0

        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def pearson_correlation(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    x_array = np.asarray(
        x,
        dtype=np.float64,
    )
    y_array = np.asarray(
        y,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
    )

    x_array = x_array[valid]
    y_array = y_array[valid]

    if x_array.size < 2:
        return float("nan")

    x_std = float(np.std(x_array))
    y_std = float(np.std(y_array))

    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0

    return float(
        np.corrcoef(
            x_array,
            y_array,
        )[0, 1]
    )


def spearman_correlation(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    x_array = np.asarray(
        x,
        dtype=np.float64,
    )
    y_array = np.asarray(
        y,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
    )

    x_array = x_array[valid]
    y_array = y_array[valid]

    if x_array.size < 2:
        return float("nan")

    return pearson_correlation(
        rank_values(x_array),
        rank_values(y_array),
    )


def monotonicity_score(
    values: Sequence[float],
    expected_direction: str = "increasing",
) -> float:
    array = finite_array(values)

    if array.size < 2:
        return float("nan")

    differences = np.diff(array)

    if expected_direction == "increasing":
        valid_steps = differences >= -1e-12
    else:
        valid_steps = differences <= 1e-12

    return float(np.mean(valid_steps))


def linear_fit(
    x: Sequence[float],
    y: Sequence[float],
) -> Dict[str, float]:
    x_array = np.asarray(
        x,
        dtype=np.float64,
    )
    y_array = np.asarray(
        y,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
    )

    x_array = x_array[valid]
    y_array = y_array[valid]

    if x_array.size < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
        }

    slope, intercept = np.polyfit(
        x_array,
        y_array,
        deg=1,
    )

    predictions = slope * x_array + intercept

    residual_sum = float(
        np.sum(
            (y_array - predictions) ** 2
        )
    )

    total_sum = float(
        np.sum(
            (y_array - np.mean(y_array)) ** 2
        )
    )

    if total_sum <= 1e-12:
        r_squared = 0.0
    else:
        r_squared = 1.0 - (
            residual_sum / total_sum
        )

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def summarize_series(
    resize_factors: Sequence[float],
    values: Sequence[float],
) -> Dict[str, float]:
    fit = linear_fit(
        resize_factors,
        values,
    )

    return {
        "pearson_r": pearson_correlation(
            resize_factors,
            values,
        ),
        "spearman_rho": spearman_correlation(
            resize_factors,
            values,
        ),
        "monotonicity_score": monotonicity_score(
            values,
            expected_direction="increasing",
        ),
        **fit,
    }


def normalize_profile(
    profile: Sequence[float],
) -> np.ndarray:
    array = np.asarray(
        profile,
        dtype=np.float64,
    )

    total = float(np.sum(array))

    if total <= 1e-12:
        return np.zeros_like(array)

    return array / total


def profile_expected_sigma(
    result: VesselScaleResult,
    sigmas: Sequence[float],
) -> float:
    profile = np.asarray(
        [
            result.scale_profile[
                f"sigma_{float(sigma):g}"
            ]
            for sigma in sigmas
        ],
        dtype=np.float64,
    )

    profile = normalize_profile(profile)

    return float(
        np.sum(
            profile
            * np.asarray(
                sigmas,
                dtype=np.float64,
            )
        )
    )


def profile_center_of_mass_index(
    result: VesselScaleResult,
    sigmas: Sequence[float],
) -> float:
    profile = np.asarray(
        [
            result.scale_profile[
                f"sigma_{float(sigma):g}"
            ]
            for sigma in sigmas
        ],
        dtype=np.float64,
    )

    profile = normalize_profile(profile)

    indices = np.arange(
        len(sigmas),
        dtype=np.float64,
    )

    return float(
        np.sum(
            profile * indices
        )
    )


def build_row(
    image_path: Path,
    resized_path: Path,
    original_shape: Tuple[int, int],
    resized_shape: Tuple[int, int],
    resize_factor: float,
    result: VesselScaleResult,
    sigmas: Sequence[float],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "image_name": image_path.name,
        "image_stem": image_path.stem,
        "resize_factor": float(
            resize_factor
        ),
        "original_height": int(
            original_shape[0]
        ),
        "original_width": int(
            original_shape[1]
        ),
        "resized_height": int(
            resized_shape[0]
        ),
        "resized_width": int(
            resized_shape[1]
        ),
        "resized_image_path": str(
            resized_path.resolve()
        ),
        "status": result.status,
        "weighted_median_sigma": float(
            result.weighted_median
        ),
        "weighted_mean_sigma": float(
            result.weighted_mean
        ),
        "weighted_q25": float(
            result.weighted_q25
        ),
        "weighted_q75": float(
            result.weighted_q75
        ),
        "weighted_std": float(
            result.weighted_std
        ),
        "profile_expected_sigma": profile_expected_sigma(
            result,
            sigmas,
        ),
        "profile_center_of_mass_index": profile_center_of_mass_index(
            result,
            sigmas,
        ),
        "fine_scale_weight_fraction": float(
            result.fine_scale.vesselness_weight_fraction
        ),
        "coarse_scale_weight_fraction": float(
            result.coarse_scale.vesselness_weight_fraction
        ),
        "normalized_entropy": float(
            result.normalized_entropy
        ),
        "histogram_concentration": float(
            result.histogram_concentration
        ),
        "bootstrap_median_std": float(
            result.bootstrap_median_std
        ),
        "boundary_response_fraction": float(
            result.boundary_response_fraction
        ),
        "valid_vessel_pixels": int(
            result.valid_vessel_pixels
        ),
        "valid_vessel_fraction_of_fov": float(
            result.valid_vessel_fraction_of_fov
        ),
        "confidence": float(
            result.confidence
        ),
    }

    for sigma in sigmas:
        key = f"sigma_{float(sigma):g}"
        row[key] = float(
            result.scale_profile[key]
        )

    return row


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    sigmas: Sequence[float],
) -> None:
    profile_fields = [
        f"sigma_{float(sigma):g}"
        for sigma in sigmas
    ]

    fields = [
        "image_name",
        "image_stem",
        "resize_factor",
        "original_height",
        "original_width",
        "resized_height",
        "resized_width",
        "resized_image_path",
        "status",
        "weighted_median_sigma",
        "weighted_mean_sigma",
        "weighted_q25",
        "weighted_q75",
        "weighted_std",
        "profile_expected_sigma",
        "profile_center_of_mass_index",
        "fine_scale_weight_fraction",
        "coarse_scale_weight_fraction",
        "normalized_entropy",
        "histogram_concentration",
        "bootstrap_median_std",
        "boundary_response_fraction",
        "valid_vessel_pixels",
        "valid_vessel_fraction_of_fov",
        "confidence",
        *profile_fields,
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def group_rows_by_image(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, List[Mapping[str, object]]]:
    grouped: Dict[
        str,
        List[Mapping[str, object]],
    ] = {}

    for row in rows:
        image_stem = str(
            row["image_stem"]
        )

        grouped.setdefault(
            image_stem,
            [],
        ).append(row)

    for image_stem in grouped:
        grouped[image_stem] = sorted(
            grouped[image_stem],
            key=lambda item: float(
                item["resize_factor"]
            ),
        )

    return grouped


def build_summary(
    rows: Sequence[Mapping[str, object]],
    config: EstimatorConfig,
    interpolation_name: str,
) -> Dict[str, object]:
    grouped = group_rows_by_image(
        rows
    )

    metrics = (
        "weighted_median_sigma",
        "weighted_mean_sigma",
        "profile_expected_sigma",
        "profile_center_of_mass_index",
        "fine_scale_weight_fraction",
        "coarse_scale_weight_fraction",
    )

    per_image_summary: Dict[
        str,
        object,
    ] = {}

    for image_stem, image_rows in grouped.items():
        resize_factors = [
            float(row["resize_factor"])
            for row in image_rows
        ]

        metric_summary = {}

        for metric in metrics:
            values = [
                float(row[metric])
                for row in image_rows
            ]

            direction = (
                "decreasing"
                if metric
                == "fine_scale_weight_fraction"
                else "increasing"
            )

            fit = linear_fit(
                resize_factors,
                values,
            )

            metric_summary[metric] = {
                "pearson_r": pearson_correlation(
                    resize_factors,
                    values,
                ),
                "spearman_rho": spearman_correlation(
                    resize_factors,
                    values,
                ),
                "monotonicity_score": monotonicity_score(
                    values,
                    expected_direction=direction,
                ),
                **fit,
                "values": [
                    float(value)
                    for value in values
                ],
            }

        per_image_summary[image_stem] = {
            "resize_factors": resize_factors,
            "status_by_scale": [
                str(row["status"])
                for row in image_rows
            ],
            "metrics": metric_summary,
        }

    aggregate_summary = {}

    for metric in metrics:
        all_correlations = [
            per_image_summary[image_stem][
                "metrics"
            ][metric]["spearman_rho"]
            for image_stem in per_image_summary
        ]

        all_monotonicity = [
            per_image_summary[image_stem][
                "metrics"
            ][metric]["monotonicity_score"]
            for image_stem in per_image_summary
        ]

        all_slopes = [
            per_image_summary[image_stem][
                "metrics"
            ][metric]["slope"]
            for image_stem in per_image_summary
        ]

        aggregate_summary[metric] = {
            "mean_spearman_rho": float(
                np.nanmean(
                    np.asarray(
                        all_correlations,
                        dtype=np.float64,
                    )
                )
            ),
            "mean_monotonicity_score": float(
                np.nanmean(
                    np.asarray(
                        all_monotonicity,
                        dtype=np.float64,
                    )
                )
            ),
            "mean_slope": float(
                np.nanmean(
                    np.asarray(
                        all_slopes,
                        dtype=np.float64,
                    )
                )
            ),
        }

    return {
        "method": (
            "VSDI_v2_1_controlled_scale_sensitivity_validation"
        ),
        "interpolation": interpolation_name,
        "number_of_images": len(grouped),
        "number_of_scale_conditions": len(
            sorted(
                {
                    float(row["resize_factor"])
                    for row in rows
                }
            )
        ),
        "configuration": {
            **asdict(config),
            "sigmas": [
                float(value)
                for value in config.sigmas
            ],
        },
        "per_image": per_image_summary,
        "aggregate": aggregate_summary,
        "interpretation_thresholds": {
            "strong_monotonicity_score": 0.80,
            "strong_spearman_absolute_value": 0.80,
            "minimum_positive_slope_for_scale_metrics": 0.0,
        },
    }


def save_per_image_scale_response_plot(
    path: Path,
    image_rows: Sequence[
        Mapping[str, object]
    ],
) -> None:
    scales = np.asarray(
        [
            float(row["resize_factor"])
            for row in image_rows
        ],
        dtype=np.float64,
    )

    median = np.asarray(
        [
            float(row["weighted_median_sigma"])
            for row in image_rows
        ],
        dtype=np.float64,
    )

    mean = np.asarray(
        [
            float(row["weighted_mean_sigma"])
            for row in image_rows
        ],
        dtype=np.float64,
    )

    expected = np.asarray(
        [
            float(row["profile_expected_sigma"])
            for row in image_rows
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        scales,
        median,
        marker="o",
        linewidth=2.0,
        label="Weighted median sigma",
    )

    axis.plot(
        scales,
        mean,
        marker="s",
        linewidth=2.0,
        label="Weighted mean sigma",
    )

    axis.plot(
        scales,
        expected,
        marker="^",
        linewidth=2.0,
        label="Profile expected sigma",
    )

    axis.set_xlabel(
        "Applied image resize factor"
    )
    axis.set_ylabel(
        "Estimated vessel scale"
    )
    axis.set_title(
        f"Scale Sensitivity — {image_rows[0]['image_name']}"
    )
    axis.grid(
        alpha=0.25
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def save_per_image_profile_evolution_plot(
    path: Path,
    image_rows: Sequence[
        Mapping[str, object]
    ],
    sigmas: Sequence[float],
) -> None:
    positions = np.arange(
        len(sigmas)
    )

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    for row in image_rows:
        profile = [
            float(
                row[
                    f"sigma_{float(sigma):g}"
                ]
            )
            for sigma in sigmas
        ]

        axis.plot(
            positions,
            profile,
            marker="o",
            linewidth=1.8,
            label=(
                f"scale={float(row['resize_factor']):.2f}"
            ),
        )

    axis.set_xticks(
        positions
    )
    axis.set_xticklabels(
        [
            str(value)
            for value in sigmas
        ]
    )
    axis.set_xlabel(
        "Dominant sigma"
    )
    axis.set_ylabel(
        "Normalized profile weight"
    )
    axis.set_title(
        f"Vessel-Scale Profile Evolution — "
        f"{image_rows[0]['image_name']}"
    )
    axis.grid(
        alpha=0.25
    )
    axis.legend(
        ncol=2
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def save_aggregate_metric_plot(
    path: Path,
    grouped_rows: Mapping[
        str,
        Sequence[Mapping[str, object]]
    ],
    metric_name: str,
    y_label: str,
    title: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for image_stem, image_rows in grouped_rows.items():
        scales = [
            float(row["resize_factor"])
            for row in image_rows
        ]

        values = [
            float(row[metric_name])
            for row in image_rows
        ]

        axis.plot(
            scales,
            values,
            marker="o",
            linewidth=1.8,
            label=image_stem,
        )

    axis.set_xlabel(
        "Applied image resize factor"
    )
    axis.set_ylabel(
        y_label
    )
    axis.set_title(
        title
    )
    axis.grid(
        alpha=0.25
    )
    axis.legend(
        title="DRIVE image"
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def save_aggregate_profile_heatmap(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    sigmas: Sequence[float],
) -> None:
    scale_values = sorted(
        {
            float(row["resize_factor"])
            for row in rows
        }
    )

    matrix: List[List[float]] = []

    for scale_factor in scale_values:
        matching_rows = [
            row
            for row in rows
            if math.isclose(
                float(row["resize_factor"]),
                scale_factor,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]

        profile_matrix = np.asarray(
            [
                [
                    float(
                        row[
                            f"sigma_{float(sigma):g}"
                        ]
                    )
                    for sigma in sigmas
                ]
                for row in matching_rows
            ],
            dtype=np.float64,
        )

        mean_profile = np.mean(
            profile_matrix,
            axis=0,
        )

        mean_profile = normalize_profile(
            mean_profile
        )

        matrix.append(
            mean_profile.tolist()
        )

    heatmap = np.asarray(
        matrix,
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    image = axis.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
    )

    axis.set_xticks(
        np.arange(
            len(sigmas)
        )
    )
    axis.set_xticklabels(
        [
            str(value)
            for value in sigmas
        ]
    )
    axis.set_yticks(
        np.arange(
            len(scale_values)
        )
    )
    axis.set_yticklabels(
        [
            f"{value:.2f}"
            for value in scale_values
        ]
    )
    axis.set_xlabel(
        "Dominant sigma"
    )
    axis.set_ylabel(
        "Applied image resize factor"
    )
    axis.set_title(
        "Aggregate Vessel-Scale Profile Shift"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )
    colorbar.set_label(
        "Mean normalized profile weight"
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def print_summary(
    summary: Mapping[str, object],
    output_dir: Path,
) -> None:
    aggregate = summary["aggregate"]

    print(
        "\n" + "=" * 78
    )
    print(
        "VESSEL-SCALE SENSITIVITY VALIDATION COMPLETE"
    )
    print(
        "=" * 78
    )

    for metric_name in (
        "weighted_median_sigma",
        "weighted_mean_sigma",
        "profile_expected_sigma",
        "profile_center_of_mass_index",
        "fine_scale_weight_fraction",
    ):
        metric_summary = aggregate[
            metric_name
        ]

        print(
            f"{metric_name}:"
        )
        print(
            "  Mean Spearman rho:       "
            f"{metric_summary['mean_spearman_rho']:.4f}"
        )
        print(
            "  Mean monotonicity score: "
            f"{metric_summary['mean_monotonicity_score']:.4f}"
        )
        print(
            "  Mean slope:              "
            f"{metric_summary['mean_slope']:.4f}"
        )
        print(
            "-" * 78
        )

    print(
        "Outputs saved to:"
    )
    print(
        f"  {output_dir.resolve()}"
    )
    print(
        "=" * 78
    )


def main() -> None:
    args = parse_args()

    if not args.image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory does not exist: "
            f"{args.image_dir.resolve()}"
        )

    image_paths = [
        resolve_image_path(
            args.image_dir,
            image_name,
        )
        for image_name in args.image_names
    ]

    config = EstimatorConfig(
        sigmas=tuple(
            args.sigmas
        ),
        response_quantile=float(
            args.response_quantile
        ),
        minimum_valid_pixels=int(
            args.minimum_valid_pixels
        ),
        minimum_component_area=int(
            args.minimum_component_area
        ),
        fov_border_exclusion_fraction=float(
            args.fov_border_exclusion_fraction
        ),
        clahe_clip_limit=float(
            args.clahe_clip_limit
        ),
        clahe_grid_size=int(
            args.clahe_grid_size
        ),
        fine_scale_max_sigma=float(
            args.fine_scale_max_sigma
        ),
        bootstrap_iterations=int(
            args.bootstrap_iterations
        ),
        bootstrap_sample_fraction=float(
            args.bootstrap_sample_fraction
        ),
        bootstrap_seed=int(
            args.bootstrap_seed
        ),
    )

    output_dir: Path = args.output_dir
    resized_root = (
        output_dir / "resized_images"
    )
    estimator_root = (
        output_dir / "estimator_outputs"
    )
    per_image_plot_dir = (
        output_dir / "per_image"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    resized_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    estimator_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    per_image_plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nStage B.5 — Vessel-Scale Sensitivity Validation"
    )
    print(
        "=" * 78
    )
    print(
        f"Images:          "
        f"{', '.join(path.name for path in image_paths)}"
    )
    print(
        "Resize factors:  "
        + ", ".join(
            f"{value:.2f}"
            for value in args.resize_factors
        )
    )
    print(
        f"Output:          "
        f"{output_dir.resolve()}"
    )
    print(
        "=" * 78
    )

    rows: List[Dict[str, object]] = []

    total_runs = (
        len(image_paths)
        * len(args.resize_factors)
    )
    run_index = 0

    for image_path in image_paths:
        image = load_image(
            image_path
        )

        original_shape = image.shape[:2]

        print(
            f"\nImage: {image_path.name} "
            f"({original_shape[1]}×{original_shape[0]})"
        )

        for scale_factor in args.resize_factors:
            run_index += 1

            resized_image = resize_image(
                image,
                scale_factor,
                args.interpolation,
            )

            resized_shape = (
                resized_image.shape[:2]
            )

            tag = scale_tag(
                scale_factor
            )

            resized_path = (
                resized_root
                / image_path.stem
                / (
                    f"{image_path.stem}"
                    f"_scale_{tag}.png"
                )
            )

            estimator_output = (
                estimator_root
                / image_path.stem
                / f"scale_{tag}"
            )

            diagnostics_path = (
                estimator_output
                / "diagnostics.yaml"
            )

            save_resized_image(
                resized_image,
                resized_path,
            )

            print(
                f"  [{run_index:02d}/{total_runs:02d}] "
                f"scale={scale_factor:.2f}, "
                f"size={resized_shape[1]}×{resized_shape[0]}"
            )

            result = None

            if not args.overwrite:
                result = load_cached_result(
                    diagnostics_path
                )

                if result is not None:
                    print(
                        "    Using cached diagnostics."
                    )

            if result is None:
                result = estimate_vessel_scale(
                    image_path=resized_path,
                    output_dir=estimator_output,
                    config=config,
                    save_arrays=bool(
                        args.save_arrays
                    ),
                )

            row = build_row(
                image_path=image_path,
                resized_path=resized_path,
                original_shape=original_shape,
                resized_shape=resized_shape,
                resize_factor=scale_factor,
                result=result,
                sigmas=config.sigmas,
            )

            rows.append(
                row
            )

            print(
                "    "
                f"status={result.status}, "
                f"median={result.weighted_median:.3f}, "
                f"mean={result.weighted_mean:.3f}, "
                f"expected={row['profile_expected_sigma']:.3f}, "
                f"fine={result.fine_scale.vesselness_weight_fraction:.3f}"
            )

    if not rows:
        raise RuntimeError(
            "No sensitivity-validation results were generated."
        )

    write_csv(
        output_dir
        / "scale_sensitivity_results.csv",
        rows,
        config.sigmas,
    )

    summary = build_summary(
        rows,
        config,
        args.interpolation,
    )

    (
        output_dir
        / "scale_sensitivity_summary.yaml"
    ).write_text(
        yaml.safe_dump(
            summary,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    grouped_rows = group_rows_by_image(
        rows
    )

    for image_stem, image_rows in grouped_rows.items():
        save_per_image_scale_response_plot(
            per_image_plot_dir
            / f"{image_stem}_scale_response.png",
            image_rows,
        )

        save_per_image_profile_evolution_plot(
            per_image_plot_dir
            / f"{image_stem}_profile_evolution.png",
            image_rows,
            config.sigmas,
        )

    save_aggregate_metric_plot(
        output_dir
        / "aggregate_median_vs_scale.png",
        grouped_rows,
        metric_name="weighted_median_sigma",
        y_label="Weighted median sigma",
        title=(
            "Controlled Resize Factor vs. "
            "Estimated Median Vessel Scale"
        ),
    )

    save_aggregate_metric_plot(
        output_dir
        / "aggregate_mean_vs_scale.png",
        grouped_rows,
        metric_name="weighted_mean_sigma",
        y_label="Weighted mean sigma",
        title=(
            "Controlled Resize Factor vs. "
            "Estimated Mean Vessel Scale"
        ),
    )

    save_aggregate_metric_plot(
        output_dir
        / "aggregate_expected_sigma_vs_scale.png",
        grouped_rows,
        metric_name="profile_expected_sigma",
        y_label="Profile expected sigma",
        title=(
            "Controlled Resize Factor vs. "
            "Profile Expected Vessel Scale"
        ),
    )

    save_aggregate_metric_plot(
        output_dir
        / "aggregate_fine_fraction_vs_scale.png",
        grouped_rows,
        metric_name="fine_scale_weight_fraction",
        y_label="Fine-scale weight fraction",
        title=(
            "Controlled Resize Factor vs. "
            "Fine-Scale Profile Fraction"
        ),
    )

    save_aggregate_profile_heatmap(
        output_dir
        / "aggregate_profile_heatmap.png",
        rows,
        config.sigmas,
    )

    print_summary(
        summary,
        output_dir,
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nValidation interrupted by user."
        )
        sys.exit(130)