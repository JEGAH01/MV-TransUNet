#!/usr/bin/env python
"""
VSDI Version 2.1
================

Label-free vessel-scale distribution estimation for retinal fundus images.

Version 2.1 design
------------------
- Restores Version 1 preprocessing (no pre-Frangi edge taper).
- Uses stronger post-Frangi FOV boundary exclusion.
- Preserves the full normalized scale-profile vector.
- Reports separate fine- and coarse-scale statistics.
- Uses the actual bootstrap weighted-median samples.
- Defines confidence from support, bootstrap stability, and boundary cleanliness.
- Keeps entropy and histogram concentration as descriptive outputs only.

Example
-------
python -m src.vessel_scale_estimator ^
    --image "datasets/DRIVE/training/image/21.png" ^
    --output-dir "cache/vessel_scale/drive_21_v21" ^
    --save-arrays
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from skimage.filters import frangi
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)


DEFAULT_SIGMAS: Tuple[float, ...] = (
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
    5.00,
)


@dataclass(frozen=True)
class EstimatorConfig:
    sigmas: Tuple[float, ...] = DEFAULT_SIGMAS
    response_quantile: float = 0.90
    minimum_valid_pixels: int = 500
    minimum_component_area: int = 8

    # Stronger post-Frangi boundary exclusion than V1.
    fov_border_exclusion_fraction: float = 0.06

    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    frangi_alpha: float = 0.5
    frangi_beta: float = 0.5
    frangi_gamma: Optional[float] = None

    fine_scale_max_sigma: float = 2.50

    bootstrap_iterations: int = 200
    bootstrap_sample_fraction: float = 0.70
    bootstrap_seed: int = 42

    minimum_fov_fraction: float = 0.15
    maximum_fov_fraction: float = 0.98


@dataclass
class ScaleGroupStatistics:
    name: str
    minimum_sigma: float
    maximum_sigma: float
    pixel_count: int
    vesselness_weight_fraction: float
    weighted_mean: float
    weighted_median: float
    weighted_q25: float
    weighted_q75: float
    weighted_iqr: float


@dataclass
class VesselScaleResult:
    image_path: str
    image_height: int
    image_width: int

    fov_pixels: int
    valid_vessel_pixels: int
    valid_vessel_fraction_of_fov: float
    response_threshold: float

    weighted_mean: float
    weighted_median: float
    weighted_q25: float
    weighted_q75: float
    weighted_std: float
    weighted_iqr: float

    normalized_entropy: float
    histogram_concentration: float
    response_strength: float

    boundary_response_fraction: float
    support_score: float
    bootstrap_stability_score: float
    boundary_cleanliness_score: float
    confidence: float

    bootstrap_median_mean: float
    bootstrap_median_std: float
    bootstrap_median_q025: float
    bootstrap_median_q975: float

    fine_scale: ScaleGroupStatistics
    coarse_scale: ScaleGroupStatistics
    scale_profile: Dict[str, float]

    status: str


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
            "At least one sigma value must be provided."
        )

    return tuple(sorted(set(values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a label-free retinal vessel-scale distribution."
        )
    )

    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        "--save-arrays",
        action="store_true",
    )

    return parser.parse_args()


def validate_config(config: EstimatorConfig) -> None:
    if not 0.0 < config.response_quantile < 1.0:
        raise ValueError(
            "response_quantile must be strictly between 0 and 1."
        )

    if config.minimum_valid_pixels < 1:
        raise ValueError(
            "minimum_valid_pixels must be at least 1."
        )

    if config.minimum_component_area < 1:
        raise ValueError(
            "minimum_component_area must be at least 1."
        )

    if not 0.0 <= config.fov_border_exclusion_fraction < 0.5:
        raise ValueError(
            "fov_border_exclusion_fraction must be in [0, 0.5)."
        )

    if config.clahe_clip_limit <= 0:
        raise ValueError(
            "clahe_clip_limit must be greater than zero."
        )

    if config.clahe_grid_size < 2:
        raise ValueError(
            "clahe_grid_size must be at least 2."
        )

    if len(config.sigmas) < 2:
        raise ValueError(
            "At least two sigma values are required."
        )

    if not (
        min(config.sigmas)
        < config.fine_scale_max_sigma
        < max(config.sigmas)
    ):
        raise ValueError(
            "fine_scale_max_sigma must lie inside the configured sigma range."
        )

    if config.bootstrap_iterations < 20:
        raise ValueError(
            "bootstrap_iterations must be at least 20."
        )

    if not 0.05 <= config.bootstrap_sample_fraction <= 1.0:
        raise ValueError(
            "bootstrap_sample_fraction must be in [0.05, 1.0]."
        )


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Input image does not exist: {path.resolve()}"
        )

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(
            f"OpenCV could not read the image: {path.resolve()}"
        )

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labelled = label(
        mask.astype(bool),
        connectivity=2,
    )
    regions = regionprops(labelled)

    if not regions:
        return np.zeros_like(mask, dtype=bool)

    largest = max(
        regions,
        key=lambda region: region.area,
    )
    return labelled == largest.label


def create_fov_mask(
    rgb: np.ndarray,
    config: EstimatorConfig,
) -> np.ndarray:
    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=5.0,
        sigmaY=5.0,
    )

    nonzero = blurred[blurred > 0]
    if nonzero.size == 0:
        raise ValueError(
            "The image contains no nonzero grayscale pixels."
        )

    low = float(np.percentile(nonzero, 5.0))
    high = float(np.percentile(nonzero, 95.0))
    threshold = low + 0.08 * (high - low)

    raw_mask = blurred > threshold

    radius = max(
        3,
        int(round(min(rgb.shape[:2]) * 0.01)),
    )

    cleaned = binary_closing(
        raw_mask,
        footprint=disk(radius),
    )
    cleaned = binary_opening(
        cleaned,
        footprint=disk(max(2, radius // 2)),
    )
    cleaned = remove_small_objects(
        cleaned,
        min_size=max(
            64,
            int(rgb.shape[0] * rgb.shape[1] * 0.01),
        ),
    )
    cleaned = largest_connected_component(cleaned)

    mask_uint8 = cleaned.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filled = np.zeros_like(mask_uint8)

    if contours:
        contour = max(
            contours,
            key=cv2.contourArea,
        )
        cv2.drawContours(
            filled,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

    fov = filled > 0
    fov_fraction = float(fov.mean())

    if not (
        config.minimum_fov_fraction
        <= fov_fraction
        <= config.maximum_fov_fraction
    ):
        raise ValueError(
            "Estimated FOV occupies an implausible image fraction: "
            f"{fov_fraction:.4f}."
        )

    return fov


def compute_fov_distance(
    fov_mask: np.ndarray,
) -> np.ndarray:
    return cv2.distanceTransform(
        fov_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    ).astype(np.float32)


def create_valid_fov(
    fov_mask: np.ndarray,
    distance_map: np.ndarray,
    exclusion_fraction: float,
) -> np.ndarray:
    if exclusion_fraction <= 0:
        return fov_mask.copy()

    maximum_distance = float(distance_map.max())
    if maximum_distance <= 0:
        return fov_mask.copy()

    threshold = max(
        1.0,
        exclusion_fraction * maximum_distance,
    )

    interior = distance_map >= threshold

    if interior.sum() < 0.5 * fov_mask.sum():
        return fov_mask.copy()

    return interior


def preprocess_green_channel(
    rgb: np.ndarray,
    fov_mask: np.ndarray,
    config: EstimatorConfig,
) -> np.ndarray:
    """
    Version 1 preprocessing restored exactly in principle:
    CLAHE green channel, inversion, robust normalization, outside-FOV zeroing.
    """
    green = rgb[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=float(config.clahe_clip_limit),
        tileGridSize=(
            int(config.clahe_grid_size),
            int(config.clahe_grid_size),
        ),
    )
    enhanced = clahe.apply(green)

    normalized = enhanced.astype(np.float32) / 255.0
    vessel_bright = 1.0 - normalized

    values = vessel_bright[fov_mask]
    lower = float(np.percentile(values, 1.0))
    upper = float(np.percentile(values, 99.0))

    if upper <= lower:
        raise ValueError(
            "Insufficient green-channel contrast inside the FOV."
        )

    vessel_bright = np.clip(
        (vessel_bright - lower) / (upper - lower),
        0.0,
        1.0,
    )

    vessel_bright[~fov_mask] = 0.0
    return vessel_bright.astype(np.float32)


def compute_multiscale_vesselness(
    preprocessed: np.ndarray,
    config: EstimatorConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    responses: List[np.ndarray] = []

    for sigma in config.sigmas:
        response = frangi(
            preprocessed,
            sigmas=(float(sigma),),
            alpha=float(config.frangi_alpha),
            beta=float(config.frangi_beta),
            gamma=config.frangi_gamma,
            black_ridges=False,
            mode="reflect",
        )

        response = np.nan_to_num(
            response,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        responses.append(response)

    response_stack = np.stack(responses, axis=0)
    winning_index = np.argmax(response_stack, axis=0)
    maximum_response = np.max(response_stack, axis=0)

    sigma_array = np.asarray(
        config.sigmas,
        dtype=np.float32,
    )
    dominant_sigma = sigma_array[winning_index]

    return (
        response_stack,
        maximum_response,
        dominant_sigma,
    )


def create_candidate_mask(
    maximum_response: np.ndarray,
    valid_fov: np.ndarray,
    config: EstimatorConfig,
) -> Tuple[np.ndarray, float]:
    responses = maximum_response[valid_fov]
    positive = responses[responses > 0]

    if positive.size == 0:
        return np.zeros_like(valid_fov), 0.0

    threshold = float(
        np.quantile(
            positive,
            config.response_quantile,
        )
    )

    candidate = (
        valid_fov
        & np.isfinite(maximum_response)
        & (maximum_response >= threshold)
    )

    candidate = remove_small_objects(
        candidate,
        min_size=config.minimum_component_area,
        connectivity=2,
    )

    return candidate.astype(bool), threshold


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError(
            "values and weights must be one-dimensional."
        )

    if len(values) != len(weights):
        raise ValueError(
            "values and weights must have equal length."
        )

    finite = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    values = values[finite]
    weights = weights[finite]

    if values.size == 0:
        return float("nan")

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cutoff = quantile * cumulative[-1]

    index = int(
        np.searchsorted(
            cumulative,
            cutoff,
            side="left",
        )
    )
    index = min(index, len(values) - 1)

    return float(values[index])


def weighted_statistics(
    values: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    finite = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    values = values[finite].astype(np.float64)
    weights = weights[finite].astype(np.float64)

    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "std": float("nan"),
            "iqr": float("nan"),
        }

    total_weight = float(weights.sum())
    mean = float(np.sum(values * weights) / total_weight)
    variance = float(
        np.sum(weights * np.square(values - mean))
        / total_weight
    )

    q25 = weighted_quantile(values, weights, 0.25)
    median = weighted_quantile(values, weights, 0.50)
    q75 = weighted_quantile(values, weights, 0.75)

    return {
        "mean": mean,
        "median": median,
        "q25": q25,
        "q75": q75,
        "std": math.sqrt(max(variance, 0.0)),
        "iqr": float(q75 - q25),
    }


def compute_scale_profile(
    values: np.ndarray,
    weights: np.ndarray,
    sigmas: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    raw = np.asarray(
        [
            float(
                weights[
                    np.isclose(values, sigma)
                ].sum()
            )
            for sigma in sigmas
        ],
        dtype=np.float64,
    )

    total = float(raw.sum())

    normalized = (
        raw / total
        if total > 0
        else np.zeros_like(raw)
    )

    profile = {
        f"sigma_{float(sigma):g}": float(weight)
        for sigma, weight in zip(sigmas, normalized)
    }

    return normalized, profile


def normalized_entropy(
    profile: np.ndarray,
) -> float:
    positive = profile[profile > 0]

    if positive.size == 0:
        return 1.0

    entropy = -float(
        np.sum(positive * np.log(positive))
    )
    maximum = math.log(len(profile))

    if maximum <= 0:
        return 0.0

    return float(
        np.clip(entropy / maximum, 0.0, 1.0)
    )


def histogram_concentration(
    profile: np.ndarray,
) -> float:
    if profile.size == 0:
        return 0.0

    raw = float(np.sum(np.square(profile)))
    minimum = 1.0 / len(profile)

    return float(
        np.clip(
            (raw - minimum)
            / max(1.0 - minimum, 1e-12),
            0.0,
            1.0,
        )
    )


def make_scale_group(
    name: str,
    values: np.ndarray,
    weights: np.ndarray,
    selected: np.ndarray,
    total_weight: float,
) -> ScaleGroupStatistics:
    group_values = values[selected]
    group_weights = weights[selected]
    stats = weighted_statistics(
        group_values,
        group_weights,
    )

    if group_values.size > 0:
        minimum_sigma = float(np.min(group_values))
        maximum_sigma = float(np.max(group_values))
    else:
        minimum_sigma = float("nan")
        maximum_sigma = float("nan")

    group_weight = float(group_weights.sum())
    weight_fraction = (
        group_weight / total_weight
        if total_weight > 0
        else 0.0
    )

    return ScaleGroupStatistics(
        name=name,
        minimum_sigma=minimum_sigma,
        maximum_sigma=maximum_sigma,
        pixel_count=int(group_values.size),
        vesselness_weight_fraction=float(weight_fraction),
        weighted_mean=float(stats["mean"]),
        weighted_median=float(stats["median"]),
        weighted_q25=float(stats["q25"]),
        weighted_q75=float(stats["q75"]),
        weighted_iqr=float(stats["iqr"]),
    )


def bootstrap_weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
    config: EstimatorConfig,
) -> np.ndarray:
    rng = np.random.default_rng(
        config.bootstrap_seed
    )

    count = len(values)
    if count == 0:
        return np.asarray([], dtype=np.float64)

    sample_size = max(
        1,
        int(round(
            config.bootstrap_sample_fraction * count
        )),
    )

    probabilities = weights.astype(np.float64)
    total = float(probabilities.sum())

    if total > 0:
        probabilities /= total
    else:
        probabilities = None

    samples = np.empty(
        config.bootstrap_iterations,
        dtype=np.float64,
    )

    for index in range(config.bootstrap_iterations):
        selected = rng.choice(
            count,
            size=sample_size,
            replace=True,
            p=probabilities,
        )

        samples[index] = weighted_quantile(
            values[selected],
            weights[selected],
            0.50,
        )

    return samples


def compute_boundary_response_fraction(
    maximum_response: np.ndarray,
    fov_mask: np.ndarray,
    valid_fov: np.ndarray,
) -> float:
    boundary_band = fov_mask & (~valid_fov)

    total_response = float(
        maximum_response[fov_mask].sum()
    )
    boundary_response = float(
        maximum_response[boundary_band].sum()
    )

    if total_response <= 0:
        return 0.0

    return float(
        np.clip(
            boundary_response / total_response,
            0.0,
            1.0,
        )
    )


def empty_scale_group(
    name: str,
) -> ScaleGroupStatistics:
    return ScaleGroupStatistics(
        name=name,
        minimum_sigma=float("nan"),
        maximum_sigma=float("nan"),
        pixel_count=0,
        vesselness_weight_fraction=0.0,
        weighted_mean=float("nan"),
        weighted_median=float("nan"),
        weighted_q25=float("nan"),
        weighted_q75=float("nan"),
        weighted_iqr=float("nan"),
    )


def compute_result(
    image_path: Path,
    image_shape: Tuple[int, int],
    dominant_sigma: np.ndarray,
    maximum_response: np.ndarray,
    candidate_mask: np.ndarray,
    fov_mask: np.ndarray,
    valid_fov: np.ndarray,
    response_threshold: float,
    config: EstimatorConfig,
) -> Tuple[VesselScaleResult, np.ndarray]:
    values = dominant_sigma[
        candidate_mask
    ].astype(np.float64)

    weights = maximum_response[
        candidate_mask
    ].astype(np.float64)

    finite = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    values = values[finite]
    weights = weights[finite]

    fov_pixels = int(valid_fov.sum())
    valid_pixels = int(values.size)
    valid_fraction = (
        valid_pixels / max(fov_pixels, 1)
    )

    boundary_fraction = (
        compute_boundary_response_fraction(
            maximum_response,
            fov_mask,
            valid_fov,
        )
    )

    if valid_pixels == 0:
        result = VesselScaleResult(
            image_path=str(image_path.resolve()),
            image_height=int(image_shape[0]),
            image_width=int(image_shape[1]),
            fov_pixels=fov_pixels,
            valid_vessel_pixels=0,
            valid_vessel_fraction_of_fov=0.0,
            response_threshold=float(response_threshold),
            weighted_mean=float("nan"),
            weighted_median=float("nan"),
            weighted_q25=float("nan"),
            weighted_q75=float("nan"),
            weighted_std=float("nan"),
            weighted_iqr=float("nan"),
            normalized_entropy=1.0,
            histogram_concentration=0.0,
            response_strength=0.0,
            boundary_response_fraction=boundary_fraction,
            support_score=0.0,
            bootstrap_stability_score=0.0,
            boundary_cleanliness_score=0.0,
            confidence=0.0,
            bootstrap_median_mean=float("nan"),
            bootstrap_median_std=float("nan"),
            bootstrap_median_q025=float("nan"),
            bootstrap_median_q975=float("nan"),
            fine_scale=empty_scale_group("fine"),
            coarse_scale=empty_scale_group("coarse"),
            scale_profile={
                f"sigma_{float(sigma):g}": 0.0
                for sigma in config.sigmas
            },
            status="failed_no_valid_vessel_pixels",
        )
        return result, np.asarray([], dtype=np.float64)

    overall = weighted_statistics(values, weights)
    profile_array, profile_dict = compute_scale_profile(
        values,
        weights,
        config.sigmas,
    )

    entropy = normalized_entropy(profile_array)
    concentration = histogram_concentration(
        profile_array
    )

    total_weight = float(weights.sum())

    fine_selected = (
        values <= config.fine_scale_max_sigma
    )
    coarse_selected = ~fine_selected

    fine_scale = make_scale_group(
        "fine",
        values,
        weights,
        fine_selected,
        total_weight,
    )
    coarse_scale = make_scale_group(
        "coarse",
        values,
        weights,
        coarse_selected,
        total_weight,
    )

    bootstrap_samples = bootstrap_weighted_median(
        values,
        weights,
        config,
    )

    bootstrap_mean = float(
        np.mean(bootstrap_samples)
    )
    bootstrap_std = float(
        np.std(
            bootstrap_samples,
            ddof=1,
        )
        if bootstrap_samples.size > 1
        else 0.0
    )
    bootstrap_q025 = float(
        np.quantile(
            bootstrap_samples,
            0.025,
        )
    )
    bootstrap_q975 = float(
        np.quantile(
            bootstrap_samples,
            0.975,
        )
    )

    sigma_range = float(
        max(config.sigmas) - min(config.sigmas)
    )

    bootstrap_stability = float(
        np.clip(
            math.exp(
                -4.0
                * bootstrap_std
                / max(sigma_range, 1e-12)
            ),
            0.0,
            1.0,
        )
    )

    support_score = float(
        np.clip(
            valid_pixels
            / max(config.minimum_valid_pixels, 1),
            0.0,
            1.0,
        )
    )

    boundary_cleanliness = float(
        np.clip(
            1.0 - boundary_fraction,
            0.0,
            1.0,
        )
    )

    # Confidence no longer penalizes natural multi-caliber anatomy.
    confidence = float(
        np.clip(
            (
                support_score
                * bootstrap_stability
                * boundary_cleanliness
            )
            ** (1.0 / 3.0),
            0.0,
            1.0,
        )
    )

    if valid_pixels < config.minimum_valid_pixels:
        status = "low_support"
    elif boundary_fraction > 0.20:
        status = "warning_boundary_response"
    elif bootstrap_stability < 0.50:
        status = "warning_unstable_scale"
    else:
        status = "ok"

    result = VesselScaleResult(
        image_path=str(image_path.resolve()),
        image_height=int(image_shape[0]),
        image_width=int(image_shape[1]),
        fov_pixels=fov_pixels,
        valid_vessel_pixels=valid_pixels,
        valid_vessel_fraction_of_fov=float(
            valid_fraction
        ),
        response_threshold=float(
            response_threshold
        ),
        weighted_mean=float(overall["mean"]),
        weighted_median=float(overall["median"]),
        weighted_q25=float(overall["q25"]),
        weighted_q75=float(overall["q75"]),
        weighted_std=float(overall["std"]),
        weighted_iqr=float(overall["iqr"]),
        normalized_entropy=entropy,
        histogram_concentration=concentration,
        response_strength=float(np.median(weights)),
        boundary_response_fraction=boundary_fraction,
        support_score=support_score,
        bootstrap_stability_score=bootstrap_stability,
        boundary_cleanliness_score=boundary_cleanliness,
        confidence=confidence,
        bootstrap_median_mean=bootstrap_mean,
        bootstrap_median_std=bootstrap_std,
        bootstrap_median_q025=bootstrap_q025,
        bootstrap_median_q975=bootstrap_q975,
        fine_scale=fine_scale,
        coarse_scale=coarse_scale,
        scale_profile=profile_dict,
        status=status,
    )

    return result, bootstrap_samples


def normalize_for_png(
    array: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    valid = np.isfinite(array)

    if mask is not None:
        valid &= mask

    values = array[valid]

    if values.size == 0:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    lower = float(np.percentile(values, 1.0))
    upper = float(np.percentile(values, 99.0))

    if upper <= lower:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    normalized = np.clip(
        (array - lower) / (upper - lower),
        0.0,
        1.0,
    )
    normalized[
        ~np.isfinite(normalized)
    ] = 0.0

    if mask is not None:
        normalized[~mask] = 0.0

    return np.round(
        normalized * 255.0
    ).astype(np.uint8)


def save_diagnostics(
    output_dir: Path,
    rgb: np.ndarray,
    fov_mask: np.ndarray,
    valid_fov: np.ndarray,
    preprocessed: np.ndarray,
    maximum_response: np.ndarray,
    dominant_sigma: np.ndarray,
    candidate_mask: np.ndarray,
    result: VesselScaleResult,
    bootstrap_samples: np.ndarray,
    config: EstimatorConfig,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_dir / "fov_mask.png"),
        fov_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(output_dir / "valid_fov_mask.png"),
        valid_fov.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(output_dir / "preprocessed_green.png"),
        normalize_for_png(
            preprocessed,
            fov_mask,
        ),
    )
    cv2.imwrite(
        str(output_dir / "vesselness.png"),
        normalize_for_png(
            maximum_response,
            valid_fov,
        ),
    )
    cv2.imwrite(
        str(output_dir / "candidate_mask.png"),
        candidate_mask.astype(np.uint8) * 255,
    )

    overlay = rgb.astype(np.float32)
    green_overlay = np.zeros_like(overlay)
    green_overlay[:, :, 1] = 255.0

    alpha = 0.55
    overlay[candidate_mask] = (
        (1.0 - alpha)
        * overlay[candidate_mask]
        + alpha
        * green_overlay[candidate_mask]
    )

    cv2.imwrite(
        str(output_dir / "candidate_overlay.png"),
        cv2.cvtColor(
            np.clip(
                overlay,
                0,
                255,
            ).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        ),
    )

    sigma_visual = np.full_like(
        dominant_sigma,
        np.nan,
        dtype=np.float32,
    )
    sigma_visual[candidate_mask] = (
        dominant_sigma[candidate_mask]
    )

    figure, axis = plt.subplots(figsize=(8, 8))
    artist = axis.imshow(
        sigma_visual,
        cmap="viridis",
        vmin=min(config.sigmas),
        vmax=max(config.sigmas),
    )
    axis.set_title("Dominant Vessel Scale Map")
    axis.axis("off")
    figure.colorbar(
        artist,
        ax=axis,
        label="Dominant sigma",
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "sigma_map.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    profile = np.asarray(
        list(result.scale_profile.values()),
        dtype=np.float64,
    )
    positions = np.arange(len(config.sigmas))

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions, profile)

    median_index = int(
        np.argmin(
            np.abs(
                np.asarray(config.sigmas)
                - result.weighted_median
            )
        )
    )
    axis.axvline(
        median_index,
        linestyle="--",
        label=(
            "Weighted median "
            f"≈ {result.weighted_median:.3f}"
        ),
    )

    fine_boundary_index = max(
        index
        for index, sigma
        in enumerate(config.sigmas)
        if sigma <= config.fine_scale_max_sigma
    )
    axis.axvline(
        fine_boundary_index + 0.5,
        linestyle=":",
        label=(
            "Fine/coarse boundary "
            f"= {config.fine_scale_max_sigma:g}"
        ),
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [str(value) for value in config.sigmas]
    )
    axis.set_xlabel("Dominant sigma")
    axis.set_ylabel(
        "Normalized vesselness weight"
    )
    axis.set_title(
        "Multiscale Vessel Profile"
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "scale_profile.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.bar(
        ["Fine", "Coarse"],
        [
            result.fine_scale.vesselness_weight_fraction,
            result.coarse_scale.vesselness_weight_fraction,
        ],
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(
        "Vesselness weight fraction"
    )
    axis.set_title(
        "Fine vs Coarse Vessel-Scale Composition"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "fine_coarse_profile.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Plot the real bootstrap samples, not a synthetic approximation.
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        bootstrap_samples,
        bins=min(
            20,
            max(
                5,
                len(np.unique(
                    bootstrap_samples
                )),
            ),
        ),
    )
    axis.axvline(
        result.weighted_median,
        linestyle="--",
        label=(
            "Observed weighted median "
            f"= {result.weighted_median:.3f}"
        ),
    )
    axis.axvline(
        result.bootstrap_median_q025,
        linestyle=":",
        label=(
            "Bootstrap 95% interval "
            f"[{result.bootstrap_median_q025:.3f}, "
            f"{result.bootstrap_median_q975:.3f}]"
        ),
    )
    axis.axvline(
        result.bootstrap_median_q975,
        linestyle=":",
    )
    axis.set_xlabel(
        "Bootstrap weighted median sigma"
    )
    axis.set_ylabel("Frequency")
    axis.set_title(
        "Bootstrap Median Stability"
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "bootstrap_stability.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_yaml(
    output_path: Path,
    config: EstimatorConfig,
    result: VesselScaleResult,
) -> None:
    payload: Dict[str, object] = {
        "method": "VSDI_v2_1",
        "description": (
            "Label-free multiscale retinal vessel profile using "
            "Version 1 preprocessing, stronger post-Frangi boundary "
            "exclusion, fine/coarse descriptors, and actual bootstrap "
            "stability analysis."
        ),
        "configuration": {
            **asdict(config),
            "sigmas": [
                float(value)
                for value in config.sigmas
            ],
        },
        "result": asdict(result),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            payload,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def estimate_vessel_scale(
    image_path: Path,
    output_dir: Path,
    config: EstimatorConfig,
    save_arrays: bool = False,
) -> VesselScaleResult:
    validate_config(config)

    rgb = load_rgb_image(image_path)
    fov_mask = create_fov_mask(rgb, config)
    distance_map = compute_fov_distance(fov_mask)

    valid_fov = create_valid_fov(
        fov_mask,
        distance_map,
        config.fov_border_exclusion_fraction,
    )

    preprocessed = preprocess_green_channel(
        rgb,
        fov_mask,
        config,
    )

    (
        response_stack,
        maximum_response,
        dominant_sigma,
    ) = compute_multiscale_vesselness(
        preprocessed,
        config,
    )

    candidate_mask, response_threshold = (
        create_candidate_mask(
            maximum_response,
            valid_fov,
            config,
        )
    )

    result, bootstrap_samples = compute_result(
        image_path=image_path,
        image_shape=rgb.shape[:2],
        dominant_sigma=dominant_sigma,
        maximum_response=maximum_response,
        candidate_mask=candidate_mask,
        fov_mask=fov_mask,
        valid_fov=valid_fov,
        response_threshold=response_threshold,
        config=config,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_diagnostics(
        output_dir=output_dir,
        rgb=rgb,
        fov_mask=fov_mask,
        valid_fov=valid_fov,
        preprocessed=preprocessed,
        maximum_response=maximum_response,
        dominant_sigma=dominant_sigma,
        candidate_mask=candidate_mask,
        result=result,
        bootstrap_samples=bootstrap_samples,
        config=config,
    )

    save_yaml(
        output_dir / "diagnostics.yaml",
        config,
        result,
    )

    if save_arrays:
        np.save(
            output_dir / "response_stack.npy",
            response_stack,
        )
        np.save(
            output_dir / "maximum_vesselness.npy",
            maximum_response,
        )
        np.save(
            output_dir / "dominant_sigma.npy",
            dominant_sigma,
        )
        np.save(
            output_dir / "candidate_mask.npy",
            candidate_mask,
        )
        np.save(
            output_dir / "fov_distance.npy",
            distance_map,
        )
        np.save(
            output_dir / "bootstrap_medians.npy",
            bootstrap_samples,
        )

    return result


def print_result(
    result: VesselScaleResult,
) -> None:
    print(
        "\nVessel-Scale Distribution Estimate — Version 2.1"
    )
    print("=" * 78)
    print(
        f"Status:                         {result.status}"
    )
    print(
        f"Weighted median sigma:          {result.weighted_median:.4f}"
    )
    print(
        f"Weighted mean sigma:            {result.weighted_mean:.4f}"
    )
    print(
        "Weighted Q25-Q75:               "
        f"{result.weighted_q25:.4f}-"
        f"{result.weighted_q75:.4f}"
    )
    print(
        f"Weighted standard deviation:    {result.weighted_std:.4f}"
    )
    print(
        f"Normalized entropy:             {result.normalized_entropy:.4f}"
    )
    print(
        f"Histogram concentration:        {result.histogram_concentration:.4f}"
    )
    print("-" * 78)
    print(
        "Fine-scale median / weight:     "
        f"{result.fine_scale.weighted_median:.4f} / "
        f"{result.fine_scale.vesselness_weight_fraction:.4f}"
    )
    print(
        "Coarse-scale median / weight:   "
        f"{result.coarse_scale.weighted_median:.4f} / "
        f"{result.coarse_scale.vesselness_weight_fraction:.4f}"
    )
    print("-" * 78)
    print(
        f"Bootstrap median std:           {result.bootstrap_median_std:.6f}"
    )
    print(
        "Bootstrap median 95% interval:  "
        f"{result.bootstrap_median_q025:.4f}-"
        f"{result.bootstrap_median_q975:.4f}"
    )
    print(
        f"Boundary response fraction:     {result.boundary_response_fraction:.6f}"
    )
    print(
        f"Valid vessel-like pixels:       {result.valid_vessel_pixels}"
    )
    print(
        "Valid fraction of FOV:          "
        f"{result.valid_vessel_fraction_of_fov:.6f}"
    )
    print("-" * 78)
    print(
        f"Support score:                  {result.support_score:.4f}"
    )
    print(
        f"Bootstrap stability score:      {result.bootstrap_stability_score:.4f}"
    )
    print(
        f"Boundary cleanliness score:     {result.boundary_cleanliness_score:.4f}"
    )
    print(
        f"Overall confidence:             {result.confidence:.4f}"
    )
    print("=" * 78)


def main() -> None:
    args = parse_args()

    config = EstimatorConfig(
        sigmas=tuple(args.sigmas),
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

    result = estimate_vessel_scale(
        image_path=args.image,
        output_dir=args.output_dir,
        config=config,
        save_arrays=bool(args.save_arrays),
    )

    print_result(result)

    print("\nDiagnostics saved to:")
    print(f"  {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()