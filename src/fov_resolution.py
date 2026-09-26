"""
D2-AR: label-free adaptive resolution (FOV) normalization.

This module factors out the *already-validated* D2-AR logic that previously
lived only in src/evaluate_adaptive_resolution.py and src/evaluate_resolution.py,
so it can be reused, byte-for-byte identical in its actual computation, by the
matched-protocol reconstruction evaluators (src/evaluate_baseline_v2.py and
src/evaluate_topology_v2.py).

Nothing here changes the D2-AR algorithm itself: the FOV-mask estimation, the
equivalent-diameter formula, the scale-clamping rule, and the resize/restore
interpolation choices are copied verbatim from the version that was already
run and validated (see experiments/d2_s_stable_fp32_seed42/
adaptive_resolution_normalization_results.yaml and
experiments/d2ar_matched_baseline_seed42/adaptive_resolution_normalization_results.yaml).

The only thing that changes is *where* it plugs in: instead of writing
resized images/masks to a disk cache and calling a separate reconstruction
pipeline (src.evaluate.reconstruct_probability_maps), this module exposes
plain in-memory functions that the matched-protocol per-image loops call
directly, so every metric downstream is computed by the exact same,
already-reported metrics code.

Default FOV/resizing configuration values match the ones used in every prior
D2-AR run in this project (see config.yaml:
evaluation.adaptive_resolution_normalization).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_FOV_CONFIG: Dict[str, Any] = {
    "grayscale_threshold": 10,
    "closing_kernel_size": 15,
    "minimum_component_fraction": 0.20,
}

DEFAULT_RESIZING_CONFIG: Dict[str, Any] = {
    "minimum_scale_factor": 0.20,
    "maximum_scale_factor": 2.00,
    "minimum_short_side": 384,
    "maximum_short_side": 1024,
}

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".bmp"}


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def list_images(directory: Path) -> List[Path]:
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def estimate_fov_mask(
    image_rgb: np.ndarray,
    grayscale_threshold: int = 10,
    closing_kernel_size: int = 15,
    minimum_component_fraction: float = 0.20,
) -> np.ndarray:
    """
    Estimate the retinal field from RGB content only.

    A pixel is considered retinal when its maximum RGB channel exceeds the
    configured dark-background threshold. The largest connected component is
    retained after morphological closing.

    Copied verbatim from src/evaluate_adaptive_resolution.py.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {image_rgb.shape}")
    if not 0 <= grayscale_threshold <= 255:
        raise ValueError("grayscale_threshold must lie in [0, 255].")
    if closing_kernel_size < 1:
        raise ValueError("closing_kernel_size must be positive.")
    if closing_kernel_size % 2 == 0:
        closing_kernel_size += 1
    if not 0.0 < minimum_component_fraction <= 1.0:
        raise ValueError("minimum_component_fraction must lie in (0, 1].")

    intensity = image_rgb.max(axis=2)
    candidate = (intensity > grayscale_threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    number, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if number <= 1:
        raise RuntimeError("No retinal FOV component was detected.")

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas)) + 1
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    image_area = int(image_rgb.shape[0] * image_rgb.shape[1])

    if largest_area / max(image_area, 1) < minimum_component_fraction:
        raise RuntimeError(
            f"Detected retinal component is implausibly small: {largest_area / max(image_area, 1):.4f}"
        )

    return (labels == largest_label).astype(np.uint8)


def equivalent_diameter(binary_mask: np.ndarray) -> float:
    """Copied verbatim from src/evaluate_adaptive_resolution.py."""
    area = float((binary_mask > 0).sum())
    if area <= 0.0:
        raise ValueError("Cannot compute equivalent diameter from an empty mask.")
    return float(np.sqrt((4.0 * area) / np.pi))


def estimate_fov_diameter_from_array(
    image_rgb: np.ndarray,
    fov_config: Mapping[str, Any] = DEFAULT_FOV_CONFIG,
) -> Tuple[float, float]:
    """Array-based counterpart of evaluate_adaptive_resolution.estimate_fov_diameter
    (which reads from a path); logic is otherwise identical."""
    fov_mask = estimate_fov_mask(
        image_rgb=image_rgb,
        grayscale_threshold=int(fov_config.get("grayscale_threshold", 10)),
        closing_kernel_size=int(fov_config.get("closing_kernel_size", 15)),
        minimum_component_fraction=float(fov_config.get("minimum_component_fraction", 0.20)),
    )
    diameter = equivalent_diameter(fov_mask)
    occupancy = float(fov_mask.mean())
    return diameter, occupancy


def calculate_source_reference_diameter(
    source_images: Sequence[Path],
    fov_config: Mapping[str, Any] = DEFAULT_FOV_CONFIG,
    verbose: bool = True,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Copied verbatim (logic) from evaluate_adaptive_resolution.calculate_source_reference."""
    records: List[Dict[str, Any]] = []
    diameters: List[float] = []

    if verbose:
        print("\nSource-domain FOV calibration (D2-AR)")
        print("-" * 72)

    for path in source_images:
        image = read_rgb(path)
        diameter, occupancy = estimate_fov_diameter_from_array(image, fov_config)
        diameters.append(diameter)
        records.append({
            "image": str(Path(path).resolve()),
            "fov_equivalent_diameter": float(diameter),
            "fov_occupancy": float(occupancy),
        })

    values = np.asarray(diameters, dtype=np.float64)
    reference = float(np.median(values))

    if verbose:
        print(f"Source images: {len(source_images)}")
        print(
            f"FOV equivalent diameter: min={values.min():.2f}, "
            f"median={np.median(values):.2f}, mean={values.mean():.2f}, max={values.max():.2f}"
        )
        print(f"Reference diameter: {reference:.2f} pixels")
        print("-" * 72)

    return reference, records


def clamp_adaptive_shape(
    original_height: int,
    original_width: int,
    raw_scale: float,
    resizing_config: Mapping[str, Any] = DEFAULT_RESIZING_CONFIG,
) -> Tuple[int, int, float, int, bool]:
    """Copied verbatim from src/evaluate_adaptive_resolution.py."""
    min_scale = float(resizing_config.get("minimum_scale_factor", 0.20))
    max_scale = float(resizing_config.get("maximum_scale_factor", 2.00))
    min_short_side = int(resizing_config.get("minimum_short_side", 384))
    max_short_side = int(resizing_config.get("maximum_short_side", 1024))

    if not 0.0 < min_scale <= max_scale:
        raise ValueError("Invalid adaptive scale-factor limits.")
    if not 1 <= min_short_side <= max_short_side:
        raise ValueError("Invalid adaptive short-side limits.")

    scale = float(np.clip(raw_scale, min_scale, max_scale))
    original_short_side = min(original_height, original_width)

    proposed_short_side = int(round(original_short_side * scale))
    final_short_side = int(np.clip(proposed_short_side, min_short_side, max_short_side))
    final_scale = float(final_short_side) / float(original_short_side)

    new_height = max(1, int(round(original_height * final_scale)))
    new_width = max(1, int(round(original_width * final_scale)))

    was_clamped = not np.isclose(final_scale, raw_scale, rtol=0.0, atol=1e-8)
    return new_height, new_width, final_scale, final_short_side, was_clamped


def compute_adaptive_target_shape(
    image_rgb: np.ndarray,
    source_reference_diameter: float,
    fov_config: Mapping[str, Any] = DEFAULT_FOV_CONFIG,
    resizing_config: Mapping[str, Any] = DEFAULT_RESIZING_CONFIG,
) -> Dict[str, Any]:
    """
    Given one target image, estimate its FOV diameter and compute the
    resize target that would match it to source_reference_diameter, under
    the same clamping rule used by every prior D2-AR run.
    """
    original_height, original_width = image_rgb.shape[:2]
    target_diameter, occupancy = estimate_fov_diameter_from_array(image_rgb, fov_config)
    raw_scale = source_reference_diameter / target_diameter

    new_height, new_width, final_scale, final_short_side, was_clamped = clamp_adaptive_shape(
        original_height=original_height,
        original_width=original_width,
        raw_scale=raw_scale,
        resizing_config=resizing_config,
    )

    return {
        "original_height": int(original_height),
        "original_width": int(original_width),
        "target_fov_equivalent_diameter": float(target_diameter),
        "target_fov_occupancy": float(occupancy),
        "source_reference_diameter": float(source_reference_diameter),
        "raw_scale_factor": float(raw_scale),
        "final_scale_factor": float(final_scale),
        "final_short_side": int(final_short_side),
        "new_height": int(new_height),
        "new_width": int(new_width),
        "scale_was_clamped": bool(was_clamped),
    }


def resize_image_for_model(image_rgb: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """Copied verbatim (interpolation rule) from src/evaluate_resolution.py:resize_image."""
    old_height, old_width = image_rgb.shape[:2]
    interpolation = (
        cv2.INTER_AREA
        if new_height < old_height or new_width < old_width
        else cv2.INTER_CUBIC
    )
    return cv2.resize(image_rgb, (new_width, new_height), interpolation=interpolation)


def resize_prediction_to_native(
    probability: np.ndarray,
    native_height: int,
    native_width: int,
) -> np.ndarray:
    """Copied verbatim (interpolation + clipping rule) from
    src/evaluate_resolution.py:restore_native_resolution."""
    restored = cv2.resize(
        probability.astype(np.float32),
        (native_width, native_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    return np.clip(restored, 0.0, 1.0)
