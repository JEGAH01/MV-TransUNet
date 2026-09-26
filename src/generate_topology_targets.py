"""
Generate topology-aware supervision targets for MV-TransUNet++.

For each binary retinal-vessel mask, this module creates:

1. Cleaned vessel mask
2. One-pixel-wide vessel skeleton
3. Endpoint heatmap
4. Junction heatmap
5. Thin-vessel mask
6. Vessel radius map
7. Vessel width map
8. Per-image topology metadata
9. Per-dataset summary reports

The generated targets support the proposed Typed Neural Spatial Graph
Supervision Module (NSGSM).

Run examples
------------
Process every configured dataset:

    python -m src.generate_topology_targets --config config.yaml

Process only DRIVE_train:

    python -m src.generate_topology_targets \
        --config config.yaml \
        --dataset DRIVE_train

Overwrite existing outputs:

    python -m src.generate_topology_targets \
        --config config.yaml \
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
)


LOGGER = logging.getLogger("generate_topology_targets")


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".ppm",
    ".bmp",
    ".gif",
)

OUTPUT_DIRECTORIES: Tuple[str, ...] = (
    "cleaned_masks",
    "skeletons",
    "endpoints",
    "junctions",
    "thin_vessels",
    "radius_maps",
    "width_maps",
    "metadata",
)


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass(frozen=True)
class TopologyTargetSettings:
    """Configuration controlling topology-target generation."""

    mask_threshold: float = 0.5

    remove_small_objects_enabled: bool = True
    minimum_object_size: int = 8

    fill_small_holes_enabled: bool = True
    maximum_hole_size: int = 8

    spur_pruning_enabled: bool = True
    spur_length_threshold: int = 5
    spur_pruning_max_iterations: int = 20

    border_margin: int = 5
    endpoint_radius: int = 2
    junction_radius: int = 3
    cluster_junctions: bool = True

    thin_vessel_enabled: bool = True
    thin_vessel_radius_threshold: float = 2.0

    cache_enabled: bool = True
    cache_directory: str = "datasets_topology"
    overwrite: bool = False


@dataclass
class ImageTopologyMetadata:
    """Topology statistics generated for one vessel annotation."""

    dataset: str
    source_mask: str
    height: int
    width: int

    vessel_pixels: int
    vessel_fraction: float

    skeleton_pixels: int
    skeleton_fraction: float

    raw_endpoint_count: int
    final_endpoint_count: int

    raw_junction_pixel_count: int
    junction_cluster_count: int

    connected_components: int
    skeleton_connected_components: int

    removed_small_object_pixels: int
    filled_hole_pixels: int
    removed_spur_pixels: int
    removed_spur_branches: int

    thin_vessel_pixels: int
    thin_vessel_fraction_of_mask: float

    mean_radius: float
    median_radius: float
    maximum_radius: float

    mean_width: float
    median_width: float
    maximum_width: float


@dataclass
class TopologyTargets:
    """All supervision targets generated from one vessel mask."""

    vessel_mask: np.ndarray
    cleaned_mask: np.ndarray
    skeleton_map: np.ndarray
    endpoint_map: np.ndarray
    junction_map: np.ndarray
    thin_vessel_map: np.ndarray
    radius_map: np.ndarray
    width_map: np.ndarray
    metadata: ImageTopologyMetadata


# ============================================================
# GENERAL UTILITIES
# ============================================================


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration root must be a dictionary: {path}"
        )

    return config


def ensure_directory(path: Path) -> None:
    """Create a directory and its parents when necessary."""

    path.mkdir(parents=True, exist_ok=True)


def list_mask_files(mask_directory: Path) -> List[Path]:
    """Return supported mask files in deterministic order."""

    if not mask_directory.exists():
        raise FileNotFoundError(
            f"Mask directory does not exist: {mask_directory}"
        )

    mask_paths = [
        path
        for path in mask_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    return sorted(
        mask_paths,
        key=lambda item: item.name.lower(),
    )


def read_binary_mask(
    mask_path: Path,
    threshold: float,
) -> np.ndarray:
    """
    Read an annotation and convert it to a boolean vessel mask.

    Multi-channel masks are converted to grayscale before thresholding.
    """

    image = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not read mask: {mask_path}"
        )

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY,
            )
        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

    image = image.astype(np.float32)

    maximum_value = float(image.max())

    if maximum_value > 1.0:
        image /= 255.0

    return image >= float(threshold)


def save_binary_png(
    output_path: Path,
    binary_map: np.ndarray,
) -> None:
    """Save a boolean or binary array as an 8-bit PNG."""

    ensure_directory(output_path.parent)

    output = (
        np.asarray(binary_map).astype(np.uint8) * 255
    )

    success = cv2.imwrite(
        str(output_path),
        output,
    )

    if not success:
        raise IOError(
            f"Failed to write image: {output_path}"
        )


def save_float_map(
    output_path: Path,
    float_map: np.ndarray,
) -> None:
    """
    Save a floating-point map losslessly as a NumPy array.

    Radius and width maps must not be stored as ordinary 8-bit PNGs
    because quantization would destroy their numerical meaning.
    """

    ensure_directory(output_path.parent)

    np.save(
        output_path,
        np.asarray(float_map, dtype=np.float32),
    )


def save_json(
    output_path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Save a JSON document with deterministic formatting."""

    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MORPHOLOGICAL MASK CLEANING
# ============================================================


def clean_mask(
    vessel_mask: np.ndarray,
    settings: TopologyTargetSettings,
) -> Tuple[np.ndarray, int, int]:
    """
    Clean a binary vessel mask conservatively.

    Returns
    -------
    cleaned_mask:
        Boolean cleaned vessel mask.
    removed_object_pixels:
        Number of foreground pixels removed as tiny components.
    filled_hole_pixels:
        Number of foreground pixels introduced by small-hole filling.
    """

    original = vessel_mask.astype(bool)
    cleaned = original.copy()

    before_object_removal = int(cleaned.sum())

    if settings.remove_small_objects_enabled:
        cleaned = remove_small_objects(
            cleaned,
            min_size=max(
                1,
                int(settings.minimum_object_size),
            ),
            connectivity=2,
        )

    after_object_removal = int(cleaned.sum())

    removed_object_pixels = max(
        0,
        before_object_removal - after_object_removal,
    )

    before_hole_filling = int(cleaned.sum())

    if settings.fill_small_holes_enabled:
        cleaned = remove_small_holes(
            cleaned,
            area_threshold=max(
                1,
                int(settings.maximum_hole_size),
            ),
            connectivity=2,
        )

    after_hole_filling = int(cleaned.sum())

    filled_hole_pixels = max(
        0,
        after_hole_filling - before_hole_filling,
    )

    return (
        cleaned.astype(bool),
        removed_object_pixels,
        filled_hole_pixels,
    )


# ============================================================
# SKELETON OPERATIONS
# ============================================================


def skeletonize_mask(
    cleaned_mask: np.ndarray,
) -> np.ndarray:
    """Create a one-pixel-wide topology-preserving skeleton."""

    return skeletonize(
        cleaned_mask.astype(bool),
    ).astype(bool)


def calculate_neighbour_count(
    skeleton_map: np.ndarray,
) -> np.ndarray:
    """
    Calculate each skeleton pixel's eight-connected neighbour count.

    The center pixel is excluded.
    """

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )
    kernel[1, 1] = 0

    count = ndi.convolve(
        skeleton_map.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )

    return count.astype(np.uint8)


def detect_raw_endpoints(
    skeleton_map: np.ndarray,
) -> np.ndarray:
    """Detect candidate endpoints using eight-neighbour degree one."""

    neighbour_count = calculate_neighbour_count(
        skeleton_map
    )

    return (
        skeleton_map
        & (neighbour_count == 1)
    )


def detect_raw_junction_pixels(
    skeleton_map: np.ndarray,
) -> np.ndarray:
    """
    Detect candidate junction pixels.

    A candidate junction has at least three active neighbours.
    Clustering is performed later.
    """

    neighbour_count = calculate_neighbour_count(
        skeleton_map
    )

    return (
        skeleton_map
        & (neighbour_count >= 3)
    )


# ============================================================
# CONSERVATIVE TERMINAL-SPUR PRUNING
# ============================================================


def get_active_neighbours(
    coordinate: Tuple[int, int],
    skeleton_map: np.ndarray,
) -> List[Tuple[int, int]]:
    """Return active eight-connected neighbours for one coordinate."""

    row, column = coordinate
    height, width = skeleton_map.shape

    neighbours: List[Tuple[int, int]] = []

    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue

            neighbour_row = row + row_offset
            neighbour_column = column + column_offset

            if not (
                0 <= neighbour_row < height
                and 0 <= neighbour_column < width
            ):
                continue

            if skeleton_map[
                neighbour_row,
                neighbour_column,
            ]:
                neighbours.append(
                    (
                        neighbour_row,
                        neighbour_column,
                    )
                )

    return neighbours


def trace_terminal_branch(
    skeleton_map: np.ndarray,
    endpoint: Tuple[int, int],
    maximum_length: int,
) -> Tuple[List[Tuple[int, int]], str]:
    """
    Trace a branch from an endpoint until a junction, another endpoint,
    a loop, or the maximum search length is reached.

    Returns
    -------
    branch:
        Ordered branch coordinates, including the starting endpoint.
    termination:
        One of:
        - "junction"
        - "endpoint"
        - "maximum_length"
        - "dead_end"
        - "loop"
    """

    branch: List[Tuple[int, int]] = [endpoint]
    visited = {endpoint}
    previous: Optional[Tuple[int, int]] = None
    current = endpoint

    while len(branch) <= maximum_length + 1:
        current_neighbours = [
            neighbour
            for neighbour in get_active_neighbours(
                current,
                skeleton_map,
            )
            if neighbour != previous
        ]

        unvisited_neighbours = [
            neighbour
            for neighbour in current_neighbours
            if neighbour not in visited
        ]

        total_degree = len(
            get_active_neighbours(
                current,
                skeleton_map,
            )
        )

        if current != endpoint:
            if total_degree >= 3:
                return branch, "junction"

            if total_degree <= 1:
                return branch, "endpoint"

        if not unvisited_neighbours:
            if current_neighbours:
                return branch, "loop"

            return branch, "dead_end"

        if len(unvisited_neighbours) > 1:
            return branch, "junction"

        next_coordinate = unvisited_neighbours[0]

        previous = current
        current = next_coordinate

        branch.append(current)
        visited.add(current)

    return branch, "maximum_length"


def prune_short_terminal_spurs(
    skeleton_map: np.ndarray,
    length_threshold: int,
    maximum_iterations: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Remove short terminal branches ending at junctions.

    Important
    ---------
    This function is deliberately conservative:

    - Only branches originating at endpoints are examined.
    - Only branches reaching a junction within the threshold are removed.
    - The terminal junction pixel itself is preserved.
    - Branches connecting two endpoints are not removed.
    """

    if length_threshold <= 0:
        return skeleton_map.copy(), 0, 0

    pruned = skeleton_map.astype(bool).copy()

    removed_pixels_total = 0
    removed_branches_total = 0

    for _ in range(maximum_iterations):
        endpoint_coordinates = np.argwhere(
            detect_raw_endpoints(pruned)
        )

        branches_removed_this_iteration = 0

        for endpoint_array in endpoint_coordinates:
            endpoint = (
                int(endpoint_array[0]),
                int(endpoint_array[1]),
            )

            if not pruned[endpoint]:
                continue

            branch, termination = trace_terminal_branch(
                skeleton_map=pruned,
                endpoint=endpoint,
                maximum_length=length_threshold,
            )

            if termination != "junction":
                continue

            # Exclude the final junction pixel from removal.
            removable_branch = branch[:-1]

            if not removable_branch:
                continue

            if len(removable_branch) > length_threshold:
                continue

            for coordinate in removable_branch:
                if pruned[coordinate]:
                    pruned[coordinate] = False
                    removed_pixels_total += 1

            removed_branches_total += 1
            branches_removed_this_iteration += 1

        if branches_removed_this_iteration == 0:
            break

    return (
        pruned,
        removed_pixels_total,
        removed_branches_total,
    )


# ============================================================
# NODE TARGET GENERATION
# ============================================================


def suppress_border_coordinates(
    binary_map: np.ndarray,
    border_margin: int,
) -> np.ndarray:
    """Remove detections close to the rectangular image boundary."""

    filtered = binary_map.astype(bool).copy()

    if border_margin <= 0:
        return filtered

    height, width = filtered.shape

    margin = min(
        int(border_margin),
        height // 2,
        width // 2,
    )

    if margin <= 0:
        return filtered

    filtered[:margin, :] = False
    filtered[height - margin :, :] = False
    filtered[:, :margin] = False
    filtered[:, width - margin :] = False

    return filtered


def draw_disk_at_coordinate(
    output_map: np.ndarray,
    coordinate: Tuple[float, float],
    radius: int,
) -> None:
    """Draw a clipped binary disk into an existing array."""

    center_row = int(round(coordinate[0]))
    center_column = int(round(coordinate[1]))

    if radius <= 0:
        if (
            0 <= center_row < output_map.shape[0]
            and 0 <= center_column < output_map.shape[1]
        ):
            output_map[
                center_row,
                center_column,
            ] = True

        return

    structuring_element = disk(radius).astype(bool)
    diameter = structuring_element.shape[0]
    disk_center = diameter // 2

    row_start = max(
        0,
        center_row - disk_center,
    )
    row_end = min(
        output_map.shape[0],
        center_row + disk_center + 1,
    )

    column_start = max(
        0,
        center_column - disk_center,
    )
    column_end = min(
        output_map.shape[1],
        center_column + disk_center + 1,
    )

    element_row_start = row_start - (
        center_row - disk_center
    )
    element_row_end = element_row_start + (
        row_end - row_start
    )

    element_column_start = column_start - (
        center_column - disk_center
    )
    element_column_end = element_column_start + (
        column_end - column_start
    )

    output_map[
        row_start:row_end,
        column_start:column_end,
    ] |= structuring_element[
        element_row_start:element_row_end,
        element_column_start:element_column_end,
    ]


def create_endpoint_target(
    skeleton_map: np.ndarray,
    border_margin: int,
    endpoint_radius: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Create endpoint supervision regions.

    Returns
    -------
    endpoint_target:
        Dilated endpoint heatmap.
    raw_endpoint_count:
        Endpoint count before border suppression.
    final_endpoint_count:
        Endpoint count after border suppression.
    """

    raw_endpoints = detect_raw_endpoints(
        skeleton_map
    )

    raw_endpoint_count = int(
        raw_endpoints.sum()
    )

    filtered_endpoints = suppress_border_coordinates(
        raw_endpoints,
        border_margin=border_margin,
    )

    endpoint_coordinates = np.argwhere(
        filtered_endpoints
    )

    endpoint_target = np.zeros_like(
        skeleton_map,
        dtype=bool,
    )

    for coordinate in endpoint_coordinates:
        draw_disk_at_coordinate(
            endpoint_target,
            coordinate=(
                float(coordinate[0]),
                float(coordinate[1]),
            ),
            radius=endpoint_radius,
        )

    return (
        endpoint_target,
        raw_endpoint_count,
        int(len(endpoint_coordinates)),
    )


def create_junction_target(
    skeleton_map: np.ndarray,
    border_margin: int,
    junction_radius: int,
    cluster_junctions: bool,
) -> Tuple[np.ndarray, int, int]:
    """
    Create compact junction supervision regions.

    Adjacent raw junction pixels are merged into one connected cluster.
    One disk is drawn around each cluster centroid.
    """

    raw_junction_pixels = detect_raw_junction_pixels(
        skeleton_map
    )

    raw_junction_pixel_count = int(
        raw_junction_pixels.sum()
    )

    filtered_junction_pixels = suppress_border_coordinates(
        raw_junction_pixels,
        border_margin=border_margin,
    )

    junction_target = np.zeros_like(
        skeleton_map,
        dtype=bool,
    )

    if not filtered_junction_pixels.any():
        return (
            junction_target,
            raw_junction_pixel_count,
            0,
        )

    if cluster_junctions:
        labelled_junctions = label(
            filtered_junction_pixels,
            connectivity=2,
        )

        properties = regionprops(
            labelled_junctions
        )

        for region in properties:
            draw_disk_at_coordinate(
                junction_target,
                coordinate=region.centroid,
                radius=junction_radius,
            )

        cluster_count = len(properties)

    else:
        coordinates = np.argwhere(
            filtered_junction_pixels
        )

        for coordinate in coordinates:
            draw_disk_at_coordinate(
                junction_target,
                coordinate=(
                    float(coordinate[0]),
                    float(coordinate[1]),
                ),
                radius=junction_radius,
            )

        cluster_count = int(len(coordinates))

    return (
        junction_target,
        raw_junction_pixel_count,
        cluster_count,
    )


# ============================================================
# VESSEL-WIDTH TARGETS
# ============================================================


def calculate_vessel_width_targets(
    cleaned_mask: np.ndarray,
    thin_radius_threshold: float,
    thin_vessel_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate radius, width, and thin-vessel maps.

    The Euclidean distance transform estimates the distance from each
    vessel pixel to its nearest background pixel.

    Approximate diameter:

        width = 2 * radius
    """

    radius_map = ndi.distance_transform_edt(
        cleaned_mask.astype(bool)
    ).astype(np.float32)

    width_map = (
        2.0 * radius_map
    ).astype(np.float32)

    if thin_vessel_enabled:
        thin_vessel_map = (
            cleaned_mask
            & (
                radius_map
                <= float(thin_radius_threshold)
            )
        )
    else:
        thin_vessel_map = np.zeros_like(
            cleaned_mask,
            dtype=bool,
        )

    return (
        radius_map,
        width_map,
        thin_vessel_map.astype(bool),
    )


# ============================================================
# METADATA
# ============================================================


def count_connected_components(
    binary_map: np.ndarray,
) -> int:
    """Count eight-connected foreground components."""

    if not binary_map.any():
        return 0

    labelled = label(
        binary_map.astype(bool),
        connectivity=2,
    )

    return int(labelled.max())


def safe_statistics(
    values: np.ndarray,
) -> Tuple[float, float, float]:
    """Return mean, median, and maximum safely."""

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.size == 0:
        return 0.0, 0.0, 0.0

    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.max(values)),
    )


def build_metadata(
    dataset_name: str,
    source_mask: Path,
    original_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    skeleton_map: np.ndarray,
    thin_vessel_map: np.ndarray,
    radius_map: np.ndarray,
    width_map: np.ndarray,
    raw_endpoint_count: int,
    final_endpoint_count: int,
    raw_junction_pixel_count: int,
    junction_cluster_count: int,
    removed_small_object_pixels: int,
    filled_hole_pixels: int,
    removed_spur_pixels: int,
    removed_spur_branches: int,
) -> ImageTopologyMetadata:
    """Create per-image topology metadata."""

    height, width = cleaned_mask.shape
    total_pixels = height * width

    vessel_pixels = int(cleaned_mask.sum())
    skeleton_pixels = int(skeleton_map.sum())
    thin_vessel_pixels = int(thin_vessel_map.sum())

    vessel_radius_values = radius_map[
        cleaned_mask
    ]

    vessel_width_values = width_map[
        cleaned_mask
    ]

    (
        mean_radius,
        median_radius,
        maximum_radius,
    ) = safe_statistics(
        vessel_radius_values
    )

    (
        mean_width,
        median_width,
        maximum_width,
    ) = safe_statistics(
        vessel_width_values
    )

    vessel_fraction = (
        vessel_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    skeleton_fraction = (
        skeleton_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    thin_fraction = (
        thin_vessel_pixels / vessel_pixels
        if vessel_pixels > 0
        else 0.0
    )

    return ImageTopologyMetadata(
        dataset=dataset_name,
        source_mask=str(source_mask),
        height=int(height),
        width=int(width),
        vessel_pixels=vessel_pixels,
        vessel_fraction=float(vessel_fraction),
        skeleton_pixels=skeleton_pixels,
        skeleton_fraction=float(skeleton_fraction),
        raw_endpoint_count=int(raw_endpoint_count),
        final_endpoint_count=int(final_endpoint_count),
        raw_junction_pixel_count=int(
            raw_junction_pixel_count
        ),
        junction_cluster_count=int(
            junction_cluster_count
        ),
        connected_components=count_connected_components(
            cleaned_mask
        ),
        skeleton_connected_components=(
            count_connected_components(
                skeleton_map
            )
        ),
        removed_small_object_pixels=int(
            removed_small_object_pixels
        ),
        filled_hole_pixels=int(
            filled_hole_pixels
        ),
        removed_spur_pixels=int(
            removed_spur_pixels
        ),
        removed_spur_branches=int(
            removed_spur_branches
        ),
        thin_vessel_pixels=thin_vessel_pixels,
        thin_vessel_fraction_of_mask=float(
            thin_fraction
        ),
        mean_radius=mean_radius,
        median_radius=median_radius,
        maximum_radius=maximum_radius,
        mean_width=mean_width,
        median_width=median_width,
        maximum_width=maximum_width,
    )


# ============================================================
# COMPLETE TARGET GENERATION
# ============================================================


def generate_topology_targets(
    vessel_mask: np.ndarray,
    settings: TopologyTargetSettings,
    dataset_name: str,
    source_mask: Path,
) -> TopologyTargets:
    """Generate all topology targets from one binary vessel mask."""

    (
        cleaned_mask,
        removed_small_object_pixels,
        filled_hole_pixels,
    ) = clean_mask(
        vessel_mask=vessel_mask,
        settings=settings,
    )

    original_skeleton = skeletonize_mask(
        cleaned_mask
    )

    if settings.spur_pruning_enabled:
        (
            final_skeleton,
            removed_spur_pixels,
            removed_spur_branches,
        ) = prune_short_terminal_spurs(
            skeleton_map=original_skeleton,
            length_threshold=(
                settings.spur_length_threshold
            ),
            maximum_iterations=(
                settings.spur_pruning_max_iterations
            ),
        )
    else:
        final_skeleton = original_skeleton
        removed_spur_pixels = 0
        removed_spur_branches = 0

    (
        endpoint_map,
        raw_endpoint_count,
        final_endpoint_count,
    ) = create_endpoint_target(
        skeleton_map=final_skeleton,
        border_margin=settings.border_margin,
        endpoint_radius=settings.endpoint_radius,
    )

    (
        junction_map,
        raw_junction_pixel_count,
        junction_cluster_count,
    ) = create_junction_target(
        skeleton_map=final_skeleton,
        border_margin=settings.border_margin,
        junction_radius=settings.junction_radius,
        cluster_junctions=(
            settings.cluster_junctions
        ),
    )

    (
        radius_map,
        width_map,
        thin_vessel_map,
    ) = calculate_vessel_width_targets(
        cleaned_mask=cleaned_mask,
        thin_radius_threshold=(
            settings.thin_vessel_radius_threshold
        ),
        thin_vessel_enabled=(
            settings.thin_vessel_enabled
        ),
    )

    metadata = build_metadata(
        dataset_name=dataset_name,
        source_mask=source_mask,
        original_mask=vessel_mask,
        cleaned_mask=cleaned_mask,
        skeleton_map=final_skeleton,
        thin_vessel_map=thin_vessel_map,
        radius_map=radius_map,
        width_map=width_map,
        raw_endpoint_count=raw_endpoint_count,
        final_endpoint_count=final_endpoint_count,
        raw_junction_pixel_count=(
            raw_junction_pixel_count
        ),
        junction_cluster_count=(
            junction_cluster_count
        ),
        removed_small_object_pixels=(
            removed_small_object_pixels
        ),
        filled_hole_pixels=filled_hole_pixels,
        removed_spur_pixels=removed_spur_pixels,
        removed_spur_branches=removed_spur_branches,
    )

    return TopologyTargets(
        vessel_mask=vessel_mask.astype(bool),
        cleaned_mask=cleaned_mask.astype(bool),
        skeleton_map=final_skeleton.astype(bool),
        endpoint_map=endpoint_map.astype(bool),
        junction_map=junction_map.astype(bool),
        thin_vessel_map=thin_vessel_map.astype(bool),
        radius_map=radius_map.astype(np.float32),
        width_map=width_map.astype(np.float32),
        metadata=metadata,
    )


# ============================================================
# CONFIGURATION
# ============================================================


def get_nested(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    default: Any,
) -> Any:
    """Read a nested configuration value safely."""

    current: Any = mapping

    for key in keys:
        if not isinstance(current, Mapping):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def parse_topology_settings(
    config: Mapping[str, Any],
    overwrite_override: Optional[bool],
) -> TopologyTargetSettings:
    """Build topology settings from config.yaml."""

    topology_config = config.get(
        "topology_targets",
        {},
    )

    if not isinstance(topology_config, Mapping):
        topology_config = {}

    overwrite = bool(
        get_nested(
            topology_config,
            ["cache", "overwrite"],
            False,
        )
    )

    if overwrite_override is not None:
        overwrite = overwrite_override

    return TopologyTargetSettings(
        mask_threshold=float(
            topology_config.get(
                "mask_threshold",
                0.5,
            )
        ),
        remove_small_objects_enabled=bool(
            get_nested(
                topology_config,
                [
                    "cleanup",
                    "remove_small_objects",
                ],
                True,
            )
        ),
        minimum_object_size=int(
            get_nested(
                topology_config,
                [
                    "cleanup",
                    "minimum_object_size",
                ],
                8,
            )
        ),
        fill_small_holes_enabled=bool(
            get_nested(
                topology_config,
                [
                    "cleanup",
                    "fill_small_holes",
                ],
                True,
            )
        ),
        maximum_hole_size=int(
            get_nested(
                topology_config,
                [
                    "cleanup",
                    "maximum_hole_size",
                ],
                8,
            )
        ),
        spur_pruning_enabled=bool(
            get_nested(
                topology_config,
                [
                    "skeleton",
                    "spur_pruning_enabled",
                ],
                True,
            )
        ),
        spur_length_threshold=int(
            get_nested(
                topology_config,
                [
                    "skeleton",
                    "spur_length_threshold",
                ],
                5,
            )
        ),
        spur_pruning_max_iterations=int(
            get_nested(
                topology_config,
                [
                    "skeleton",
                    "max_iterations",
                ],
                20,
            )
        ),
        border_margin=int(
            get_nested(
                topology_config,
                [
                    "nodes",
                    "border_margin",
                ],
                5,
            )
        ),
        endpoint_radius=int(
            get_nested(
                topology_config,
                [
                    "nodes",
                    "endpoint_radius",
                ],
                2,
            )
        ),
        junction_radius=int(
            get_nested(
                topology_config,
                [
                    "nodes",
                    "junction_radius",
                ],
                3,
            )
        ),
        cluster_junctions=bool(
            get_nested(
                topology_config,
                [
                    "nodes",
                    "cluster_junctions",
                ],
                True,
            )
        ),
        thin_vessel_enabled=bool(
            get_nested(
                topology_config,
                [
                    "thin_vessels",
                    "enabled",
                ],
                True,
            )
        ),
        thin_vessel_radius_threshold=float(
            get_nested(
                topology_config,
                [
                    "thin_vessels",
                    "radius_threshold",
                ],
                2.0,
            )
        ),
        cache_enabled=bool(
            get_nested(
                topology_config,
                [
                    "cache",
                    "enabled",
                ],
                True,
            )
        ),
        cache_directory=str(
            get_nested(
                topology_config,
                [
                    "cache",
                    "directory",
                ],
                "datasets_topology",
            )
        ),
        overwrite=overwrite,
    )


def discover_dataset_mask_directories(
    config: Mapping[str, Any],
) -> Dict[str, Path]:
    """
    Discover all configured mask directories.

    The function supports:
    - train_dataset
    - test_dataset
    - external_datasets
    """

    dataset_config = config.get(
        "dataset",
        {},
    )

    if not isinstance(dataset_config, Mapping):
        raise ValueError(
            "config.yaml does not contain a valid dataset section."
        )

    datasets: Dict[str, Path] = {}

    for section_name in (
        "train_dataset",
        "test_dataset",
    ):
        section = dataset_config.get(
            section_name
        )

        if not isinstance(section, Mapping):
            continue

        mask_directory = section.get(
            "mask_dir"
        )

        if not mask_directory:
            continue

        dataset_name = str(
            section.get(
                "name",
                section_name,
            )
        ).strip()

        datasets[dataset_name] = Path(
            str(mask_directory)
        )

    external_datasets = dataset_config.get(
        "external_datasets",
        {},
    )

    if isinstance(external_datasets, Mapping):
        for dataset_name, dataset_section in (
            external_datasets.items()
        ):
            if not isinstance(
                dataset_section,
                Mapping,
            ):
                continue

            mask_directory = dataset_section.get(
                "mask_dir"
            )

            if not mask_directory:
                continue

            datasets[str(dataset_name)] = Path(
                str(mask_directory)
            )

    if not datasets:
        raise ValueError(
            "No dataset mask directories were found in config.yaml."
        )

    return datasets


# ============================================================
# OUTPUT MANAGEMENT
# ============================================================


def create_dataset_output_directories(
    dataset_output_root: Path,
) -> Dict[str, Path]:
    """Create and return all dataset output directories."""

    directories: Dict[str, Path] = {}

    for directory_name in OUTPUT_DIRECTORIES:
        directory_path = (
            dataset_output_root / directory_name
        )
        ensure_directory(directory_path)
        directories[directory_name] = directory_path

    return directories


def expected_output_paths(
    output_directories: Mapping[str, Path],
    mask_path: Path,
) -> Dict[str, Path]:
    """Construct all expected output paths for one mask."""

    stem = mask_path.stem

    return {
        "cleaned_mask": (
            output_directories["cleaned_masks"]
            / f"{stem}.png"
        ),
        "skeleton": (
            output_directories["skeletons"]
            / f"{stem}.png"
        ),
        "endpoint": (
            output_directories["endpoints"]
            / f"{stem}.png"
        ),
        "junction": (
            output_directories["junctions"]
            / f"{stem}.png"
        ),
        "thin_vessel": (
            output_directories["thin_vessels"]
            / f"{stem}.png"
        ),
        "radius_map": (
            output_directories["radius_maps"]
            / f"{stem}.npy"
        ),
        "width_map": (
            output_directories["width_maps"]
            / f"{stem}.npy"
        ),
        "metadata": (
            output_directories["metadata"]
            / f"{stem}.json"
        ),
    }


def outputs_exist(
    output_paths: Mapping[str, Path],
) -> bool:
    """Return True when every expected output exists."""

    return all(
        path.exists()
        for path in output_paths.values()
    )


def save_topology_targets(
    targets: TopologyTargets,
    output_paths: Mapping[str, Path],
) -> None:
    """Persist all generated targets."""

    save_binary_png(
        output_paths["cleaned_mask"],
        targets.cleaned_mask,
    )

    save_binary_png(
        output_paths["skeleton"],
        targets.skeleton_map,
    )

    save_binary_png(
        output_paths["endpoint"],
        targets.endpoint_map,
    )

    save_binary_png(
        output_paths["junction"],
        targets.junction_map,
    )

    save_binary_png(
        output_paths["thin_vessel"],
        targets.thin_vessel_map,
    )

    save_float_map(
        output_paths["radius_map"],
        targets.radius_map,
    )

    save_float_map(
        output_paths["width_map"],
        targets.width_map,
    )

    save_json(
        output_paths["metadata"],
        asdict(targets.metadata),
    )


def read_existing_metadata(
    metadata_path: Path,
) -> ImageTopologyMetadata:
    """Read cached per-image metadata."""

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    return ImageTopologyMetadata(**payload)


# ============================================================
# DATASET PROCESSING
# ============================================================


def process_dataset(
    dataset_name: str,
    mask_directory: Path,
    topology_root: Path,
    settings: TopologyTargetSettings,
) -> List[ImageTopologyMetadata]:
    """Generate topology targets for one complete dataset."""

    LOGGER.info(
        "Processing dataset: %s",
        dataset_name,
    )
    LOGGER.info(
        "Mask directory: %s",
        mask_directory,
    )

    mask_paths = list_mask_files(
        mask_directory
    )

    if not mask_paths:
        LOGGER.warning(
            "No supported masks found for %s.",
            dataset_name,
        )
        return []

    dataset_output_root = (
        topology_root / dataset_name
    )

    output_directories = (
        create_dataset_output_directories(
            dataset_output_root
        )
    )

    metadata_records: List[
        ImageTopologyMetadata
    ] = []

    for index, mask_path in enumerate(
        mask_paths,
        start=1,
    ):
        paths = expected_output_paths(
            output_directories,
            mask_path,
        )

        if (
            not settings.overwrite
            and outputs_exist(paths)
        ):
            metadata = read_existing_metadata(
                paths["metadata"]
            )
            metadata_records.append(metadata)

            LOGGER.info(
                "[%d/%d] Cached: %s",
                index,
                len(mask_paths),
                mask_path.name,
            )
            continue

        try:
            vessel_mask = read_binary_mask(
                mask_path=mask_path,
                threshold=settings.mask_threshold,
            )

            targets = generate_topology_targets(
                vessel_mask=vessel_mask,
                settings=settings,
                dataset_name=dataset_name,
                source_mask=mask_path,
            )

            save_topology_targets(
                targets=targets,
                output_paths=paths,
            )

            metadata_records.append(
                targets.metadata
            )

            LOGGER.info(
                (
                    "[%d/%d] Generated: %s | "
                    "skeleton=%d | endpoints=%d | "
                    "junctions=%d | components=%d"
                ),
                index,
                len(mask_paths),
                mask_path.name,
                targets.metadata.skeleton_pixels,
                targets.metadata.final_endpoint_count,
                targets.metadata.junction_cluster_count,
                (
                    targets.metadata
                    .skeleton_connected_components
                ),
            )

        except Exception:
            LOGGER.exception(
                "Failed to process mask: %s",
                mask_path,
            )
            raise

    write_dataset_reports(
        dataset_name=dataset_name,
        records=metadata_records,
        output_root=dataset_output_root,
    )

    return metadata_records


# ============================================================
# REPORT GENERATION
# ============================================================


def numeric_mean(
    values: Iterable[float],
) -> float:
    """Calculate a finite arithmetic mean."""

    valid_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if not valid_values:
        return 0.0

    return float(
        np.mean(valid_values)
    )


def build_dataset_summary(
    dataset_name: str,
    records: Sequence[ImageTopologyMetadata],
) -> Dict[str, Any]:
    """Aggregate image metadata into one dataset summary."""

    return {
        "dataset": dataset_name,
        "images": len(records),
        "average_vessel_pixels": numeric_mean(
            record.vessel_pixels
            for record in records
        ),
        "average_vessel_fraction": numeric_mean(
            record.vessel_fraction
            for record in records
        ),
        "average_skeleton_pixels": numeric_mean(
            record.skeleton_pixels
            for record in records
        ),
        "average_endpoints": numeric_mean(
            record.final_endpoint_count
            for record in records
        ),
        "average_junctions": numeric_mean(
            record.junction_cluster_count
            for record in records
        ),
        "average_connected_components": numeric_mean(
            record.connected_components
            for record in records
        ),
        "average_skeleton_components": numeric_mean(
            record.skeleton_connected_components
            for record in records
        ),
        "average_removed_spur_pixels": numeric_mean(
            record.removed_spur_pixels
            for record in records
        ),
        "average_removed_spur_branches": numeric_mean(
            record.removed_spur_branches
            for record in records
        ),
        "average_thin_vessel_fraction": numeric_mean(
            record.thin_vessel_fraction_of_mask
            for record in records
        ),
        "average_radius": numeric_mean(
            record.mean_radius
            for record in records
        ),
        "average_width": numeric_mean(
            record.mean_width
            for record in records
        ),
        "maximum_observed_width": max(
            (
                record.maximum_width
                for record in records
            ),
            default=0.0,
        ),
    }


def write_dataset_reports(
    dataset_name: str,
    records: Sequence[ImageTopologyMetadata],
    output_root: Path,
) -> None:
    """Write JSON and CSV reports for one dataset."""

    summary = build_dataset_summary(
        dataset_name=dataset_name,
        records=records,
    )

    save_json(
        output_root / "dataset_summary.json",
        summary,
    )

    csv_path = (
        output_root / "image_topology_report.csv"
    )

    ensure_directory(csv_path.parent)

    if records:
        field_names = list(
            asdict(records[0]).keys()
        )

        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=field_names,
            )

            writer.writeheader()

            for record in records:
                writer.writerow(
                    asdict(record)
                )

    LOGGER.info(
        (
            "Completed %s | images=%d | "
            "avg endpoints=%.2f | avg junctions=%.2f | "
            "avg thin fraction=%.4f"
        ),
        dataset_name,
        len(records),
        summary["average_endpoints"],
        summary["average_junctions"],
        summary["average_thin_vessel_fraction"],
    )


def write_global_report(
    topology_root: Path,
    dataset_summaries: Sequence[Dict[str, Any]],
) -> None:
    """Write a global report containing every processed dataset."""

    save_json(
        topology_root / "all_dataset_summaries.json",
        {
            "datasets": list(dataset_summaries),
        },
    )

    csv_path = (
        topology_root / "all_dataset_summaries.csv"
    )

    if not dataset_summaries:
        return

    field_names = list(
        dataset_summaries[0].keys()
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
        )

        writer.writeheader()

        for summary in dataset_summaries:
            writer.writerow(summary)


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate topology-aware supervision targets "
            "for MV-TransUNet++."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Optional configured dataset name. "
            "When omitted, every dataset is processed."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached topology targets.",
    )

    return parser.parse_args()


def main() -> None:
    """Program entry point."""

    configure_logging()

    arguments = parse_arguments()

    config = load_yaml(
        arguments.config
    )

    overwrite_override: Optional[bool]

    if arguments.overwrite:
        overwrite_override = True
    else:
        overwrite_override = None

    settings = parse_topology_settings(
        config=config,
        overwrite_override=overwrite_override,
    )

    if not settings.cache_enabled:
        raise ValueError(
            "topology_targets.cache.enabled is false. "
            "Enable it before generating targets."
        )

    datasets = discover_dataset_mask_directories(
        config
    )

    if arguments.dataset is not None:
        requested_name = (
            arguments.dataset.strip()
        )

        matching_name = next(
            (
                name
                for name in datasets
                if name.lower()
                == requested_name.lower()
            ),
            None,
        )

        if matching_name is None:
            available = ", ".join(
                sorted(datasets.keys())
            )

            raise ValueError(
                f"Unknown dataset '{requested_name}'. "
                f"Available datasets: {available}"
            )

        datasets = {
            matching_name: datasets[matching_name]
        }

    topology_root = Path(
        settings.cache_directory
    )

    ensure_directory(topology_root)

    LOGGER.info(
        "Topology output root: %s",
        topology_root,
    )

    LOGGER.info(
        "Settings: %s",
        settings,
    )

    dataset_summaries: List[
        Dict[str, Any]
    ] = []

    for dataset_name, mask_directory in (
        datasets.items()
    ):
        records = process_dataset(
            dataset_name=dataset_name,
            mask_directory=mask_directory,
            topology_root=topology_root,
            settings=settings,
        )

        if records:
            dataset_summaries.append(
                build_dataset_summary(
                    dataset_name,
                    records,
                )
            )

    write_global_report(
        topology_root=topology_root,
        dataset_summaries=dataset_summaries,
    )

    LOGGER.info(
        "Topology-target generation completed successfully."
    )


if __name__ == "__main__":
    main()