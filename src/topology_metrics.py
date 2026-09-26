"""
Topology-sensitive evaluation metrics for retinal vessel segmentation.

This module is evaluation-only. It does not modify the model or training loss.

Metrics:
    - hard clDice
    - topology precision and sensitivity
    - tolerance-aware skeleton precision, recall, and F1
    - connected-component statistics
    - fragmentation ratio
    - largest-connected-component ratio
    - thin-vessel Dice, precision, recall, and F1
    - differentiable soft skeletonization for Stage D.2
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    label,
)
from skimage.morphology import disk, skeletonize


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class TopologyMetricConfig:
    prediction_threshold: float = 0.5
    skeleton_tolerance_px: int = 1
    thin_radius_px: float = 2.0
    thin_roi_dilation_px: int = 2
    minimum_component_size: int = 0
    connectivity: int = 2
    epsilon: float = 1e-7

    def validate(self) -> None:
        if not 0.0 <= self.prediction_threshold <= 1.0:
            raise ValueError("prediction_threshold must be in [0, 1].")

        if self.skeleton_tolerance_px < 0:
            raise ValueError("skeleton_tolerance_px must be >= 0.")

        if self.thin_radius_px <= 0:
            raise ValueError("thin_radius_px must be > 0.")

        if self.thin_roi_dilation_px < 0:
            raise ValueError("thin_roi_dilation_px must be >= 0.")

        if self.minimum_component_size < 0:
            raise ValueError("minimum_component_size must be >= 0.")

        if self.connectivity not in (1, 2):
            raise ValueError("connectivity must be 1 or 2.")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0.")


def _to_numpy(value: ArrayLike) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _normalise_to_nhw(
    value: ArrayLike,
    name: str,
) -> np.ndarray:
    """
    Convert an input array to [N, H, W].

    Accepted shapes:
        [H, W]
        [N, H, W]
        [N, 1, H, W]
    """

    array = _to_numpy(value)

    if array.ndim == 2:
        array = array[None, ...]

    elif array.ndim == 3:
        pass

    elif array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(
                f"{name} must have one channel. "
                f"Received shape {array.shape}."
            )

        array = array[:, 0, ...]

    else:
        raise ValueError(
            f"{name} must have 2, 3, or 4 dimensions. "
            f"Received shape {array.shape}."
        )

    return array


def _binarise_prediction(
    prediction: ArrayLike,
    threshold: float,
    from_logits: bool,
) -> np.ndarray:
    prediction_array = _normalise_to_nhw(
        prediction,
        "prediction",
    ).astype(np.float32)

    if not np.isfinite(prediction_array).all():
        raise ValueError(
            "prediction contains NaN or infinite values."
        )

    if from_logits:
        prediction_array = 1.0 / (
            1.0 + np.exp(-prediction_array)
        )

    return prediction_array >= threshold


def _binarise_target(
    target: ArrayLike,
) -> np.ndarray:
    target_array = _normalise_to_nhw(
        target,
        "target",
    )

    if not np.isfinite(target_array).all():
        raise ValueError(
            "target contains NaN or infinite values."
        )

    return target_array > 0.5


def _safe_divide(
    numerator: float,
    denominator: float,
    epsilon: float,
    empty_value: float,
) -> float:
    if denominator <= epsilon:
        return float(empty_value)

    return float(
        numerator / (denominator + epsilon)
    )


def _harmonic_mean(
    first: float,
    second: float,
    epsilon: float,
) -> float:
    denominator = first + second

    if denominator <= epsilon:
        return 0.0

    return float(
        (2.0 * first * second)
        / (denominator + epsilon)
    )


def _dilate(
    mask: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)

    return binary_dilation(
        mask,
        structure=disk(radius_px),
    )


def _remove_small_components(
    mask: np.ndarray,
    minimum_component_size: int,
    connectivity: int,
) -> np.ndarray:
    mask = mask.astype(bool)

    if minimum_component_size <= 1:
        return mask

    structure = generate_binary_structure(
        rank=2,
        connectivity=connectivity,
    )

    component_map, number_of_components = label(
        mask,
        structure=structure,
    )

    if number_of_components == 0:
        return mask

    component_sizes = np.bincount(
        component_map.ravel()
    )

    keep_components = (
        component_sizes >= minimum_component_size
    )

    keep_components[0] = False

    return keep_components[component_map]


def _component_statistics(
    mask: np.ndarray,
    connectivity: int,
    epsilon: float,
) -> Dict[str, float]:
    structure = generate_binary_structure(
        rank=2,
        connectivity=connectivity,
    )

    component_map, component_count = label(
        mask,
        structure=structure,
    )

    foreground_pixels = int(mask.sum())

    if component_count == 0 or foreground_pixels == 0:
        return {
            "component_count": 0,
            "largest_component_size": 0,
            "largest_component_ratio": 1.0,
        }

    component_sizes = np.bincount(
        component_map.ravel()
    )[1:]

    largest_component_size = int(
        component_sizes.max()
    )

    largest_component_ratio = _safe_divide(
        numerator=float(largest_component_size),
        denominator=float(foreground_pixels),
        epsilon=epsilon,
        empty_value=1.0,
    )

    return {
        "component_count": int(component_count),
        "largest_component_size": (
            largest_component_size
        ),
        "largest_component_ratio": (
            largest_component_ratio
        ),
    }


def hard_cldice(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    """
    Calculate hard clDice.

    Topology precision:
        predicted skeleton inside ground-truth mask

    Topology sensitivity:
        ground-truth skeleton inside predicted mask
    """

    prediction_mask = prediction_mask.astype(bool)
    target_mask = target_mask.astype(bool)

    prediction_skeleton = skeletonize(
        prediction_mask
    )

    target_skeleton = skeletonize(
        target_mask
    )

    prediction_skeleton_pixels = float(
        prediction_skeleton.sum()
    )

    target_skeleton_pixels = float(
        target_skeleton.sum()
    )

    if (
        prediction_skeleton_pixels <= epsilon
        and target_skeleton_pixels <= epsilon
    ):
        topology_precision = 1.0
        topology_sensitivity = 1.0

    else:
        topology_precision = _safe_divide(
            numerator=float(
                np.logical_and(
                    prediction_skeleton,
                    target_mask,
                ).sum()
            ),
            denominator=prediction_skeleton_pixels,
            epsilon=epsilon,
            empty_value=0.0,
        )

        topology_sensitivity = _safe_divide(
            numerator=float(
                np.logical_and(
                    target_skeleton,
                    prediction_mask,
                ).sum()
            ),
            denominator=target_skeleton_pixels,
            epsilon=epsilon,
            empty_value=0.0,
        )

    cldice = _harmonic_mean(
        topology_precision,
        topology_sensitivity,
        epsilon,
    )

    return {
        "cldice": cldice,
        "topology_precision": topology_precision,
        "topology_sensitivity": topology_sensitivity,
    }


def tolerant_skeleton_metrics(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    tolerance_px: int = 1,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    """
    Skeleton precision and recall with spatial tolerance.

    A one-pixel displacement should not automatically count as
    a complete structural failure.
    """

    prediction_skeleton = skeletonize(
        prediction_mask.astype(bool)
    )

    target_skeleton = skeletonize(
        target_mask.astype(bool)
    )

    dilated_target_skeleton = _dilate(
        target_skeleton,
        tolerance_px,
    )

    dilated_prediction_skeleton = _dilate(
        prediction_skeleton,
        tolerance_px,
    )

    prediction_pixels = float(
        prediction_skeleton.sum()
    )

    target_pixels = float(
        target_skeleton.sum()
    )

    if (
        prediction_pixels <= epsilon
        and target_pixels <= epsilon
    ):
        precision = 1.0
        recall = 1.0

    else:
        precision = _safe_divide(
            numerator=float(
                np.logical_and(
                    prediction_skeleton,
                    dilated_target_skeleton,
                ).sum()
            ),
            denominator=prediction_pixels,
            epsilon=epsilon,
            empty_value=0.0,
        )

        recall = _safe_divide(
    numerator=float(
        np.logical_and(
            target_skeleton,
            dilated_prediction_skeleton,
        ).sum()
    ),
    denominator=target_pixels,
    epsilon=epsilon,
    empty_value=0.0,
    
)

    skeleton_f1 = _harmonic_mean(
        precision,
        recall,
        epsilon,
    )

    return {
        "skeleton_precision": precision,
        "skeleton_recall": recall,
        "skeleton_f1": skeleton_f1,
        "prediction_skeleton_pixels": int(
            prediction_pixels
        ),
        "target_skeleton_pixels": int(
            target_pixels
        ),
    }


def thin_vessel_metrics(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    thin_radius_px: float = 2.0,
    roi_dilation_px: int = 2,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    """
    Evaluate prediction quality on thin target vessels.

    The Euclidean distance transform estimates the local vessel
    radius. Vessel pixels with radius <= thin_radius_px are treated
    as thin vessels.
    """

    prediction_mask = prediction_mask.astype(bool)
    target_mask = target_mask.astype(bool)

    local_radius = distance_transform_edt(
        target_mask
    )

    thin_target = (
        target_mask
        & (local_radius <= thin_radius_px)
    )

    thin_roi = _dilate(
        thin_target,
        roi_dilation_px,
    )

    thin_prediction = (
        prediction_mask
        & thin_roi
    )

    true_positive = float(
        (thin_prediction & thin_target).sum()
    )

    false_positive = float(
        (thin_prediction & ~thin_target).sum()
    )

    false_negative = float(
        (~thin_prediction & thin_target).sum()
    )

    prediction_pixels = (
        true_positive + false_positive
    )

    target_pixels = (
        true_positive + false_negative
    )

    if (
        prediction_pixels <= epsilon
        and target_pixels <= epsilon
    ):
        precision = 1.0
        recall = 1.0
        dice = 1.0

    else:
        precision = _safe_divide(
            true_positive,
            prediction_pixels,
            epsilon,
            empty_value=0.0,
        )

        recall = _safe_divide(
            true_positive,
            target_pixels,
            epsilon,
            empty_value=0.0,
        )

        dice = _safe_divide(
            2.0 * true_positive,
            prediction_pixels + target_pixels,
            epsilon,
            empty_value=0.0,
        )

    return {
        "thin_vessel_dice": dice,
        "thin_vessel_precision": precision,
        "thin_vessel_recall": recall,
        "thin_vessel_f1": _harmonic_mean(
            precision,
            recall,
            epsilon,
        ),
        "thin_target_pixels": int(
            thin_target.sum()
        ),
        "thin_roi_pixels": int(
            thin_roi.sum()
        ),
    }


def evaluate_topology_single(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    config: Optional[
        TopologyMetricConfig
    ] = None,
) -> Dict[str, float]:
    if config is None:
        config = TopologyMetricConfig()

    config.validate()

    prediction_mask = (
        prediction_mask.astype(bool)
    )

    target_mask = target_mask.astype(bool)

    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "prediction and target shapes differ: "
            f"{prediction_mask.shape} versus "
            f"{target_mask.shape}"
        )

    prediction_mask = _remove_small_components(
        prediction_mask,
        minimum_component_size=(
            config.minimum_component_size
        ),
        connectivity=config.connectivity,
    )

    cldice_results = hard_cldice(
        prediction_mask,
        target_mask,
        epsilon=config.epsilon,
    )

    skeleton_results = (
        tolerant_skeleton_metrics(
            prediction_mask,
            target_mask,
            tolerance_px=(
                config.skeleton_tolerance_px
            ),
            epsilon=config.epsilon,
        )
    )

    thin_results = thin_vessel_metrics(
        prediction_mask,
        target_mask,
        thin_radius_px=config.thin_radius_px,
        roi_dilation_px=(
            config.thin_roi_dilation_px
        ),
        epsilon=config.epsilon,
    )

    prediction_components = (
        _component_statistics(
            prediction_mask,
            connectivity=config.connectivity,
            epsilon=config.epsilon,
        )
    )

    target_components = _component_statistics(
        target_mask,
        connectivity=config.connectivity,
        epsilon=config.epsilon,
    )

    prediction_component_count = int(
        prediction_components["component_count"]
    )

    target_component_count = int(
        target_components["component_count"]
    )

    excess_component_count = max(
        0,
        prediction_component_count
        - target_component_count,
    )

    fragmentation_ratio = _safe_divide(
        numerator=float(
            excess_component_count
        ),
        denominator=float(
            max(target_component_count, 1)
        ),
        epsilon=config.epsilon,
        empty_value=0.0,
    )

    result: Dict[str, float] = {}

    result.update(cldice_results)
    result.update(skeleton_results)
    result.update(thin_results)

    result.update(
        {
            "prediction_component_count": (
                prediction_component_count
            ),
            "target_component_count": (
                target_component_count
            ),
            "excess_component_count": (
                excess_component_count
            ),
            "fragmentation_ratio": (
                fragmentation_ratio
            ),
            "prediction_largest_component_size": int(
                prediction_components[
                    "largest_component_size"
                ]
            ),
            "target_largest_component_size": int(
                target_components[
                    "largest_component_size"
                ]
            ),
            "prediction_largest_component_ratio": float(
                prediction_components[
                    "largest_component_ratio"
                ]
            ),
            "target_largest_component_ratio": float(
                target_components[
                    "largest_component_ratio"
                ]
            ),
            "largest_component_ratio_error": abs(
                float(
                    prediction_components[
                        "largest_component_ratio"
                    ]
                )
                - float(
                    target_components[
                        "largest_component_ratio"
                    ]
                )
            ),
            "prediction_foreground_pixels": int(
                prediction_mask.sum()
            ),
            "target_foreground_pixels": int(
                target_mask.sum()
            ),
        }
    )

    return result


def evaluate_topology_batch(
    prediction: ArrayLike,
    target: ArrayLike,
    config: Optional[
        TopologyMetricConfig
    ] = None,
    from_logits: bool = False,
    image_names: Optional[
        Iterable[str]
    ] = None,
) -> Dict[str, Any]:
    """
    Evaluate a batch using equal-weight macro averaging.
    """

    if config is None:
        config = TopologyMetricConfig()

    config.validate()

    prediction_masks = _binarise_prediction(
        prediction,
        threshold=config.prediction_threshold,
        from_logits=from_logits,
    )

    target_masks = _binarise_target(
        target
    )

    if (
        prediction_masks.shape
        != target_masks.shape
    ):
        raise ValueError(
            "prediction and target shapes differ: "
            f"{prediction_masks.shape} versus "
            f"{target_masks.shape}"
        )

    number_of_images = int(
        prediction_masks.shape[0]
    )

    if image_names is None:
        names = [
            f"image_{index:04d}"
            for index in range(number_of_images)
        ]

    else:
        names = [
            str(name)
            for name in image_names
        ]

        if len(names) != number_of_images:
            raise ValueError(
                "image_names length must equal "
                "the batch size."
            )

    per_image: List[
        Dict[str, Any]
    ] = []

    for index in range(number_of_images):
        metrics = evaluate_topology_single(
            prediction_masks[index],
            target_masks[index],
            config=config,
        )

        per_image.append(
            {
                "image_name": names[index],
                **metrics,
            }
        )

    numeric_keys = [
        key
        for key, value in per_image[0].items()
        if (
            key != "image_name"
            and isinstance(
                value,
                (int, float, np.number),
            )
        )
    ]

    mean_metrics = {
        key: float(
            np.mean(
                [
                    float(item[key])
                    for item in per_image
                ]
            )
        )
        for key in numeric_keys
    }

    std_metrics = {
        key: float(
            np.std(
                [
                    float(item[key])
                    for item in per_image
                ],
                ddof=0,
            )
        )
        for key in numeric_keys
    }

    return {
        "config": asdict(config),
        "number_of_images": (
            number_of_images
        ),
        "mean": mean_metrics,
        "std": std_metrics,
        "per_image": per_image,
    }


# ============================================================
# DIFFERENTIABLE SOFT TOPOLOGY FUNCTIONS
# These are not connected to training yet.
# ============================================================


def _validate_soft_tensor(
    tensor: torch.Tensor,
) -> None:
    if not isinstance(
        tensor,
        torch.Tensor,
    ):
        raise TypeError(
            "Expected torch.Tensor."
        )

    if tensor.ndim != 4:
        raise ValueError(
            "Expected [N, 1, H, W]. "
            f"Received {tuple(tensor.shape)}."
        )

    if tensor.shape[1] != 1:
        raise ValueError(
            "Soft topology functions require "
            "one output channel."
        )

    if not tensor.is_floating_point():
        raise TypeError(
            "Soft topology functions require "
            "floating-point tensors."
        )


def soft_erode(
    image: torch.Tensor,
) -> torch.Tensor:
    _validate_soft_tensor(image)

    erosion_height = -F.max_pool2d(
        -image,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0),
    )

    erosion_width = -F.max_pool2d(
        -image,
        kernel_size=(1, 3),
        stride=1,
        padding=(0, 1),
    )

    return torch.minimum(
        erosion_height,
        erosion_width,
    )


def soft_dilate(
    image: torch.Tensor,
) -> torch.Tensor:
    _validate_soft_tensor(image)

    return F.max_pool2d(
        image,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def soft_open(
    image: torch.Tensor,
) -> torch.Tensor:
    return soft_dilate(
        soft_erode(image)
    )


def soft_skeletonize(
    image: torch.Tensor,
    iterations: int = 10,
) -> torch.Tensor:
    _validate_soft_tensor(image)

    if iterations < 1:
        raise ValueError(
            "iterations must be >= 1."
        )

    image = image.clamp(
        0.0,
        1.0,
    )

    opened = soft_open(image)

    skeleton = F.relu(
        image - opened
    )

    for _ in range(iterations):
        image = soft_erode(image)

        opened = soft_open(image)

        delta = F.relu(
            image - opened
        )

        skeleton = (
            skeleton
            + F.relu(
                delta - skeleton * delta
            )
        )

    return skeleton.clamp(
        0.0,
        1.0,
    )


def soft_cldice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 10,
    from_logits: bool = True,
    epsilon: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    _validate_soft_tensor(prediction)
    _validate_soft_tensor(target)

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have "
            "identical shapes."
        )

    if from_logits:
        prediction_probability = torch.sigmoid(
            prediction
        )
    else:
        prediction_probability = prediction.clamp(
            0.0,
            1.0,
        )

    target_probability = target.float().clamp(
        0.0,
        1.0,
    )

    prediction_skeleton = soft_skeletonize(
        prediction_probability,
        iterations=iterations,
    )

    target_skeleton = soft_skeletonize(
        target_probability,
        iterations=iterations,
    )

    spatial_dimensions = tuple(
        range(1, prediction.ndim)
    )

    topology_precision = (
        (
            prediction_skeleton
            * target_probability
        ).sum(dim=spatial_dimensions)
        + epsilon
    ) / (
        prediction_skeleton.sum(
            dim=spatial_dimensions
        )
        + epsilon
    )

    topology_sensitivity = (
        (
            target_skeleton
            * prediction_probability
        ).sum(dim=spatial_dimensions)
        + epsilon
    ) / (
        target_skeleton.sum(
            dim=spatial_dimensions
        )
        + epsilon
    )

    score = (
        2.0
        * topology_precision
        * topology_sensitivity
    ) / (
        topology_precision
        + topology_sensitivity
        + epsilon
    )

    if reduction == "none":
        return score

    if reduction == "mean":
        return score.mean()

    if reduction == "sum":
        return score.sum()

    raise ValueError(
        "reduction must be 'none', "
        "'mean', or 'sum'."
    )


def soft_cldice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 10,
    from_logits: bool = True,
    epsilon: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    score = soft_cldice_score(
        prediction=prediction,
        target=target,
        iterations=iterations,
        from_logits=from_logits,
        epsilon=epsilon,
        reduction=reduction,
    )

    return 1.0 - score


def _self_test() -> None:
    target = np.zeros(
        (1, 64, 64),
        dtype=np.uint8,
    )

    target[0, 32, 8:56] = 1
    target[0, 16:49, 32] = 1

    perfect_prediction = target.copy()

    broken_prediction = target.copy()
    broken_prediction[
        0,
        32,
        29:36,
    ] = 0

    config = TopologyMetricConfig(
        prediction_threshold=0.5,
        skeleton_tolerance_px=1,
        thin_radius_px=2.0,
        thin_roi_dilation_px=2,
    )

    perfect_result = evaluate_topology_batch(
        prediction=perfect_prediction,
        target=target,
        config=config,
        image_names=["perfect"],
    )

    broken_result = evaluate_topology_batch(
        prediction=broken_prediction,
        target=target,
        config=config,
        image_names=["broken"],
    )

    perfect_cldice = perfect_result[
        "mean"
    ]["cldice"]

    broken_cldice = broken_result[
        "mean"
    ]["cldice"]

    assert perfect_cldice > 0.999
    assert broken_cldice < perfect_cldice

    logits = torch.full(
        (2, 1, 64, 64),
        -8.0,
        dtype=torch.float32,
    )

    soft_target = torch.zeros_like(
        logits
    )

    soft_target[:, :, 32, 8:56] = 1.0
    logits[:, :, 32, 8:56] = 8.0

    soft_score = soft_cldice_score(
        prediction=logits,
        target=soft_target,
        iterations=10,
        from_logits=True,
    )

    assert torch.isfinite(
        soft_score
    )

    print(
        "topology_metrics.py self-test passed"
    )

    print(
        f"Perfect hard clDice: "
        f"{perfect_cldice:.6f}"
    )

    print(
        f"Broken hard clDice:  "
        f"{broken_cldice:.6f}"
    )

    print(
        f"Soft clDice:         "
        f"{float(soft_score):.6f}"
    )


if __name__ == "__main__":
    _self_test()