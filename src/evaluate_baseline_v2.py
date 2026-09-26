"""
Fair baseline evaluation for MV-TransUNet using the exact same overlapping
full-resolution patch reconstruction protocol as evaluate_topology_v2.py.

Place this file at:
    src/evaluate_baseline_v2.py

Example:
    python -m src.evaluate_baseline_v2 ^
        --config config.yaml ^
        --checkpoint checkpoints\YOUR_BASELINE_RUN\best_model.pth ^
        --datasets DRIVE_test

TTA:
    python -m src.evaluate_baseline_v2 ^
        --config config.yaml ^
        --checkpoint checkpoints\YOUR_BASELINE_RUN\best_model.pth ^
        --datasets DRIVE_test ^
        --tta ^
        --output-dir evaluation_outputs\baseline_reconstructed_tta
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from skimage.morphology import skeletonize
from tqdm import tqdm

from src.fov_resolution import (
    DEFAULT_FOV_CONFIG,
    DEFAULT_RESIZING_CONFIG,
    calculate_source_reference_diameter,
    compute_adaptive_target_shape,
    list_images,
    resize_image_for_model,
    resize_prediction_to_native,
)
from models.mv_transunet import MVTransUNet
from src.datasets import (
    CLAHEEnhancement,
    load_binary_mask,
    load_rgb_image,
    normalize_filename,
)


PathLike = str | Path

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".ppm", ".gif", ".bmp",
}

IMAGENET_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


# ============================================================
# ARGUMENTS / CONFIGURATION
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline MVTransUNet with overlapping "
            "full-resolution grid-patch reconstruction."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_outputs/baseline_reconstructed",
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Example: --datasets DRIVE_test",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--model-input-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--patch-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--tta",
        action="store_true",
    )

    parser.add_argument(
        "--save-probabilities",
        action="store_true",
    )

    parser.add_argument(
        "--adaptive-resolution",
        action="store_true",
        help=(
            "Enable D2-AR: resize each image so its label-free FOV diameter "
            "matches the DRIVE source domain before reconstruction, then "
            "resize predictions back to native resolution before scoring."
        ),
    )

    parser.add_argument(
        "--adaptive-source-dir",
        type=str,
        default="datasets/DRIVE/training/image",
        help="DRIVE source images used to calibrate the reference FOV diameter.",
    )

    parser.add_argument(
        "--scale-ratio-cache",
        type=str,
        default=None,
        help=(
            "Path to a JSON file mapping image filename -> scale_ratio, "
            "produced by src/precompute_scale_ratios.py. Required when the "
            "checkpoint's config has model.vessel_attention.scale_conditioning "
            "enabled; ignored otherwise."
        ),
    )

    return parser.parse_args()


def load_config(path: PathLike) -> Dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    return config


def seed_everything(
    seed: int,
    deterministic: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(
    config: Mapping[str, Any],
) -> torch.device:
    hardware = config.get("hardware", {})
    requested = str(
        hardware.get("device", "auto")
    ).lower()
    gpu_id = int(
        hardware.get("gpu_id", 0)
    )

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")

    if requested == "cuda":
        print(
            "WARNING: CUDA requested but unavailable; using CPU."
        )

    return torch.device("cpu")


# ============================================================
# DATASET DISCOVERY / PAIRING
# ============================================================


@dataclass(frozen=True)
class DatasetSpecification:
    name: str
    image_dir: Path
    mask_dir: Path


@dataclass(frozen=True)
class SamplePair:
    key: str
    image_path: Path
    mask_path: Path


def supported_files(directory: PathLike) -> List[Path]:
    directory = Path(directory)

    if not directory.is_dir():
        return []

    return sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ],
        key=lambda path: path.name.lower(),
    )


def pair_files(
    image_dir: PathLike,
    mask_dir: PathLike,
) -> List[SamplePair]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir.resolve()}"
        )

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Mask directory not found: {mask_dir.resolve()}"
        )

    images: Dict[str, Path] = {}
    masks: Dict[str, Path] = {}

    for path in supported_files(image_dir):
        key = normalize_filename(path.name)

        if key in images:
            raise RuntimeError(
                f"Duplicate normalized image key: {key}"
            )

        images[key] = path

    for path in supported_files(mask_dir):
        key = normalize_filename(path.name)

        if key in masks:
            raise RuntimeError(
                f"Duplicate normalized mask key: {key}"
            )

        masks[key] = path

    keys = sorted(set(images).intersection(masks))

    if not keys:
        raise RuntimeError(
            "No image-mask pairs were found after "
            "filename normalization."
        )

    return [
        SamplePair(
            key=key,
            image_path=images[key],
            mask_path=masks[key],
        )
        for key in keys
    ]


def add_dataset_if_available(
    result: Dict[str, DatasetSpecification],
    name: str,
    image_dir: PathLike,
    mask_dir: PathLike,
) -> None:
    image_dir = Path(str(image_dir))
    mask_dir = Path(str(mask_dir))

    if image_dir.is_dir() and mask_dir.is_dir():
        result[name] = DatasetSpecification(
            name=name,
            image_dir=image_dir,
            mask_dir=mask_dir,
        )


def resolve_datasets(
    config: Mapping[str, Any],
    requested: Optional[Sequence[str]],
) -> Dict[str, DatasetSpecification]:
    result: Dict[str, DatasetSpecification] = {}
    dataset_config = config.get("dataset", {})

    test_dataset = dataset_config.get("test_dataset")

    if isinstance(test_dataset, Mapping):
        add_dataset_if_available(
            result=result,
            name=str(
                test_dataset.get("name", "DRIVE_test")
            ),
            image_dir=test_dataset.get(
                "image_dir",
                "datasets_processed/DRIVE_test/images",
            ),
            mask_dir=test_dataset.get(
                "mask_dir",
                "datasets_processed/DRIVE_test/masks",
            ),
        )

    external = dataset_config.get("external_datasets", {})

    if isinstance(external, Mapping):
        for name, entry in external.items():
            if not isinstance(entry, Mapping):
                continue

            add_dataset_if_available(
                result=result,
                name=str(name),
                image_dir=entry.get("image_dir", ""),
                mask_dir=entry.get("mask_dir", ""),
            )

    defaults = {
        "DRIVE_test": (
            "datasets_processed/DRIVE_test/images",
            "datasets_processed/DRIVE_test/masks",
        ),
        "STARE": (
            "datasets_processed/STARE/images",
            "datasets_processed/STARE/masks",
        ),
        "CHASE_DB1": (
            "datasets_processed/CHASE_DB1/images",
            "datasets_processed/CHASE_DB1/masks",
        ),
        "HRF": (
            "datasets_processed/HRF/images",
            "datasets_processed/HRF/masks",
        ),
    }

    for name, (image_dir, mask_dir) in defaults.items():
        if name not in result:
            add_dataset_if_available(
                result=result,
                name=name,
                image_dir=image_dir,
                mask_dir=mask_dir,
            )

    if requested:
        requested_lower = {
            name.lower()
            for name in requested
        }

        result = {
            name: specification
            for name, specification in result.items()
            if name.lower() in requested_lower
        }

    if not result:
        raise RuntimeError(
            "No requested evaluation datasets were found."
        )

    return result


# ============================================================
# MODEL / CHECKPOINT
# ============================================================


def build_model(
    config: Mapping[str, Any],
) -> MVTransUNet:
    model_config = config.get("model", {})
    deep_supervision_config = model_config.get(
        "deep_supervision",
        {},
    )

    constructor_arguments = {
        "pretrained": False,
        "transformer_channels": int(
            model_config.get(
                "transformer",
                {},
            ).get(
                "embed_dim",
                768,
            )
        ),
        "vessel_reduction_ratio": int(
            model_config.get(
                "vessel_attention",
                {},
            ).get(
                "reduction_ratio",
                16,
            )
        ),
        "output_channels": int(
            model_config.get("output_channels", 1)
        ),
        "deep_supervision": bool(
            deep_supervision_config.get(
                "enabled",
                True,
            )
        ),
        "vessel_attention_scale_conditioning": bool(
            model_config.get(
                "vessel_attention",
                {},
            ).get(
                "scale_conditioning",
                False,
            )
        ),
    }

    # Newer project versions accept decoder_dropout_rate. Older ones do
    # not. This keeps evaluation compatible while preserving state_dict
    # compatibility, because Dropout2d has no learnable parameters.
    decoder_dropout_rate = float(
        model_config.get(
            "decoder",
            {},
        ).get(
            "dropout_rate",
            0.1,
        )
    )

    try:
        return MVTransUNet(
            **constructor_arguments,
            decoder_dropout_rate=decoder_dropout_rate,
        )
    except TypeError as error:
        if "decoder_dropout_rate" not in str(error):
            raise

        return MVTransUNet(
            **constructor_arguments,
        )


def resolve_checkpoint(
    config: Mapping[str, Any],
    explicit_path: Optional[str],
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
    else:
        checkpoint = config.get("checkpoint", {})
        directory = Path(
            str(
                checkpoint.get(
                    "directory",
                    "checkpoints",
                )
            )
        )
        filename = str(
            checkpoint.get(
                "best_model_name",
                "best_model.pth",
            )
        )
        path = directory / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}"
        )

    return path


def load_checkpoint(
    model: MVTransUNet,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, Mapping):
        if "model_state" in checkpoint:
            state = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        raise TypeError(
            "Checkpoint must contain a state dictionary."
        )

    if all(
        str(key).startswith("module.")
        for key in state
    ):
        state = {
            str(key)[7:]: value
            for key, value in state.items()
        }

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()
    return dict(checkpoint)


# ============================================================
# PATCH PREPROCESSING / GEOMETRY
# ============================================================


def normalize_rgb_patch(
    patch_rgb: np.ndarray,
) -> torch.Tensor:
    patch = patch_rgb.astype(
        np.float32
    ) / 255.0

    patch = (
        patch - IMAGENET_MEAN
    ) / IMAGENET_STD

    patch = np.transpose(
        patch,
        (2, 0, 1),
    )

    return torch.from_numpy(
        np.ascontiguousarray(patch)
    ).float()


def positions_for_axis(
    length: int,
    patch_size: int,
    stride: int,
) -> List[int]:
    if length <= patch_size:
        return [0]

    positions = list(
        range(
            0,
            length - patch_size + 1,
            stride,
        )
    )

    final_position = length - patch_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


def pad_to_patch_size(
    image: np.ndarray,
    patch_size: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = image.shape[:2]

    padded_height = max(height, patch_size)
    padded_width = max(width, patch_size)

    padded = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=padded_height - height,
        left=0,
        right=padded_width - width,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return padded, (height, width)


@dataclass(frozen=True)
class PatchLocation:
    top: int
    left: int


def build_patch_grid(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> List[PatchLocation]:
    tops = positions_for_axis(
        height,
        patch_size,
        stride,
    )

    lefts = positions_for_axis(
        width,
        patch_size,
        stride,
    )

    return [
        PatchLocation(top=top, left=left)
        for top in tops
        for left in lefts
    ]


# ============================================================
# INFERENCE
# ============================================================


def flip_tensor(
    tensor: torch.Tensor,
    horizontal: bool,
    vertical: bool,
) -> torch.Tensor:
    dimensions: List[int] = []

    if vertical:
        dimensions.append(-2)

    if horizontal:
        dimensions.append(-1)

    if not dimensions:
        return tensor

    return torch.flip(tensor, dimensions)


def extract_main_logits(
    output: Any,
) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, Mapping):
        for key in (
            "main_output",
            "segmentation_logits",
            "logits",
            "output",
        ):
            if key in output:
                value = output[key]

                if isinstance(value, torch.Tensor):
                    return value

    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]

    raise TypeError(
        "Could not extract baseline segmentation logits "
        f"from output type {type(output).__name__}."
    )


@torch.inference_mode()
def predict_patch_batch(
    model: MVTransUNet,
    patches: torch.Tensor,
    amp_enabled: bool,
    use_tta: bool,
    scale_ratio: float = 1.0,
) -> np.ndarray:
    variants = [(False, False)]

    if use_tta:
        variants.extend(
            [
                (True, False),
                (False, True),
                (True, True),
            ]
        )

    accumulated: Optional[torch.Tensor] = None

    for horizontal, vertical in variants:
        model_input = flip_tensor(
            patches,
            horizontal=horizontal,
            vertical=vertical,
        )

        with torch.autocast(
            device_type=patches.device.type,
            enabled=amp_enabled,
        ):
            raw_output = model(model_input, scale_ratio=scale_ratio)

        probability = torch.sigmoid(
            extract_main_logits(raw_output).float()
        )

        probability = flip_tensor(
            probability,
            horizontal=horizontal,
            vertical=vertical,
        )

        if accumulated is None:
            accumulated = probability
        else:
            accumulated += probability

    assert accumulated is not None

    return (
        accumulated / float(len(variants))
    ).detach().cpu().numpy()


@torch.inference_mode()
def reconstruct_image_probability(
    model: MVTransUNet,
    image_rgb: np.ndarray,
    device: torch.device,
    patch_size: int,
    model_input_size: int,
    stride: int,
    patch_batch_size: int,
    amp_enabled: bool,
    use_tta: bool,
    scale_ratio: float = 1.0,
) -> np.ndarray:
    if patch_batch_size < 1:
        raise ValueError(
            "patch_batch_size must be at least 1."
        )

    padded_image, original_shape = pad_to_patch_size(
        image=image_rgb,
        patch_size=patch_size,
    )

    padded_height, padded_width = (
        padded_image.shape[:2]
    )

    locations = build_patch_grid(
        height=padded_height,
        width=padded_width,
        patch_size=patch_size,
        stride=stride,
    )

    probability_sum = np.zeros(
        (padded_height, padded_width),
        dtype=np.float32,
    )

    count = np.zeros(
        (padded_height, padded_width),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(locations),
        patch_batch_size,
    ):
        batch_locations = locations[
            start:start + patch_batch_size
        ]

        tensors: List[torch.Tensor] = []

        for location in batch_locations:
            patch = padded_image[
                location.top:
                location.top + patch_size,
                location.left:
                location.left + patch_size,
            ]

            if patch_size != model_input_size:
                patch_for_model = cv2.resize(
                    patch,
                    (model_input_size, model_input_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                patch_for_model = patch

            tensors.append(
                normalize_rgb_patch(patch_for_model)
            )

        patch_batch = torch.stack(
            tensors,
            dim=0,
        ).to(
            device,
            non_blocking=True,
        )

        batch_probability = predict_patch_batch(
            model=model,
            patches=patch_batch,
            amp_enabled=amp_enabled,
            use_tta=use_tta,
            scale_ratio=scale_ratio,
        )

        for local_index, location in enumerate(
            batch_locations
        ):
            patch_probability = batch_probability[
                local_index,
                0,
            ]

            if model_input_size != patch_size:
                patch_probability = cv2.resize(
                    patch_probability,
                    (patch_size, patch_size),
                    interpolation=cv2.INTER_LINEAR,
                )

            probability_sum[
                location.top:
                location.top + patch_size,
                location.left:
                location.left + patch_size,
            ] += patch_probability

            count[
                location.top:
                location.top + patch_size,
                location.left:
                location.left + patch_size,
            ] += 1.0

    if np.any(count <= 0):
        raise RuntimeError(
            "Patch grid left uncovered pixels."
        )

    original_height, original_width = original_shape

    return (
        probability_sum / count
    )[
        :original_height,
        :original_width,
    ]


# ============================================================
# METRICS
# ============================================================


def safe_divide(
    numerator: float,
    denominator: float,
    epsilon: float = 1e-7,
) -> float:
    return float(
        numerator / (denominator + epsilon)
    )


def binary_skeleton(
    mask: np.ndarray,
) -> np.ndarray:
    return skeletonize(
        mask.astype(bool)
    ).astype(np.uint8)


def dilate(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    size = 2 * radius + 1

    return cv2.dilate(
        mask.astype(np.uint8),
        np.ones(
            (size, size),
            dtype=np.uint8,
        ),
        iterations=1,
    )


def cldice_score(
    prediction: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-7,
) -> float:
    predicted_skeleton = binary_skeleton(
        prediction
    ).astype(bool)

    target_skeleton = binary_skeleton(
        target
    ).astype(bool)

    topology_precision = (
        np.logical_and(
            predicted_skeleton,
            target.astype(bool),
        ).sum()
        + epsilon
    ) / (
        predicted_skeleton.sum()
        + epsilon
    )

    topology_sensitivity = (
        np.logical_and(
            target_skeleton,
            prediction.astype(bool),
        ).sum()
        + epsilon
    ) / (
        target_skeleton.sum()
        + epsilon
    )

    return float(
        (
            2.0
            * topology_precision
            * topology_sensitivity
        )
        / (
            topology_precision
            + topology_sensitivity
            + epsilon
        )
    )


def thin_vessel_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    width_threshold: float = 3.0,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    target_uint8 = target.astype(np.uint8)

    distance = cv2.distanceTransform(
        target_uint8,
        cv2.DIST_L2,
        5,
    )

    local_width = distance * 2.0

    target_skeleton = binary_skeleton(
        target_uint8
    ).astype(bool)

    thin_target = np.logical_and(
        target_skeleton,
        local_width <= width_threshold,
    )

    predicted_skeleton = binary_skeleton(
        prediction
    ).astype(bool)

    target_tolerance = dilate(
        thin_target,
        radius=1,
    ).astype(bool)

    prediction_tolerance = dilate(
        predicted_skeleton,
        radius=1,
    ).astype(bool)

    matched_prediction = np.logical_and(
        predicted_skeleton,
        target_tolerance,
    ).sum()

    matched_target = np.logical_and(
        thin_target,
        prediction_tolerance,
    ).sum()

    precision = (
        matched_prediction + epsilon
    ) / (
        predicted_skeleton.sum() + epsilon
    )

    recall = (
        matched_target + epsilon
    ) / (
        thin_target.sum() + epsilon
    )

    dice = (
        2.0 * precision * recall
    ) / (
        precision + recall + epsilon
    )

    return {
        "thin_vessel_precision": float(precision),
        "thin_vessel_recall": float(recall),
        "thin_vessel_dice": float(dice),
    }


def segmentation_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    prediction = (
        probability >= threshold
    ).astype(np.uint8)

    target = target.astype(np.uint8)

    prediction_bool = prediction.astype(bool)
    target_bool = target.astype(bool)

    true_positive = float(
        np.logical_and(
            prediction_bool,
            target_bool,
        ).sum()
    )

    true_negative = float(
        np.logical_and(
            ~prediction_bool,
            ~target_bool,
        ).sum()
    )

    false_positive = float(
        np.logical_and(
            prediction_bool,
            ~target_bool,
        ).sum()
    )

    false_negative = float(
        np.logical_and(
            ~prediction_bool,
            target_bool,
        ).sum()
    )

    result = {
        "dice": safe_divide(
            2.0 * true_positive,
            (
                2.0 * true_positive
                + false_positive
                + false_negative
            ),
        ),
        "iou": safe_divide(
            true_positive,
            (
                true_positive
                + false_positive
                + false_negative
            ),
        ),
        "sensitivity": safe_divide(
            true_positive,
            true_positive + false_negative,
        ),
        "specificity": safe_divide(
            true_negative,
            true_negative + false_positive,
        ),
        "accuracy": safe_divide(
            true_positive + true_negative,
            (
                true_positive
                + true_negative
                + false_positive
                + false_negative
            ),
        ),
        "precision": safe_divide(
            true_positive,
            true_positive + false_positive,
        ),
        "cldice": cldice_score(
            prediction,
            target,
        ),
    }

    if np.unique(target).size > 1:
        result["roc_auc"] = float(
            roc_auc_score(
                target.reshape(-1),
                probability.reshape(-1),
            )
        )
    else:
        result["roc_auc"] = float("nan")

    result.update(
        thin_vessel_metrics(
            prediction,
            target,
        )
    )

    return result


# ============================================================
# OUTPUT
# ============================================================


def save_gray(
    path: Path,
    array: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if array.dtype == bool:
        output = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        output = (
            np.clip(array, 0.0, 1.0)
            * 255.0
        ).round().astype(np.uint8)
    else:
        output = np.clip(
            array,
            0,
            255,
        ).astype(np.uint8)

    if not cv2.imwrite(str(path), output):
        raise IOError(
            f"Could not save: {path.resolve()}"
        )


def save_comparison(
    path: Path,
    image_rgb: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> None:
    target_rgb = cv2.cvtColor(
        target.astype(np.uint8) * 255,
        cv2.COLOR_GRAY2RGB,
    )

    probability_rgb = cv2.cvtColor(
        cv2.applyColorMap(
            (
                np.clip(probability, 0.0, 1.0)
                * 255
            ).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS,
        ),
        cv2.COLOR_BGR2RGB,
    )

    prediction_rgb = cv2.cvtColor(
        prediction.astype(np.uint8) * 255,
        cv2.COLOR_GRAY2RGB,
    )

    comparison = np.concatenate(
        [
            image_rgb,
            target_rgb,
            probability_rgb,
            prediction_rgb,
        ],
        axis=1,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(path),
        cv2.cvtColor(
            comparison,
            cv2.COLOR_RGB2BGR,
        ),
    ):
        raise IOError(
            f"Could not save comparison: {path.resolve()}"
        )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(
        value,
        (np.floating, np.integer),
    ):
        value = value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            {
                key: json_safe(value)
                for key, value in payload.items()
            },
            stream,
            indent=2,
            sort_keys=True,
        )


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    excluded = {
        "dataset",
        "sample",
        "image_path",
        "mask_path",
    }

    metric_names = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in excluded
        }
    )

    summary: Dict[str, float] = {}

    for metric in metric_names:
        values = np.asarray(
            [
                float(row[metric])
                for row in rows
                if metric in row
            ],
            dtype=np.float64,
        )

        values = values[np.isfinite(values)]

        summary[metric] = (
            float(values.mean())
            if values.size
            else float("nan")
        )

        summary[f"{metric}_std"] = (
            float(values.std(ddof=0))
            if values.size
            else float("nan")
        )

    return summary


DISPLAY_METRICS = (
    "dice",
    "iou",
    "sensitivity",
    "specificity",
    "accuracy",
    "precision",
    "roc_auc",
    "cldice",
    "thin_vessel_dice",
    "thin_vessel_recall",
)


def print_summary(
    summary: Mapping[str, Any],
) -> None:
    print()
    print("=" * 78)
    print(
        f"Results: {summary['dataset']} "
        f"({summary['number_of_images']} images)"
    )
    print("=" * 78)

    for metric in DISPLAY_METRICS:
        if metric not in summary:
            continue

        mean = float(summary[metric])
        standard_deviation = float(
            summary.get(
                f"{metric}_std",
                float("nan"),
            )
        )

        if math.isfinite(mean):
            print(
                f"{metric:24s}: "
                f"{mean:.6f} ± {standard_deviation:.6f}"
            )
        else:
            print(f"{metric:24s}: N/A")


# ============================================================
# EVALUATION
# ============================================================


def evaluate_dataset(
    model: MVTransUNet,
    specification: DatasetSpecification,
    device: torch.device,
    output_root: Path,
    threshold: float,
    patch_size: int,
    model_input_size: int,
    stride: int,
    patch_batch_size: int,
    amp_enabled: bool,
    use_tta: bool,
    clahe_enabled: bool,
    save_probabilities: bool,
    adaptive_resolution_enabled: bool = False,
    source_reference_diameter: Optional[float] = None,
    scale_ratio_lookup: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    pairs = pair_files(
        specification.image_dir,
        specification.mask_dir,
    )

    dataset_root = output_root / specification.name

    clahe = (
        CLAHEEnhancement()
        if clahe_enabled
        else None
    )

    rows: List[Dict[str, Any]] = []

    progress = tqdm(
        pairs,
        desc=f"Reconstructing {specification.name}",
    )

    for pair in progress:
        image = load_rgb_image(pair.image_path)

        target = (
            load_binary_mask(pair.mask_path) > 0.5
        ).astype(np.uint8)

        if image.shape[:2] != target.shape[:2]:
            raise RuntimeError(
                f"Shape mismatch for {pair.key}: "
                f"image={image.shape[:2]}, "
                f"mask={target.shape[:2]}"
            )

        native_height, native_width = image.shape[:2]
        scale_info: Optional[Dict[str, Any]] = None

        if scale_ratio_lookup is not None:
            # Namespaced by dataset name: these datasets reuse generic
            # sequential filenames (e.g. '002.jpg' exists in both CHASE_DB1
            # and HRF), so a bare filename key would silently collide across
            # datasets. Must match src/precompute_scale_ratios.py's --dataset-name.
            image_key = f"{specification.name}/{pair.image_path.name}"
            if image_key not in scale_ratio_lookup:
                raise KeyError(
                    f"No precomputed scale_ratio for image '{image_key}' in "
                    f"{specification.name}. Re-run "
                    "src/precompute_scale_ratios.py for this dataset's "
                    "image directory before evaluating a scale-conditioned "
                    "checkpoint."
                )
            image_scale_ratio = float(scale_ratio_lookup[image_key])
        else:
            image_scale_ratio = 1.0

        if adaptive_resolution_enabled:
            scale_info = compute_adaptive_target_shape(
                image_rgb=image,
                source_reference_diameter=source_reference_diameter,
                fov_config=DEFAULT_FOV_CONFIG,
                resizing_config=DEFAULT_RESIZING_CONFIG,
            )
            image_for_reconstruction = resize_image_for_model(
                image,
                scale_info["new_height"],
                scale_info["new_width"],
            )
        else:
            image_for_reconstruction = image

        image_for_model = (
            clahe(image_for_reconstruction)
            if clahe is not None
            else image_for_reconstruction
        )

        probability = reconstruct_image_probability(
            model=model,
            image_rgb=image_for_model,
            device=device,
            patch_size=patch_size,
            model_input_size=model_input_size,
            stride=stride,
            patch_batch_size=patch_batch_size,
            amp_enabled=amp_enabled,
            use_tta=use_tta,
            scale_ratio=image_scale_ratio,
        )

        if adaptive_resolution_enabled:
            probability = resize_prediction_to_native(
                probability,
                native_height,
                native_width,
            )

        prediction = (
            probability >= threshold
        ).astype(np.uint8)

        row: Dict[str, Any] = {
            "dataset": specification.name,
            "sample": pair.key,
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
        }

        if scale_info is not None:
            row["adaptive_resolution_final_scale_factor"] = scale_info["final_scale_factor"]
            row["adaptive_resolution_target_fov_diameter"] = scale_info["target_fov_equivalent_diameter"]
            row["adaptive_resolution_scale_was_clamped"] = scale_info["scale_was_clamped"]

        row.update(
            segmentation_metrics(
                probability=probability,
                target=target,
                threshold=threshold,
            )
        )

        rows.append(row)

        save_gray(
            dataset_root
            / "predictions"
            / "segmentation"
            / f"{pair.key}.png",
            prediction,
        )

        if save_probabilities:
            save_gray(
                dataset_root
                / "predictions"
                / "probability"
                / f"{pair.key}.png",
                probability,
            )

        save_comparison(
            dataset_root
            / "comparisons"
            / f"{pair.key}.png",
            image_rgb=image,
            target=target,
            probability=probability,
            prediction=prediction,
        )

        progress.set_postfix(
            dice=f"{row['dice']:.4f}",
            cldice=f"{row['cldice']:.4f}",
        )

    summary: Dict[str, Any] = summarize_rows(rows)

    summary.update(
        {
            "dataset": specification.name,
            "number_of_images": len(rows),
            "threshold": threshold,
            "patch_size": patch_size,
            "model_input_size": model_input_size,
            "stride": stride,
            "patch_batch_size": patch_batch_size,
            "tta_enabled": use_tta,
            "evaluation_protocol": (
                "overlapping_full_resolution_patch_reconstruction"
            ),
            "model": "MVTransUNet_baseline",
            "adaptive_resolution_enabled": adaptive_resolution_enabled,
            "adaptive_resolution_source_reference_diameter": source_reference_diameter,
        }
    )

    write_csv(
        dataset_root / "per_image_metrics.csv",
        rows,
    )

    write_json(
        dataset_root / "summary_metrics.json",
        summary,
    )

    return summary


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    seed_config = config.get("seed", {})

    seed_everything(
        seed=int(seed_config.get("value", 42)),
        deterministic=bool(
            seed_config.get("deterministic", True)
        ),
    )

    device = resolve_device(config)

    hardware = config.get("hardware", {})

    amp_enabled = bool(
        hardware.get("mixed_precision", True)
        and device.type == "cuda"
    )

    patch_config = config.get(
        "patch_training",
        {},
    )

    patch_size = int(
        arguments.patch_size
        if arguments.patch_size is not None
        else patch_config.get(
            "patch_size",
            256,
        )
    )

    model_input_size = int(
        arguments.model_input_size
        if arguments.model_input_size is not None
        else patch_config.get(
            "model_input_size",
            patch_size,
        )
    )

    stride = int(
        arguments.stride
        if arguments.stride is not None
        else patch_config.get(
            "validation_stride",
            max(1, patch_size // 2),
        )
    )

    if patch_size < 1:
        raise ValueError(
            "patch_size must be positive."
        )

    if model_input_size < 1:
        raise ValueError(
            "model_input_size must be positive."
        )

    if stride < 1 or stride > patch_size:
        raise ValueError(
            "stride must be between 1 and patch_size."
        )

    threshold = float(
        arguments.threshold
        if arguments.threshold is not None
        else config.get(
            "evaluation",
            {},
        ).get(
            "threshold",
            config.get(
                "validation",
                {},
            ).get(
                "threshold",
                0.5,
            ),
        )
    )

    checkpoint_path = resolve_checkpoint(
        config=config,
        explicit_path=arguments.checkpoint,
    )

    model = build_model(config).to(device)

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    scale_ratio_lookup: Optional[Dict[str, float]] = None
    if bool(getattr(model, "vessel_attention_scale_conditioning", False)):
        if not arguments.scale_ratio_cache:
            raise ValueError(
                "This checkpoint's config has "
                "model.vessel_attention.scale_conditioning enabled, so "
                "--scale-ratio-cache <path> is required (see "
                "src/precompute_scale_ratios.py)."
            )
        with open(arguments.scale_ratio_cache, "r", encoding="utf-8") as stream:
            scale_ratio_lookup = {
                str(key): float(value)
                for key, value in json.load(stream).items()
            }
        print(
            "Scale-ratio cache:",
            Path(arguments.scale_ratio_cache).resolve(),
            f"({len(scale_ratio_lookup)} images)",
        )

    datasets = resolve_datasets(
        config=config,
        requested=arguments.datasets,
    )

    clahe_enabled = bool(
        config.get(
            "preprocessing",
            {},
        ).get(
            "clahe",
            {},
        ).get(
            "enabled",
            True,
        )
    )

    output_root = Path(arguments.output_dir)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("MV-TransUNet Baseline Reconstructed Evaluation")
    print("=" * 78)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device.index or 0
            ),
        )

    print("Mixed precision:", amp_enabled)
    print("Checkpoint:", checkpoint_path.resolve())
    print(
        "Saved epoch:",
        checkpoint.get("epoch", "unknown"),
    )
    print(
        "Saved best patch Dice:",
        checkpoint.get("best_dice", "unknown"),
    )
    print("Patch size:", patch_size)
    print("Model input size:", model_input_size)
    print("Stride:", stride)
    print("Patch batch size:", arguments.patch_batch_size)
    print("Threshold:", threshold)
    print("CLAHE:", clahe_enabled)
    print("TTA:", bool(arguments.tta))
    print("Datasets:", ", ".join(datasets.keys()))
    print("Output directory:", output_root.resolve())

    adaptive_resolution_enabled = bool(arguments.adaptive_resolution)
    source_reference_diameter: Optional[float] = None

    if adaptive_resolution_enabled:
        source_images = list_images(Path(arguments.adaptive_source_dir))
        (
            source_reference_diameter,
            source_calibration_records,
        ) = calculate_source_reference_diameter(
            source_images=source_images,
            fov_config=DEFAULT_FOV_CONFIG,
        )
        with (
            output_root / "adaptive_resolution_source_calibration.json"
        ).open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "source_reference_fov_equivalent_diameter": source_reference_diameter,
                    "source_image_directory": str(
                        Path(arguments.adaptive_source_dir).resolve()
                    ),
                    "source_image_count": len(source_images),
                    "per_image": source_calibration_records,
                },
                stream,
                indent=2,
            )

    print("Adaptive resolution normalization (D2-AR):", adaptive_resolution_enabled)

    summaries: List[Dict[str, Any]] = []

    for specification in datasets.values():
        print()
        print("-" * 78)
        print(f"Preparing {specification.name}")
        print("Images:", specification.image_dir.resolve())
        print("Masks:", specification.mask_dir.resolve())

        number_of_pairs = len(
            pair_files(
                specification.image_dir,
                specification.mask_dir,
            )
        )

        print("Matched images:", number_of_pairs)

        summary = evaluate_dataset(
            model=model,
            specification=specification,
            device=device,
            output_root=output_root,
            threshold=threshold,
            scale_ratio_lookup=scale_ratio_lookup,
            patch_size=patch_size,
            model_input_size=model_input_size,
            stride=stride,
            patch_batch_size=int(
                arguments.patch_batch_size
            ),
            amp_enabled=amp_enabled,
            use_tta=bool(arguments.tta),
            clahe_enabled=clahe_enabled,
            save_probabilities=bool(
                arguments.save_probabilities
            ),
            adaptive_resolution_enabled=adaptive_resolution_enabled,
            source_reference_diameter=source_reference_diameter,
        )

        summaries.append(summary)
        print_summary(summary)

    write_csv(
        output_root / "all_dataset_summary.csv",
        summaries,
    )

    with (
        output_root / "all_dataset_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            {
                str(summary["dataset"]): {
                    key: json_safe(value)
                    for key, value in summary.items()
                }
                for summary in summaries
            },
            stream,
            indent=2,
            sort_keys=True,
        )

    print()
    print("=" * 78)
    print("Baseline reconstructed evaluation complete")
    print("=" * 78)
    print(
        "Combined summary:",
        (
            output_root
            / "all_dataset_summary.csv"
        ).resolve(),
    )


if __name__ == "__main__":
    main()