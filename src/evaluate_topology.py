"""
Topology-aware evaluation pipeline for MV-TransUNetTopology.

This evaluator is intentionally separate from src/evaluate.py so that:

1. Baseline MV-TransUNet evaluation remains unchanged.
2. Topology checkpoints are loaded into the correct architecture.
3. Segmentation and topology-specific metrics are reported together.
4. DRIVE_test remains the official held-out literature-comparison split.
5. STARE, CHASE_DB1, and HRF can be evaluated as zero-shot datasets.

Expected checkpoint:
    checkpoints/topology_full_seed42/best_model.pth

Typical command:
    python -m src.evaluate_topology ^
        --config config_topology.yaml ^
        --checkpoint checkpoints/topology_full_seed42/best_model.pth

The script evaluates every dataset whose image and mask directories exist.
Results are written to:
    evaluation_outputs/topology_full_seed42/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml

from sklearn.metrics import roc_auc_score
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.mv_transunet_topology import MVTransUNetTopology
from src.datasets import (
    CLAHEEnhancement,
    get_validation_transform,
    load_binary_mask,
    load_rgb_image,
    normalize_filename,
)


PathLike = str | Path


# ============================================================
# CONFIGURATION AND REPRODUCIBILITY
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an MVTransUNetTopology checkpoint."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config_topology.yaml",
        help="Topology training configuration file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path. When omitted, checkpoint.directory and "
            "checkpoint.best_model_name are read from the configuration."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_outputs/topology_full_seed42",
        help="Directory used for metrics and prediction images.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Segmentation threshold. When omitted, validation.threshold "
            "or evaluation.threshold is used, falling back to 0.5."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size. Batch size 1 is recommended locally.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Zero is safest on Windows.",
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable flip test-time augmentation.",
    )

    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save 8-bit probability maps in addition to binary predictions.",
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Optional dataset names to evaluate, for example DRIVE_test STARE. "
            "By default, every available configured/default dataset is used."
        ),
    )

    return parser.parse_args()


def load_config(path: PathLike) -> Dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration is empty or invalid: {config_path.resolve()}"
        )

    return config


def seed_everything(
    seed: int,
    deterministic: bool = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_device(
    config: Mapping[str, Any],
) -> torch.device:
    hardware = config.get("hardware", {})
    requested = str(hardware.get("device", "auto")).strip().lower()
    gpu_id = int(hardware.get("gpu_id", 0))

    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            "hardware.device must be one of: auto, cuda, cpu."
        )

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid hardware.gpu_id={gpu_id}; "
                f"detected {torch.cuda.device_count()} CUDA device(s)."
            )

        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")

    if requested == "cuda":
        print(
            "WARNING: CUDA was requested but is unavailable. "
            "Evaluation will continue on CPU."
        )

    return torch.device("cpu")


# ============================================================
# DATASET RESOLUTION
# ============================================================


@dataclass(frozen=True)
class DatasetSpecification:
    name: str
    image_dir: Path
    mask_dir: Path


@dataclass(frozen=True)
class EvaluationSample:
    key: str
    image_path: Path
    mask_path: Path


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".ppm",
    ".gif",
    ".bmp",
}


def get_supported_files(
    directory: PathLike,
) -> List[Path]:
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


def pair_image_mask_files(
    image_dir: PathLike,
    mask_dir: PathLike,
) -> List[EvaluationSample]:
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

    image_files = get_supported_files(image_dir)
    mask_files = get_supported_files(mask_dir)

    if not image_files:
        raise RuntimeError(
            f"No supported images found in {image_dir.resolve()}."
        )

    if not mask_files:
        raise RuntimeError(
            f"No supported masks found in {mask_dir.resolve()}."
        )

    image_dictionary: Dict[str, Path] = {}

    for image_path in image_files:
        key = normalize_filename(image_path.name)

        if key in image_dictionary:
            raise RuntimeError(
                f"Duplicate normalized image key '{key}'."
            )

        image_dictionary[key] = image_path

    mask_dictionary: Dict[str, Path] = {}

    for mask_path in mask_files:
        key = normalize_filename(mask_path.name)

        if key in mask_dictionary:
            raise RuntimeError(
                f"Duplicate normalized mask key '{key}'."
            )

        mask_dictionary[key] = mask_path

    shared_keys = sorted(
        set(image_dictionary).intersection(mask_dictionary)
    )

    if not shared_keys:
        raise RuntimeError(
            "No image-mask pairs matched after filename normalization.\n"
            f"Images: {image_dir.resolve()}\n"
            f"Masks:  {mask_dir.resolve()}"
        )

    missing_masks = sorted(
        set(image_dictionary).difference(mask_dictionary)
    )

    missing_images = sorted(
        set(mask_dictionary).difference(image_dictionary)
    )

    if missing_masks:
        print(
            f"WARNING: {len(missing_masks)} image(s) have no mask in "
            f"{image_dir.name}."
        )

    if missing_images:
        print(
            f"WARNING: {len(missing_images)} mask(s) have no image in "
            f"{mask_dir.name}."
        )

    return [
        EvaluationSample(
            key=key,
            image_path=image_dictionary[key],
            mask_path=mask_dictionary[key],
        )
        for key in shared_keys
    ]


class TopologyEvaluationDataset(Dataset):
    """Whole-image evaluation dataset using the training preprocessing."""

    def __init__(
        self,
        image_dir: PathLike,
        mask_dir: PathLike,
        image_size: int,
        clahe_enabled: bool,
    ) -> None:
        self.samples = pair_image_mask_files(
            image_dir=image_dir,
            mask_dir=mask_dir,
        )

        self.transform = get_validation_transform(
            image_size=image_size
        )

        self.clahe = (
            CLAHEEnhancement()
            if clahe_enabled
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        sample = self.samples[index]

        image = load_rgb_image(sample.image_path)
        mask = load_binary_mask(sample.mask_path)

        original_height, original_width = mask.shape[:2]

        if self.clahe is not None:
            image = self.clahe(image)

        transformed = self.transform(
            image=image,
            mask=mask,
        )

        image_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"]

        if not torch.is_tensor(mask_tensor):
            mask_tensor = torch.as_tensor(mask_tensor)

        mask_tensor = mask_tensor.float()

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        elif (
            mask_tensor.ndim == 3
            and mask_tensor.shape[-1] == 1
        ):
            mask_tensor = mask_tensor.permute(2, 0, 1)

        mask_tensor = (
            mask_tensor > 0.5
        ).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "key": sample.key,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            "original_height": int(original_height),
            "original_width": int(original_width),
        }


def build_loader(
    specification: DatasetSpecification,
    image_size: int,
    batch_size: int,
    num_workers: int,
    clahe_enabled: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = TopologyEvaluationDataset(
        image_dir=specification.image_dir,
        mask_dir=specification.mask_dir,
        image_size=image_size,
        clahe_enabled=clahe_enabled,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(
            num_workers > 0
        ),
    )


def _add_dataset_if_valid(
    specifications: Dict[str, DatasetSpecification],
    name: str,
    image_dir: PathLike,
    mask_dir: PathLike,
) -> None:
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)

    if image_path.is_dir() and mask_path.is_dir():
        specifications[name] = DatasetSpecification(
            name=name,
            image_dir=image_path,
            mask_dir=mask_path,
        )


def resolve_dataset_specifications(
    config: Mapping[str, Any],
    requested_names: Optional[Sequence[str]],
) -> Dict[str, DatasetSpecification]:
    dataset_config = config.get("dataset", {})
    specifications: Dict[str, DatasetSpecification] = {}

    test_dataset = dataset_config.get("test_dataset")

    if isinstance(test_dataset, Mapping):
        _add_dataset_if_valid(
            specifications=specifications,
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

    external_datasets = dataset_config.get(
        "external_datasets",
        {},
    )

    if isinstance(external_datasets, Mapping):
        for dataset_name, entry in external_datasets.items():
            if not isinstance(entry, Mapping):
                continue

            _add_dataset_if_valid(
                specifications=specifications,
                name=str(dataset_name),
                image_dir=entry.get("image_dir", ""),
                mask_dir=entry.get("mask_dir", ""),
            )

    # Reliable defaults for the prepared project structure.
    default_datasets = {
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

    for dataset_name, (
        image_dir,
        mask_dir,
    ) in default_datasets.items():
        if dataset_name not in specifications:
            _add_dataset_if_valid(
                specifications=specifications,
                name=dataset_name,
                image_dir=image_dir,
                mask_dir=mask_dir,
            )

    if requested_names:
        requested_lookup = {
            name.lower(): name
            for name in requested_names
        }

        specifications = {
            name: specification
            for name, specification in specifications.items()
            if name.lower() in requested_lookup
        }

        missing_requested = [
            name
            for name in requested_names
            if name.lower() not in {
                available.lower()
                for available in specifications
            }
        ]

        if missing_requested:
            print(
                "WARNING: requested datasets were not found: "
                + ", ".join(missing_requested)
            )

    if not specifications:
        raise RuntimeError(
            "No evaluation datasets were found. Expected at least:\n"
            "datasets_processed/DRIVE_test/images\n"
            "datasets_processed/DRIVE_test/masks"
        )

    return specifications


# ============================================================
# MODEL AND CHECKPOINT
# ============================================================


def resolve_checkpoint_path(
    config: Mapping[str, Any],
    explicit_path: Optional[str],
) -> Path:
    if explicit_path:
        checkpoint_path = Path(explicit_path)
    else:
        checkpoint_config = config.get("checkpoint", {})
        checkpoint_directory = Path(
            str(
                checkpoint_config.get(
                    "directory",
                    "checkpoints/topology_full_seed42",
                )
            )
        )
        checkpoint_name = str(
            checkpoint_config.get(
                "best_model_name",
                "best_model.pth",
            )
        )
        checkpoint_path = (
            checkpoint_directory
            / checkpoint_name
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Topology checkpoint not found: "
            f"{checkpoint_path.resolve()}"
        )

    return checkpoint_path


def build_model(
    config: Mapping[str, Any],
    device: torch.device,
) -> MVTransUNetTopology:
    model_config = config.get("model", {})
    backbone_config = model_config.get("backbone", {})
    transformer_config = model_config.get("transformer", {})
    vessel_attention_config = model_config.get(
        "vessel_attention",
        {},
    )
    decoder_config = model_config.get("decoder", {})
    deep_supervision_config = model_config.get(
        "deep_supervision",
        {},
    )
    topology_branch_config = model_config.get(
        "topology_branch",
        {},
    )

    model = MVTransUNetTopology(
        pretrained=False,
        transformer_channels=int(
            transformer_config.get(
                "embed_dim",
                768,
            )
        ),
        vessel_reduction_ratio=int(
            vessel_attention_config.get(
                "reduction_ratio",
                16,
            )
        ),
        output_channels=int(
            model_config.get(
                "output_channels",
                1,
            )
        ),
        deep_supervision=bool(
            deep_supervision_config.get(
                "enabled",
                True,
            )
        ),
        decoder_dropout_rate=float(
            decoder_config.get(
                "dropout_rate",
                0.10,
            )
        ),
        topology_channels=int(
            topology_branch_config.get(
                "topology_channels",
                128,
            )
        ),
        topology_branch_channels=int(
            topology_branch_config.get(
                "branch_channels",
                32,
            )
        ),
        topology_head_hidden_channels=int(
            topology_branch_config.get(
                "head_hidden_channels",
                64,
            )
        ),
        topology_refinement_blocks=int(
            topology_branch_config.get(
                "refinement_blocks",
                2,
            )
        ),
        topology_fusion_dropout=float(
            topology_branch_config.get(
                "fusion_dropout",
                0.10,
            )
        ),
        topology_head_dropout=float(
            topology_branch_config.get(
                "head_dropout",
                0.0,
            )
        ),
    )

    return model.to(device)


def load_checkpoint(
    model: MVTransUNetTopology,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must contain a dictionary."
        )

    if "model_state" in checkpoint:
        state_dictionary = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dictionary = checkpoint["state_dict"]
    else:
        # Support a raw state_dict checkpoint.
        state_dictionary = checkpoint

    if not isinstance(state_dictionary, Mapping):
        raise TypeError(
            "Checkpoint does not contain a valid model state dictionary."
        )

    # Handle checkpoints saved from DataParallel.
    if all(
        str(key).startswith("module.")
        for key in state_dictionary.keys()
    ):
        state_dictionary = {
            str(key)[7:]: value
            for key, value in state_dictionary.items()
        }

    model.load_state_dict(
        state_dictionary,
        strict=True,
    )

    model.eval()

    return dict(checkpoint)


# ============================================================
# INFERENCE
# ============================================================


TOPOLOGY_OUTPUT_KEYS = (
    "topology_segmentation_logits",
    "skeleton_logits",
    "endpoint_logits",
    "junction_logits",
)


def _extract_required_outputs(
    outputs: Any,
) -> Dict[str, torch.Tensor]:
    if not isinstance(outputs, Mapping):
        raise TypeError(
            "MVTransUNetTopology must return a dictionary when "
            "return_topology=True."
        )

    missing = [
        key
        for key in TOPOLOGY_OUTPUT_KEYS
        if key not in outputs
    ]

    if missing:
        raise KeyError(
            "Model output is missing required topology entries: "
            f"{missing}"
        )

    extracted = {
        key: outputs[key]
        for key in TOPOLOGY_OUTPUT_KEYS
    }

    for key, value in extracted.items():
        if not torch.is_tensor(value):
            raise TypeError(
                f"Model output '{key}' is not a tensor."
            )

    return extracted


def _flip_tensor(
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

    return torch.flip(
        tensor,
        dims=dimensions,
    )


def predict_logits(
    model: MVTransUNetTopology,
    images: torch.Tensor,
    use_tta: bool,
    amp_enabled: bool,
) -> Dict[str, torch.Tensor]:
    variants = [
        (False, False),
    ]

    if use_tta:
        variants.extend(
            [
                (True, False),
                (False, True),
                (True, True),
            ]
        )

    accumulated: Dict[str, torch.Tensor] = {}

    for horizontal, vertical in variants:
        transformed_images = _flip_tensor(
            images,
            horizontal=horizontal,
            vertical=vertical,
        )

        with torch.autocast(
            device_type=images.device.type,
            enabled=amp_enabled,
        ):
            raw_outputs = model(
                transformed_images,
                return_topology=True,
            )

        outputs = _extract_required_outputs(
            raw_outputs
        )

        for key, logits in outputs.items():
            restored_logits = _flip_tensor(
                logits,
                horizontal=horizontal,
                vertical=vertical,
            )

            if key not in accumulated:
                accumulated[key] = restored_logits.float()
            else:
                accumulated[key] = (
                    accumulated[key]
                    + restored_logits.float()
                )

    number_of_variants = float(
        len(variants)
    )

    return {
        key: value / number_of_variants
        for key, value in accumulated.items()
    }


# ============================================================
# TOPOLOGY UTILITIES
# ============================================================


def binary_skeleton(
    binary_mask: np.ndarray,
) -> np.ndarray:
    return skeletonize(
        binary_mask.astype(bool)
    ).astype(np.uint8)


def skeleton_neighbour_count(
    skeleton: np.ndarray,
) -> np.ndarray:
    skeleton_uint8 = skeleton.astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    neighbourhood_sum = cv2.filter2D(
        skeleton_uint8,
        ddepth=cv2.CV_16S,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )

    # Remove the centre pixel from its own count.
    return (
        neighbourhood_sum
        - skeleton_uint8.astype(np.int16)
    )


def endpoint_map_from_skeleton(
    skeleton: np.ndarray,
) -> np.ndarray:
    neighbours = skeleton_neighbour_count(
        skeleton
    )

    return np.logical_and(
        skeleton > 0,
        neighbours == 1,
    ).astype(np.uint8)


def junction_map_from_skeleton(
    skeleton: np.ndarray,
) -> np.ndarray:
    neighbours = skeleton_neighbour_count(
        skeleton
    )

    return np.logical_and(
        skeleton > 0,
        neighbours >= 3,
    ).astype(np.uint8)


def binary_dilation(
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    size = (
        2 * radius + 1
    )

    kernel = np.ones(
        (size, size),
        dtype=np.uint8,
    )

    return cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    )


def calculate_cldice(
    prediction: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-7,
) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    predicted_skeleton = binary_skeleton(
        prediction
    ).astype(bool)

    target_skeleton = binary_skeleton(
        target
    ).astype(bool)

    topology_precision = (
        np.logical_and(
            predicted_skeleton,
            target,
        ).sum()
        + epsilon
    ) / (
        predicted_skeleton.sum()
        + epsilon
    )

    topology_sensitivity = (
        np.logical_and(
            target_skeleton,
            prediction,
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


def calculate_thin_vessel_metrics(
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

    local_width = (
        distance * 2.0
    )

    target_skeleton = binary_skeleton(
        target_uint8
    ).astype(bool)

    thin_target = np.logical_and(
        target_skeleton,
        local_width <= width_threshold,
    )

    prediction_skeleton = binary_skeleton(
        prediction
    ).astype(bool)

    # One-pixel tolerance makes the centerline comparison less sensitive
    # to tiny rasterisation shifts.
    thin_target_tolerance = binary_dilation(
        thin_target.astype(np.uint8),
        radius=1,
    ).astype(bool)

    prediction_tolerance = binary_dilation(
        prediction_skeleton.astype(np.uint8),
        radius=1,
    ).astype(bool)

    true_positive = np.logical_and(
        prediction_skeleton,
        thin_target_tolerance,
    ).sum()

    matched_target = np.logical_and(
        thin_target,
        prediction_tolerance,
    ).sum()

    precision = (
        true_positive + epsilon
    ) / (
        prediction_skeleton.sum()
        + epsilon
    )

    recall = (
        matched_target + epsilon
    ) / (
        thin_target.sum()
        + epsilon
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


# ============================================================
# METRICS
# ============================================================


def safe_divide(
    numerator: float,
    denominator: float,
    epsilon: float = 1e-7,
) -> float:
    return float(
        numerator
        / (
            denominator
            + epsilon
        )
    )


def calculate_binary_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    probability = probability.astype(np.float64)
    target = target.astype(np.uint8)
    prediction = (
        probability >= threshold
    ).astype(np.uint8)

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
            np.logical_not(prediction_bool),
            np.logical_not(target_bool),
        ).sum()
    )

    false_positive = float(
        np.logical_and(
            prediction_bool,
            np.logical_not(target_bool),
        ).sum()
    )

    false_negative = float(
        np.logical_and(
            np.logical_not(prediction_bool),
            target_bool,
        ).sum()
    )

    dice = safe_divide(
        2.0 * true_positive,
        (
            2.0 * true_positive
            + false_positive
            + false_negative
        ),
    )

    iou = safe_divide(
        true_positive,
        (
            true_positive
            + false_positive
            + false_negative
        ),
    )

    sensitivity = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    accuracy = safe_divide(
        true_positive + true_negative,
        (
            true_positive
            + true_negative
            + false_positive
            + false_negative
        ),
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    if np.unique(target).size > 1:
        roc_auc = float(
            roc_auc_score(
                target.reshape(-1),
                probability.reshape(-1),
            )
        )
    else:
        roc_auc = float("nan")

    metrics = {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "precision": precision,
        "roc_auc": roc_auc,
        "cldice": calculate_cldice(
            prediction=prediction,
            target=target,
        ),
    }

    metrics.update(
        calculate_thin_vessel_metrics(
            prediction=prediction,
            target=target,
        )
    )

    return metrics


def calculate_map_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
    tolerance_radius: int = 1,
) -> Dict[str, float]:
    prediction = (
        probability >= threshold
    ).astype(np.uint8)

    target = target.astype(np.uint8)

    dilated_target = binary_dilation(
        target,
        radius=tolerance_radius,
    ).astype(bool)

    dilated_prediction = binary_dilation(
        prediction,
        radius=tolerance_radius,
    ).astype(bool)

    prediction_bool = prediction.astype(bool)
    target_bool = target.astype(bool)

    matched_prediction = np.logical_and(
        prediction_bool,
        dilated_target,
    ).sum()

    matched_target = np.logical_and(
        target_bool,
        dilated_prediction,
    ).sum()

    precision = safe_divide(
        float(matched_prediction),
        float(prediction_bool.sum()),
    )

    recall = safe_divide(
        float(matched_target),
        float(target_bool.sum()),
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# SAVING
# ============================================================


def ensure_output_directories(
    root: Path,
) -> Dict[str, Path]:
    directories = {
        "root": root,
        "segmentation": root / "predictions" / "segmentation",
        "probability": root / "predictions" / "probability",
        "skeleton": root / "predictions" / "skeleton",
        "endpoint": root / "predictions" / "endpoint",
        "junction": root / "predictions" / "junction",
        "comparison": root / "comparisons",
    }

    for directory in directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


def save_grayscale_image(
    path: Path,
    array: np.ndarray,
) -> None:
    array = np.asarray(array)

    if array.dtype == bool:
        array = array.astype(np.uint8) * 255

    elif np.issubdtype(
        array.dtype,
        np.floating,
    ):
        array = np.clip(
            array,
            0.0,
            1.0,
        )
        array = (
            array * 255.0
        ).round().astype(np.uint8)

    else:
        array = np.clip(
            array,
            0,
            255,
        ).astype(np.uint8)

    success = cv2.imwrite(
        str(path),
        array,
    )

    if not success:
        raise IOError(
            f"Failed to save image: {path.resolve()}"
        )


def denormalize_image(
    image_tensor: torch.Tensor,
) -> np.ndarray:
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ).view(3, 1, 1)

    standard_deviation = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ).view(3, 1, 1)

    image = (
        image_tensor
        * standard_deviation
        + mean
    ).clamp(0.0, 1.0)

    image = image.permute(
        1,
        2,
        0,
    ).detach().cpu().numpy()

    return (
        image * 255.0
    ).round().astype(np.uint8)


def save_comparison_image(
    path: Path,
    image_rgb: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    skeleton_probability: np.ndarray,
) -> None:
    target_rgb = cv2.cvtColor(
        (
            target.astype(np.uint8)
            * 255
        ),
        cv2.COLOR_GRAY2RGB,
    )

    probability_rgb = cv2.applyColorMap(
        (
            np.clip(
                probability,
                0.0,
                1.0,
            )
            * 255
        ).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    probability_rgb = cv2.cvtColor(
        probability_rgb,
        cv2.COLOR_BGR2RGB,
    )

    prediction_rgb = cv2.cvtColor(
        (
            prediction.astype(np.uint8)
            * 255
        ),
        cv2.COLOR_GRAY2RGB,
    )

    skeleton_rgb = cv2.applyColorMap(
        (
            np.clip(
                skeleton_probability,
                0.0,
                1.0,
            )
            * 255
        ).astype(np.uint8),
        cv2.COLORMAP_MAGMA,
    )
    skeleton_rgb = cv2.cvtColor(
        skeleton_rgb,
        cv2.COLOR_BGR2RGB,
    )

    panels = [
        image_rgb,
        target_rgb,
        probability_rgb,
        prediction_rgb,
        skeleton_rgb,
    ]

    panel_height = min(
        panel.shape[0]
        for panel in panels
    )

    resized_panels = [
        cv2.resize(
            panel,
            (
                int(
                    panel.shape[1]
                    * (
                        panel_height
                        / panel.shape[0]
                    )
                ),
                panel_height,
            ),
            interpolation=cv2.INTER_AREA,
        )
        for panel in panels
    ]

    comparison_rgb = np.concatenate(
        resized_panels,
        axis=1,
    )

    comparison_bgr = cv2.cvtColor(
        comparison_rgb,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(path),
        comparison_bgr,
    )

    if not success:
        raise IOError(
            f"Failed to save comparison: {path.resolve()}"
        )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def json_safe_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (np.floating, np.integer),
    ):
        value = value.item()

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def write_json(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    safe_data = {
        key: json_safe_value(value)
        for key, value in data.items()
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            safe_data,
            file,
            indent=2,
            sort_keys=True,
        )


# ============================================================
# DATASET EVALUATION
# ============================================================


def mean_metric_rows(
    rows: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    metric_keys = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in {
                "sample",
                "dataset",
                "image_path",
                "mask_path",
            }
        }
    )

    summary: Dict[str, float] = {}

    for key in metric_keys:
        values = np.asarray(
            [
                float(row[key])
                for row in rows
                if (
                    key in row
                    and row[key] is not None
                )
            ],
            dtype=np.float64,
        )

        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.size == 0:
            summary[key] = float("nan")
            summary[f"{key}_std"] = float("nan")
            continue

        summary[key] = float(
            finite_values.mean()
        )
        summary[f"{key}_std"] = float(
            finite_values.std(ddof=0)
        )

    return summary


def evaluate_dataset(
    model: MVTransUNetTopology,
    specification: DatasetSpecification,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_tta: bool,
    amp_enabled: bool,
    output_root: Path,
    save_probabilities: bool,
) -> Dict[str, float]:
    dataset_output_root = (
        output_root
        / specification.name
    )

    directories = ensure_output_directories(
        dataset_output_root
    )

    per_image_rows: List[Dict[str, Any]] = []

    progress = tqdm(
        loader,
        desc=f"Evaluating {specification.name}",
        leave=True,
    )

    with torch.inference_mode():
        for batch in progress:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            output_logits = predict_logits(
                model=model,
                images=images,
                use_tta=use_tta,
                amp_enabled=amp_enabled,
            )

            output_probabilities = {
                key: torch.sigmoid(logits)
                for key, logits in output_logits.items()
            }

            batch_size = int(
                images.shape[0]
            )

            for batch_index in range(batch_size):
                key = str(
                    batch["key"][batch_index]
                )

                target = (
                    masks[batch_index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    > 0.5
                ).astype(np.uint8)

                segmentation_probability = (
                    output_probabilities[
                        "topology_segmentation_logits"
                    ][batch_index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                skeleton_probability = (
                    output_probabilities[
                        "skeleton_logits"
                    ][batch_index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                endpoint_probability = (
                    output_probabilities[
                        "endpoint_logits"
                    ][batch_index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                junction_probability = (
                    output_probabilities[
                        "junction_logits"
                    ][batch_index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                prediction = (
                    segmentation_probability
                    >= threshold
                ).astype(np.uint8)

                ground_truth_skeleton = (
                    binary_skeleton(target)
                )

                ground_truth_endpoint = (
                    endpoint_map_from_skeleton(
                        ground_truth_skeleton
                    )
                )

                ground_truth_junction = (
                    junction_map_from_skeleton(
                        ground_truth_skeleton
                    )
                )

                segmentation_metrics = (
                    calculate_binary_metrics(
                        probability=segmentation_probability,
                        target=target,
                        threshold=threshold,
                    )
                )

                skeleton_metrics = (
                    calculate_map_metrics(
                        probability=skeleton_probability,
                        target=ground_truth_skeleton,
                        threshold=threshold,
                        tolerance_radius=1,
                    )
                )

                endpoint_metrics = (
                    calculate_map_metrics(
                        probability=endpoint_probability,
                        target=ground_truth_endpoint,
                        threshold=threshold,
                        tolerance_radius=2,
                    )
                )

                junction_metrics = (
                    calculate_map_metrics(
                        probability=junction_probability,
                        target=ground_truth_junction,
                        threshold=threshold,
                        tolerance_radius=2,
                    )
                )

                row: Dict[str, Any] = {
                    "dataset": specification.name,
                    "sample": key,
                    "image_path": str(
                        batch["image_path"][batch_index]
                    ),
                    "mask_path": str(
                        batch["mask_path"][batch_index]
                    ),
                    **segmentation_metrics,
                    **{
                        f"skeleton_{name}": value
                        for name, value in skeleton_metrics.items()
                    },
                    **{
                        f"endpoint_{name}": value
                        for name, value in endpoint_metrics.items()
                    },
                    **{
                        f"junction_{name}": value
                        for name, value in junction_metrics.items()
                    },
                }

                per_image_rows.append(row)

                save_grayscale_image(
                    directories["segmentation"]
                    / f"{key}.png",
                    prediction,
                )

                save_grayscale_image(
                    directories["skeleton"]
                    / f"{key}.png",
                    skeleton_probability >= threshold,
                )

                save_grayscale_image(
                    directories["endpoint"]
                    / f"{key}.png",
                    endpoint_probability >= threshold,
                )

                save_grayscale_image(
                    directories["junction"]
                    / f"{key}.png",
                    junction_probability >= threshold,
                )

                if save_probabilities:
                    save_grayscale_image(
                        directories["probability"]
                        / f"{key}.png",
                        segmentation_probability,
                    )

                image_rgb = denormalize_image(
                    images[batch_index]
                )

                save_comparison_image(
                    directories["comparison"]
                    / f"{key}.png",
                    image_rgb=image_rgb,
                    target=target,
                    probability=segmentation_probability,
                    prediction=prediction,
                    skeleton_probability=skeleton_probability,
                )

            if per_image_rows:
                progress.set_postfix(
                    dice=(
                        f"{per_image_rows[-1]['dice']:.4f}"
                    ),
                    cldice=(
                        f"{per_image_rows[-1]['cldice']:.4f}"
                    ),
                )

    summary = mean_metric_rows(
        per_image_rows
    )

    summary.update(
        {
            "dataset": specification.name,
            "number_of_images": len(per_image_rows),
            "threshold": threshold,
            "tta_enabled": bool(use_tta),
        }
    )

    write_csv(
        dataset_output_root
        / "per_image_metrics.csv",
        per_image_rows,
    )

    write_json(
        dataset_output_root
        / "summary_metrics.json",
        summary,
    )

    return summary


# ============================================================
# REPORTING
# ============================================================


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
    "skeleton_f1",
    "endpoint_f1",
    "junction_f1",
)


def print_dataset_summary(
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

        value = summary[metric]
        standard_deviation = summary.get(
            f"{metric}_std"
        )

        if (
            value is None
            or not np.isfinite(float(value))
        ):
            rendered = "N/A"
        elif (
            standard_deviation is not None
            and np.isfinite(
                float(standard_deviation)
            )
        ):
            rendered = (
                f"{float(value):.6f} "
                f"± {float(standard_deviation):.6f}"
            )
        else:
            rendered = f"{float(value):.6f}"

        print(
            f"{metric:24s}: {rendered}"
        )


def write_combined_summary(
    output_root: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(
        output_root / "all_dataset_summary.csv",
        summaries,
    )

    json_payload = {
        str(summary["dataset"]): {
            key: json_safe_value(value)
            for key, value in summary.items()
        }
        for summary in summaries
    }

    with (
        output_root
        / "all_dataset_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_payload,
            file,
            indent=2,
            sort_keys=True,
        )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    arguments = parse_arguments()

    config = load_config(
        arguments.config
    )

    seed = int(
        config.get(
            "seed",
            {},
        ).get(
            "value",
            42,
        )
    )

    deterministic = bool(
        config.get(
            "seed",
            {},
        ).get(
            "deterministic",
            True,
        )
    )

    seed_everything(
        seed=seed,
        deterministic=deterministic,
    )

    device = resolve_device(
        config
    )

    hardware_config = config.get(
        "hardware",
        {},
    )

    amp_enabled = bool(
        hardware_config.get(
            "mixed_precision",
            True,
        )
        and device.type == "cuda"
    )

    patch_config = config.get(
        "patch_training",
        {},
    )

    if bool(
        patch_config.get(
            "enabled",
            False,
        )
    ):
        image_size = int(
            patch_config.get(
                "model_input_size",
                256,
            )
        )
    else:
        image_size = int(
            config.get(
                "preprocessing",
                {},
            ).get(
                "image_size",
                {},
            ).get(
                "height",
                256,
            )
        )

    if arguments.threshold is not None:
        threshold = float(
            arguments.threshold
        )
    else:
        threshold = float(
            config.get(
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

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Evaluation threshold must be in [0, 1]."
        )

    checkpoint_path = resolve_checkpoint_path(
        config=config,
        explicit_path=arguments.checkpoint,
    )

    model = build_model(
        config=config,
        device=device,
    )

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    specifications = resolve_dataset_specifications(
        config=config,
        requested_names=arguments.datasets,
    )

    output_root = Path(
        arguments.output_dir
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
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

    pin_memory = bool(
        device.type == "cuda"
    )

    print("=" * 78)
    print("MV-TransUNetTopology Evaluation")
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
        "Saved best Dice:",
        checkpoint.get(
            "best_dice",
            checkpoint.get(
                "dice",
                "unknown",
            ),
        ),
    )
    print("Input size:", image_size)
    print("Threshold:", threshold)
    print("TTA:", bool(arguments.tta))
    print(
        "Datasets:",
        ", ".join(specifications.keys()),
    )
    print("Output directory:", output_root.resolve())

    summaries: List[Dict[str, Any]] = []

    for specification in specifications.values():
        print()
        print("-" * 78)
        print(
            f"Preparing {specification.name}"
        )
        print(
            "Images:",
            specification.image_dir.resolve(),
        )
        print(
            "Masks:",
            specification.mask_dir.resolve(),
        )

        loader = build_loader(
            specification=specification,
            image_size=image_size,
            batch_size=int(
                arguments.batch_size
            ),
            num_workers=int(
                arguments.num_workers
            ),
            clahe_enabled=clahe_enabled,
            pin_memory=pin_memory,
        )

        print(
            "Matched images:",
            len(loader.dataset),
        )

        summary = evaluate_dataset(
            model=model,
            specification=specification,
            loader=loader,
            device=device,
            threshold=threshold,
            use_tta=bool(arguments.tta),
            amp_enabled=amp_enabled,
            output_root=output_root,
            save_probabilities=bool(
                arguments.save_probabilities
            ),
        )

        summaries.append(summary)
        print_dataset_summary(summary)

    write_combined_summary(
        output_root=output_root,
        summaries=summaries,
    )

    print()
    print("=" * 78)
    print("Evaluation complete")
    print("=" * 78)
    print(
        "Combined summary:",
        (
            output_root
            / "all_dataset_summary.csv"
        ).resolve(),
    )
    print(
        "Prediction folders:",
        (
            output_root
        ).resolve(),
    )


if __name__ == "__main__":
    main()