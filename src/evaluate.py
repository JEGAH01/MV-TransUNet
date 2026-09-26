"""
Clean, publication-oriented MV-TransUNet evaluation pipeline.

Protocol
--------
1. Reconstruct the exact DRIVE internal validation split used during training.
2. Run native-resolution sliding-window inference with overlapping patches.
3. Fuse overlapping probability maps with a Hann window.
4. Select one binarization threshold on reconstructed validation images only.
5. Freeze that threshold.
6. Evaluate the official DRIVE test set and all external datasets with exactly
   the same reconstruction, fusion, preprocessing, and metric implementation.

Important
---------
- The checkpoint's saved ``best_dice`` is the patch-validation score used for
  checkpoint selection. It is not directly comparable to reconstructed
  whole-image Dice.
- Test-set thresholds are never tuned.
- Test-time augmentation is intentionally disabled in this standardized
  protocol.
- Metrics come from ``src.metrics.calculate_metrics`` as the single source of
  truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from tqdm import tqdm

from models.mv_transunet import MVTransUNet
from src.datasets import (
    GridPatchRetinalDataset,
    create_dataloader,
    create_split_indices,
    get_validation_transform,
    load_binary_mask,
    load_sample_pairs,
)
from src.metrics import calculate_metrics


SamplePair = Tuple[Path, Path]


# ============================================================
# CONFIGURATION AND REPRODUCIBILITY
# ============================================================


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load and validate the YAML configuration."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml is empty or does not contain a mapping.")

    return config


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed NumPy and PyTorch for reproducible evaluation."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    """Resolve CPU/CUDA from config and print the selected runtime."""

    hardware = config.get("hardware", {})
    requested = str(hardware.get("device", "auto")).strip().lower()
    gpu_id = int(hardware.get("gpu_id", 0))

    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            "hardware.device must be one of: auto, cuda, cpu. "
            f"Received: {requested!r}"
        )

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        if not 0 <= gpu_id < torch.cuda.device_count():
            raise ValueError(
                f"hardware.gpu_id={gpu_id} is invalid; "
                f"{torch.cuda.device_count()} CUDA device(s) detected."
            )

        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        print("CUDA available: True")
        print("PyTorch version:", torch.__version__)
        print("PyTorch CUDA runtime:", torch.version.cuda)
        print("Selected GPU:", torch.cuda.get_device_name(gpu_id))

        return device

    if requested == "cuda":
        print(
            "WARNING: CUDA was requested, but torch.cuda.is_available() is False. "
            "Evaluation will continue on CPU."
        )

    return torch.device("cpu")


# ============================================================
# MODEL AND CHECKPOINT
# ============================================================


def get_main_output(outputs: Any) -> torch.Tensor:
    """Extract final logits from tensor or deep-supervision dictionary output."""

    if torch.is_tensor(outputs):
        return outputs

    if isinstance(outputs, dict):
        for key in ("main_output", "out", "logits"):
            if key in outputs:
                return outputs[key]

        raise KeyError(
            "Model returned a dictionary without main_output, out, or logits."
        )

    if isinstance(outputs, (tuple, list)) and outputs:
        if torch.is_tensor(outputs[0]):
            return outputs[0]

    raise TypeError(
        "Unsupported model output type. Expected tensor, dictionary, tuple, or list."
    )


def resolve_checkpoint_path(config: Mapping[str, Any]) -> Path:
    """
    Support both checkpoint styles used across the project:

    1. checkpoint.directory: checkpoints/experiment_D
    2. checkpoint.root_directory + experiment.name
    """

    checkpoint = config.get("checkpoint", {})
    best_name = str(checkpoint.get("best_model_name", "best_model.pth"))

    explicit_directory = checkpoint.get("directory")
    if explicit_directory:
        return Path(str(explicit_directory)) / best_name

    root = Path(str(checkpoint.get("root_directory", "checkpoints")))
    experiment_name = str(
        config.get("experiment", {}).get("name", "default_experiment")
    ).strip()

    return root / experiment_name / best_name


def resolve_output_directory(config: Mapping[str, Any]) -> Path:
    """Resolve experiment output directory while supporting old/new config styles."""

    experiment = config.get("experiment", {})

    explicit = experiment.get("output_directory")
    if explicit:
        return Path(str(explicit))

    root = Path(str(experiment.get("output_root_directory", "experiments")))
    name = str(experiment.get("name", "default_experiment")).strip()

    return root / name


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    config: Mapping[str, Any],
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Build the exact configured architecture and load its checkpoint."""

    model_config = config.get("model", {})
    deep_supervision = model_config.get("deep_supervision", {})

    model = MVTransUNet(
        pretrained=False,
        transformer_channels=int(
            model_config.get("transformer", {}).get("embed_dim", 768)
        ),
        vessel_reduction_ratio=int(
            model_config.get("vessel_attention", {}).get("reduction_ratio", 16)
        ),
        output_channels=int(model_config.get("output_channels", 1)),
        deep_supervision=bool(deep_supervision.get("enabled", True)),
        decoder_dropout_rate=float(
            model_config.get("decoder", {}).get("dropout_rate", 0.1)
        ),
    )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Loaded checkpoint:", checkpoint_path.resolve())
    print(
        "  saved epoch:",
        checkpoint.get("epoch", "unknown"),
        "| saved patch-validation best_dice:",
        checkpoint.get("best_dice", "unknown"),
    )

    return model, checkpoint


# ============================================================
# DATASET RESOLUTION
# ============================================================


def reconstruct_internal_validation_pairs(
    config: Mapping[str, Any],
) -> List[SamplePair]:
    """Recreate the exact image-level validation split used by train.py."""

    dataset_config = config["dataset"]
    train_config = dataset_config["train_dataset"]

    pairs = load_sample_pairs(
        train_config["image_dir"],
        train_config["mask_dir"],
    )

    if not pairs:
        raise RuntimeError("No DRIVE training image-mask pairs were found.")

    split_seed = int(
        dataset_config.get("split_seed", config["seed"]["value"])
    )
    validation_ratio = float(dataset_config.get("validation_ratio", 0.2))

    _, validation_indices = create_split_indices(
        dataset_size=len(pairs),
        validation_ratio=validation_ratio,
        seed=split_seed,
    )

    validation_pairs = [pairs[index] for index in validation_indices]

    if not validation_pairs:
        raise RuntimeError("The reconstructed validation split is empty.")

    print(
        "Reconstructed DRIVE internal validation split:",
        f"{len(validation_pairs)} image(s) from {len(pairs)} training image(s).",
    )

    return validation_pairs


def normalize_external_dataset_entries(
    external_datasets: Any,
) -> List[Tuple[str, Mapping[str, Any]]]:
    """Normalize dictionary/list external-dataset config formats."""

    if not external_datasets:
        return []

    if isinstance(external_datasets, dict):
        return [
            (str(name), dataset_config)
            for name, dataset_config in external_datasets.items()
        ]

    if isinstance(external_datasets, Sequence):
        entries: List[Tuple[str, Mapping[str, Any]]] = []

        for dataset_config in external_datasets:
            if not isinstance(dataset_config, Mapping):
                raise TypeError(
                    "Each external dataset entry must be a mapping."
                )

            name = str(dataset_config["name"])
            entries.append((name, dataset_config))

        return entries

    raise TypeError(
        "dataset.external_datasets must be a mapping or sequence of mappings."
    )


def load_configured_pairs(
    dataset_name: str,
    dataset_config: Mapping[str, Any],
) -> List[SamplePair]:
    """Load and validate image-mask pairs for one configured dataset."""

    if "image_dir" not in dataset_config or "mask_dir" not in dataset_config:
        raise KeyError(
            f"{dataset_name} requires image_dir and mask_dir in config.yaml."
        )

    pairs = load_sample_pairs(
        dataset_config["image_dir"],
        dataset_config["mask_dir"],
    )

    if not pairs:
        raise RuntimeError(
            f"No image-mask pairs were found for {dataset_name}."
        )

    print(f"{dataset_name}: discovered {len(pairs)} image-mask pair(s).")
    return pairs


# ============================================================
# RECONSTRUCTION CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class ReconstructionSettings:
    patch_size: int
    model_input_size: int
    stride: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    fusion_mode: str
    minimum_weight: float
    clahe: bool
    seed: int


def resolve_reconstruction_settings(
    config: Mapping[str, Any],
) -> ReconstructionSettings:
    """Read and validate all reconstruction settings."""

    patch_config = config.get("patch_training", {})

    if not bool(patch_config.get("enabled", False)):
        raise ValueError(
            "This standardized evaluator requires patch_training.enabled: true."
        )

    evaluation = config.get("evaluation", {})
    reconstruction = evaluation.get("reconstruction", {})
    dataloader = config.get("dataloader", {})

    model_input_size = int(patch_config.get("model_input_size", 256))
    patch_size = int(patch_config.get("patch_size", model_input_size))
    stride = int(
        reconstruction.get(
            "stride",
            patch_config.get("validation_stride", max(1, patch_size // 2)),
        )
    )

    if patch_size < 1 or model_input_size < 1 or stride < 1:
        raise ValueError(
            "patch_size, model_input_size, and reconstruction stride "
            "must all be positive."
        )

    if stride > patch_size:
        raise ValueError(
            "Reconstruction stride cannot exceed patch_size because that "
            "would leave uncovered image regions."
        )

    fusion_mode = str(
        reconstruction.get("fusion_mode", "hann")
    ).strip().lower()

    if fusion_mode not in {"hann", "uniform"}:
        raise ValueError(
            "evaluation.reconstruction.fusion_mode must be hann or uniform."
        )

    minimum_weight = float(reconstruction.get("minimum_weight", 0.05))

    if not 0.0 < minimum_weight <= 1.0:
        raise ValueError(
            "evaluation.reconstruction.minimum_weight must lie in (0, 1]."
        )

    return ReconstructionSettings(
        patch_size=patch_size,
        model_input_size=model_input_size,
        stride=stride,
        batch_size=int(reconstruction.get("batch_size", 1)),
        num_workers=int(dataloader.get("num_workers", 0)),
        pin_memory=bool(dataloader.get("pin_memory", True)),
        fusion_mode=fusion_mode,
        minimum_weight=minimum_weight,
        clahe=bool(
            config.get("preprocessing", {})
            .get("clahe", {})
            .get("enabled", True)
        ),
        seed=int(
            config["dataset"].get(
                "split_seed",
                config["seed"]["value"],
            )
        ),
    )


def create_fusion_window(
    height: int,
    width: int,
    mode: str,
    minimum_weight: float,
) -> np.ndarray:
    """Create a uniform or positive-floor 2-D Hann fusion window."""

    if mode == "uniform":
        return np.ones((height, width), dtype=np.float32)

    vertical = (
        np.hanning(height).astype(np.float32)
        if height > 1
        else np.ones(1, dtype=np.float32)
    )
    horizontal = (
        np.hanning(width).astype(np.float32)
        if width > 1
        else np.ones(1, dtype=np.float32)
    )

    window = np.outer(vertical, horizontal).astype(np.float32)
    maximum = float(window.max())

    if maximum <= 0.0:
        return np.ones((height, width), dtype=np.float32)

    window /= maximum
    return np.maximum(window, minimum_weight).astype(np.float32)


def build_reconstruction_dataset(
    samples: Sequence[SamplePair],
    settings: ReconstructionSettings,
) -> GridPatchRetinalDataset:
    """Create a complete deterministic patch grid for native-resolution inference."""

    return GridPatchRetinalDataset(
        samples=samples,
        patch_size=settings.patch_size,
        model_input_size=settings.model_input_size,
        stride=settings.stride,
        transform=get_validation_transform(
            image_size=settings.model_input_size
        ),
        clahe=settings.clahe,
        include_empty_patches=True,
        minimum_vessel_fraction=0.0,
       
    )


# ============================================================
# PATCH INFERENCE AND NATIVE-RESOLUTION RECONSTRUCTION
# ============================================================


@torch.inference_mode()
def reconstruct_probability_maps(
    model: torch.nn.Module,
    samples: Sequence[SamplePair],
    settings: ReconstructionSettings,
    device: torch.device,
    description: str,
) -> Tuple[
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
]:
    """
    Reconstruct one probability map per source image.

    The network output is resized back to the original extracted patch size
    before placement. This is essential when patch_size != model_input_size.
    """

    dataset = build_reconstruction_dataset(samples, settings)

    loader = create_dataloader(
        dataset=dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=False,
        drop_last=False,
        seed=settings.seed,
    )

    patch_height = int(dataset.patch_height)
    patch_width = int(dataset.patch_width)

    fusion_window = create_fusion_window(
        height=patch_height,
        width=patch_width,
        mode=settings.fusion_mode,
        minimum_weight=settings.minimum_weight,
    )

    targets: Dict[int, np.ndarray] = {}
    weighted_sums: Dict[int, np.ndarray] = {}
    weight_sums: Dict[int, np.ndarray] = {}

    for image_index, (_, mask_path) in enumerate(dataset.samples):
        target = load_binary_mask(mask_path).astype(np.uint8)
        height, width = target.shape

        targets[image_index] = target
        weighted_sums[image_index] = np.zeros(
            (height, width),
            dtype=np.float32,
        )
        weight_sums[image_index] = np.zeros(
            (height, width),
            dtype=np.float32,
        )

    model.eval()

    for batch in tqdm(loader, desc=description):
        images = batch["image"].to(device, non_blocking=True)

        with autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            logits = get_main_output(model(images))
            probabilities = torch.sigmoid(logits)

        if probabilities.shape[-2:] != (patch_height, patch_width):
            probabilities = F.interpolate(
                probabilities.float(),
                size=(patch_height, patch_width),
                mode="bilinear",
                align_corners=False,
            )

        probabilities_np = (
            probabilities.float().cpu().numpy()[:, 0]
        )

        source_indices = batch["source_image_index"]
        top_values = batch["top"]
        left_values = batch["left"]

        for batch_index, patch_probability in enumerate(probabilities_np):
            source_index = int(source_indices[batch_index].item())
            top = int(top_values[batch_index].item())
            left = int(left_values[batch_index].item())

            target_height, target_width = targets[source_index].shape
            bottom = min(top + patch_height, target_height)
            right = min(left + patch_width, target_width)

            valid_height = bottom - top
            valid_width = right - left

            if valid_height <= 0 or valid_width <= 0:
                raise RuntimeError(
                    f"Patch at top={top}, left={left} lies outside "
                    f"source image {source_index}."
                )

            valid_probability = patch_probability[
                :valid_height,
                :valid_width,
            ]
            valid_weight = fusion_window[
                :valid_height,
                :valid_width,
            ]

            weighted_sums[source_index][top:bottom, left:right] += (
                valid_probability * valid_weight
            )
            weight_sums[source_index][top:bottom, left:right] += valid_weight

    reconstructed: Dict[int, np.ndarray] = {}

    for image_index in sorted(targets):
        weight_sum = weight_sums[image_index]

        if np.any(weight_sum <= 0.0):
            uncovered_pixels = int(np.sum(weight_sum <= 0.0))
            raise RuntimeError(
                f"Reconstruction left {uncovered_pixels} uncovered pixel(s) "
                f"in source image {image_index}. Check stride and patch_size."
            )

        reconstructed[image_index] = (
            weighted_sums[image_index] / weight_sum
        ).astype(np.float32)

    return reconstructed, targets, weight_sums


# ============================================================
# THRESHOLD SELECTION AND METRICS
# ============================================================


def threshold_candidates(
    config: Mapping[str, Any],
) -> np.ndarray:
    """Build a configurable threshold grid; default is 0.20 to 0.80 by 0.01."""

    threshold_config = (
        config.get("evaluation", {}).get("threshold_sweep", {})
    )

    start = float(threshold_config.get("start", 0.20))
    stop = float(threshold_config.get("stop", 0.96))
    step = float(threshold_config.get("step", 0.01))

    if not 0.0 <= start < stop <= 1.0:
        raise ValueError(
            "Threshold sweep must satisfy 0 <= start < stop <= 1."
        )

    if step <= 0.0:
        raise ValueError("Threshold sweep step must be positive.")

    number = int(round((stop - start) / step))
    candidates = start + np.arange(number + 1, dtype=np.float64) * step
    candidates = candidates[candidates <= stop + 1e-10]

    return np.round(candidates, 6)


def select_validation_threshold(
    reconstructed: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    candidates: Iterable[float],
) -> Tuple[float, float, Dict[float, float]]:
    """
    Select the threshold maximizing macro image-level Dice.

    When several thresholds are numerically tied, select the one closest to
    0.5, then the lower one. This deterministic tie-break avoids arbitrary
    dependence on candidate ordering.
    """

    scores: Dict[float, float] = {}

    print("\nReconstructed validation threshold sweep:")

    for threshold in candidates:
        threshold = float(threshold)

        dice_values = [
            float(
                calculate_metrics(
                    prediction=reconstructed[index] > threshold,
                    target=targets[index],
                )["dice"]
            )
            for index in sorted(reconstructed)
        ]

        mean_dice = float(np.mean(dice_values))
        scores[threshold] = mean_dice
        print(f"  threshold={threshold:.2f} -> Dice={mean_dice:.4f}")

    if not scores:
        raise RuntimeError("Threshold sweep produced no candidates.")

    best_dice = max(scores.values())
    tolerance = 1e-12

    tied_thresholds = [
        threshold
        for threshold, dice in scores.items()
        if abs(dice - best_dice) <= tolerance
    ]

    best_threshold = min(
        tied_thresholds,
        key=lambda value: (abs(value - 0.5), value),
    )

    print(
        "\nSelected validation threshold:",
        f"{best_threshold:.2f}",
        f"(reconstructed validation Dice={best_dice:.4f})",
    )

    return float(best_threshold), float(best_dice), scores


def summarize_reconstructed_dataset(
    dataset_name: str,
    samples: Sequence[SamplePair],
    reconstructed: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    weight_sums: Mapping[int, np.ndarray],
    threshold: float,
    output_directory: Optional[Path],
    save_outputs: bool,
) -> Dict[str, Any]:
    """Calculate macro image metrics and optionally save per-image outputs."""

    case_metrics: List[Dict[str, float]] = []
    case_records: List[Dict[str, Any]] = []

    if save_outputs and output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)

    for image_index in sorted(reconstructed):
        probability = reconstructed[image_index]
        target = targets[image_index]
        prediction = probability > float(threshold)

        metrics = {
            key: float(value)
            for key, value in calculate_metrics(
                prediction=prediction,
                target=target,
                probability=probability,
            ).items()
        }

        image_path, mask_path = samples[image_index]

        case_metrics.append(metrics)
        case_records.append(
            {
                "index": int(image_index),
                "image": str(image_path),
                "mask": str(mask_path),
                **metrics,
            }
        )

        if save_outputs and output_directory is not None:
            stem = f"{image_index:02d}_{Path(image_path).stem}"

            cv2.imwrite(
                str(output_directory / f"{stem}_prediction.png"),
                prediction.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                str(output_directory / f"{stem}_probability.png"),
                np.clip(probability * 255.0, 0, 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(output_directory / f"{stem}_target.png"),
                target.astype(np.uint8) * 255,
            )

            normalized_weight = weight_sums[image_index] / max(
                float(weight_sums[image_index].max()),
                1e-8,
            )
            cv2.imwrite(
                str(output_directory / f"{stem}_weight_sum.png"),
                np.clip(normalized_weight * 255.0, 0, 255).astype(np.uint8),
            )

    if not case_metrics:
        raise RuntimeError(f"No cases were evaluated for {dataset_name}.")

    metric_names = case_metrics[0].keys()
    summary = {
        metric_name: float(
            np.mean([case[metric_name] for case in case_metrics])
        )
        for metric_name in metric_names
    }

    summary.update(
        {
            "threshold": float(threshold),
            "images_evaluated": int(len(case_metrics)),
            "aggregation": "macro_mean_over_whole_images",
            "inference": "native_resolution_overlapping_patch_reconstruction",
            "tta": False,
            "cases": case_records,
        }
    )

    return summary


# ============================================================
# EVALUATION ORCHESTRATION
# ============================================================


def evaluate_one_dataset(
    dataset_name: str,
    samples: Sequence[SamplePair],
    model: torch.nn.Module,
    settings: ReconstructionSettings,
    device: torch.device,
    threshold: float,
    experiment_output_directory: Path,
    save_outputs: bool,
) -> Dict[str, Any]:
    """Run the complete fixed-threshold reconstruction protocol."""

    print("\n" + "=" * 72)
    print(f"Evaluating {dataset_name}")
    print(
        f"patch={settings.patch_size}, input={settings.model_input_size}, "
        f"stride={settings.stride}, fusion={settings.fusion_mode}, "
        f"threshold={threshold:.2f}, TTA=False"
    )
    print("=" * 72)

    reconstructed, targets, weight_sums = reconstruct_probability_maps(
        model=model,
        samples=samples,
        settings=settings,
        device=device,
        description=f"Reconstructing {dataset_name}",
    )

    output_directory = (
        experiment_output_directory / "predictions" / dataset_name
    )

    results = summarize_reconstructed_dataset(
        dataset_name=dataset_name,
        samples=samples,
        reconstructed=reconstructed,
        targets=targets,
        weight_sums=weight_sums,
        threshold=threshold,
        output_directory=output_directory,
        save_outputs=save_outputs,
    )

    print(
        f"{dataset_name}: "
        f"Dice={results['dice']:.4f}, "
        f"Thin Dice={results.get('thin_vessel_dice', float('nan')):.4f}, "
        f"Sensitivity={results['sensitivity']:.4f}, "
        f"Specificity={results['specificity']:.4f}, "
        f"Accuracy={results['accuracy']:.4f}, "
        f"AUC={results.get('auc', float('nan')):.4f}"
    )

    return results


def save_yaml(data: Mapping[str, Any], path: Path) -> None:
    """Save nested results as readable YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            dict(data),
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def main() -> None:
    config = load_config("config.yaml")

    seed = int(config.get("seed", {}).get("value", 42))
    deterministic = bool(
        config.get("seed", {}).get("deterministic", True)
    )
    seed_everything(seed, deterministic)

    device = resolve_device(config)
    settings = resolve_reconstruction_settings(config)

    print("\nEvaluation device:", device)
    print("Standardized protocol: reconstruction=True, TTA=False")
    print(
        "Reconstruction settings:",
        {
            "patch_size": settings.patch_size,
            "model_input_size": settings.model_input_size,
            "stride": settings.stride,
            "batch_size": settings.batch_size,
            "num_workers": settings.num_workers,
            "fusion_mode": settings.fusion_mode,
            "minimum_weight": settings.minimum_weight,
            "clahe": settings.clahe,
        },
    )

    checkpoint_path = resolve_checkpoint_path(config)
    output_directory = resolve_output_directory(config)
    output_directory.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        config=config,
    )

    evaluation_config = config.get("evaluation", {})
    save_outputs = bool(evaluation_config.get("save_predictions", True))

    all_results: Dict[str, Any] = {
        "protocol": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_patch_validation_best_dice": checkpoint.get("best_dice"),
            "threshold_selection_dataset": "DRIVE_internal_validation",
            "threshold_selection_metric": "macro whole-image Dice",
            "test_time_augmentation": False,
            "patch_size": settings.patch_size,
            "model_input_size": settings.model_input_size,
            "stride": settings.stride,
            "fusion_mode": settings.fusion_mode,
            "minimum_fusion_weight": settings.minimum_weight,
            "clahe": settings.clahe,
        }
    }

    # --------------------------------------------------------
    # 1. INTERNAL VALIDATION: RECONSTRUCT, THEN TUNE THRESHOLD
    # --------------------------------------------------------

    validation_pairs = reconstruct_internal_validation_pairs(config)

    print("\n" + "=" * 72)
    print("Reconstructing DRIVE_internal_validation for threshold selection")
    print("This split is used only to select the operating threshold.")
    print("=" * 72)

    validation_reconstructed, validation_targets, validation_weights = (
        reconstruct_probability_maps(
            model=model,
            samples=validation_pairs,
            settings=settings,
            device=device,
            description="Reconstructing DRIVE_internal_validation",
        )
    )

    tuned_threshold, tuned_validation_dice, sweep_scores = (
        select_validation_threshold(
            reconstructed=validation_reconstructed,
            targets=validation_targets,
            candidates=threshold_candidates(config),
        )
    )

    validation_results = summarize_reconstructed_dataset(
        dataset_name="DRIVE_internal_validation",
        samples=validation_pairs,
        reconstructed=validation_reconstructed,
        targets=validation_targets,
        weight_sums=validation_weights,
        threshold=tuned_threshold,
        output_directory=(
            output_directory
            / "predictions"
            / "DRIVE_internal_validation"
        ),
        save_outputs=save_outputs,
    )

    validation_results["threshold_sweep"] = {
        f"{threshold:.6f}": float(dice)
        for threshold, dice in sweep_scores.items()
    }
    validation_results["selected_threshold"] = tuned_threshold
    validation_results["selected_threshold_dice"] = tuned_validation_dice

    all_results["DRIVE_internal_validation"] = validation_results
    all_results["protocol"]["frozen_threshold"] = tuned_threshold

    # --------------------------------------------------------
    # 2. OFFICIAL HELD-OUT DRIVE TEST
    # --------------------------------------------------------

    test_dataset_config = config.get("dataset", {}).get("test_dataset")

    if test_dataset_config is None:
        raise KeyError(
            "config.yaml has no dataset.test_dataset entry. The official "
            "DRIVE held-out evaluation cannot be performed."
        )

    drive_test_pairs = load_configured_pairs(
        "DRIVE_test",
        test_dataset_config,
    )

    all_results["DRIVE_test"] = evaluate_one_dataset(
        dataset_name="DRIVE_test",
        samples=drive_test_pairs,
        model=model,
        settings=settings,
        device=device,
        threshold=tuned_threshold,
        experiment_output_directory=output_directory,
        save_outputs=save_outputs,
    )

    # --------------------------------------------------------
    # 3. ZERO-SHOT CROSS-DATASET GENERALIZATION
    # --------------------------------------------------------

    external_entries = normalize_external_dataset_entries(
        config.get("dataset", {}).get("external_datasets", {})
    )

    for dataset_name, dataset_config in external_entries:
        external_pairs = load_configured_pairs(
            dataset_name,
            dataset_config,
        )

        all_results[dataset_name] = evaluate_one_dataset(
            dataset_name=dataset_name,
            samples=external_pairs,
            model=model,
            settings=settings,
            device=device,
            threshold=tuned_threshold,
            experiment_output_directory=output_directory,
            save_outputs=save_outputs,
        )

    # --------------------------------------------------------
    # SAVE AND REPORT
    # --------------------------------------------------------

    results_path = output_directory / "evaluation_results.yaml"
    save_yaml(all_results, results_path)

    print("\n" + "=" * 72)
    print("Evaluation complete")
    print("=" * 72)
    print(f"Frozen validation-selected threshold: {tuned_threshold:.2f}")

    for dataset_name, results in all_results.items():
        if dataset_name == "protocol":
            continue

        print(
            f"{dataset_name:35s} "
            f"Dice={results['dice']:.4f} | "
            f"Thin={results.get('thin_vessel_dice', float('nan')):.4f} | "
            f"Se={results['sensitivity']:.4f} | "
            f"Sp={results['specificity']:.4f} | "
            f"Acc={results['accuracy']:.4f} | "
            f"AUC={results.get('auc', float('nan')):.4f}"
        )

    print("\nResults saved to:", results_path.resolve())


if __name__ == "__main__":
    main()