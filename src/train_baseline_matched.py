"""Strictly matched baseline training for MV-TransUNet-HVS experiments.

This script trains the non-topological MV-TransUNet baseline while matching the
shared data, optimization, validation, and segmentation-loss settings of the
MV-TransUNet-HVS topology experiment.

The baseline objective is:
    total = main_segmentation_loss + auxiliary_weight * auxiliary_loss

where each segmentation loss is:
    BCE_weight * BCEWithLogits + Dice_weight * SoftDice

Run from the project root:
    python -m src.train_baseline_matched \
        --config config_baseline_matched_topology_seed42.yaml
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.mv_transunet import MVTransUNet
from models.topology_losses import (
    compute_width_adaptive_weight_map,
    weighted_bce_dice_loss,
)
from src.datasets import build_dataloaders, build_patch_dataloaders


LOSS_KEYS = (
    "total_loss",
    "main_loss",
    "main_bce_loss",
    "main_dice_loss",
    "auxiliary_loss",
)


@dataclass(frozen=True)
class BaselineLossOutput:
    total_loss: torch.Tensor
    main_loss: torch.Tensor
    main_bce_loss: torch.Tensor
    main_dice_loss: torch.Tensor
    auxiliary_loss: torch.Tensor

    def as_dict(self, detach: bool = False) -> Dict[str, torch.Tensor]:
        values = {
            "total_loss": self.total_loss,
            "main_loss": self.main_loss,
            "main_bce_loss": self.main_bce_loss,
            "main_dice_loss": self.main_dice_loss,
            "auxiliary_loss": self.auxiliary_loss,
        }
        if detach:
            return {name: value.detach() for name, value in values.items()}
        return values


class SoftDiceLoss(nn.Module):
    """Batch-mean soft Dice loss computed from raw logits."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        target = target.to(dtype=probabilities.dtype)
        dimensions = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * target).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()


class MatchedBaselineLoss(nn.Module):
    """BCE-Dice main loss plus topology-matched deep supervision.

    Optionally applies vessel-width-adaptive per-pixel weighting to the
    main segmentation loss only (auxiliary deep-supervision stages remain
    unweighted). This reuses the same `compute_width_adaptive_weight_map`
    and `weighted_bce_dice_loss` functions used by the topology model's
    A8 configuration (`models/topology_losses.py`), so that the "Weighted
    Baseline" ablation isolates the width-adaptive loss term from the
    topology supervision (TFFM, skeleton/endpoint/junction heads) that
    otherwise differs between the baseline and the full topology model.
    Disabled by default (`use_width_adaptive_weighting=False`), which
    reproduces the original unweighted baseline exactly.
    """

    def __init__(
        self,
        bce_weight: float = 0.50,
        dice_weight: float = 0.50,
        auxiliary_weight: float = 0.20,
        auxiliary_stage_weights: Sequence[float] = (0.25, 0.50, 0.75),
        use_width_adaptive_weighting: bool = False,
        width_adaptive_reference_width: float = 3.0,
        width_adaptive_maximum_weight: float = 5.0,
        width_adaptive_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("BCE and Dice weights must be non-negative and not both zero.")
        if auxiliary_weight < 0:
            raise ValueError("auxiliary_weight must be non-negative.")
        if not auxiliary_stage_weights:
            raise ValueError("At least one auxiliary stage weight is required.")
        if any(weight < 0 for weight in auxiliary_stage_weights):
            raise ValueError("Auxiliary stage weights must be non-negative.")
        if width_adaptive_reference_width <= 0.0:
            raise ValueError("width_adaptive_reference_width must be positive.")
        if width_adaptive_maximum_weight < 1.0:
            raise ValueError("width_adaptive_maximum_weight must be at least 1.0.")

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.auxiliary_weight = float(auxiliary_weight)
        self.auxiliary_stage_weights = tuple(float(value) for value in auxiliary_stage_weights)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

        self.use_width_adaptive_weighting = bool(use_width_adaptive_weighting)
        self.width_adaptive_reference_width = float(width_adaptive_reference_width)
        self.width_adaptive_maximum_weight = float(width_adaptive_maximum_weight)
        self.width_adaptive_epsilon = float(width_adaptive_epsilon)

    @staticmethod
    def _extract_outputs(outputs: object) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        if torch.is_tensor(outputs):
            return outputs, ()
        if not isinstance(outputs, Mapping):
            raise TypeError("Model output must be a tensor or mapping.")
        if "main_output" not in outputs:
            raise KeyError("Model output mapping is missing 'main_output'.")
        auxiliary = outputs.get("auxiliary_outputs", ())
        if auxiliary is None:
            auxiliary = ()
        return outputs["main_output"], auxiliary

    @staticmethod
    def _match_target(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if target.shape[-2:] == logits.shape[-2:]:
            return target
        return F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")

    def _segmentation_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matched_target = self._match_target(target, logits).to(dtype=logits.dtype)
        bce_loss = self.bce(logits, matched_target)
        dice_loss = self.dice(logits, matched_target)
        combined = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return combined, bce_loss, dice_loss

    def _weighted_main_segmentation_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matched_target = self._match_target(target, logits).to(dtype=logits.dtype)
        weight_map = compute_width_adaptive_weight_map(
            mask_target=matched_target,
            reference_width=self.width_adaptive_reference_width,
            maximum_weight=self.width_adaptive_maximum_weight,
            epsilon=self.width_adaptive_epsilon,
        )
        combined, bce_loss, dice_loss = weighted_bce_dice_loss(
            logits=logits,
            targets=matched_target,
            weight_map=weight_map,
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
        )
        return combined, bce_loss, dice_loss

    def forward(self, outputs: object, target: torch.Tensor) -> BaselineLossOutput:
        main_logits, auxiliary_outputs = self._extract_outputs(outputs)
        if self.use_width_adaptive_weighting:
            main_loss, main_bce_loss, main_dice_loss = self._weighted_main_segmentation_loss(
                main_logits, target
            )
        else:
            main_loss, main_bce_loss, main_dice_loss = self._segmentation_loss(
                main_logits, target
            )

        auxiliary_loss = main_loss.new_zeros(())
        if auxiliary_outputs:
            if len(auxiliary_outputs) != len(self.auxiliary_stage_weights):
                raise ValueError(
                    "Number of auxiliary outputs does not match auxiliary_stage_weights: "
                    f"{len(auxiliary_outputs)} vs {len(self.auxiliary_stage_weights)}."
                )
            weight_sum = sum(self.auxiliary_stage_weights)
            if weight_sum <= 0:
                raise ValueError("The sum of auxiliary stage weights must be positive.")
            for stage_logits, stage_weight in zip(
                auxiliary_outputs, self.auxiliary_stage_weights
            ):
                stage_loss, _, _ = self._segmentation_loss(stage_logits, target)
                auxiliary_loss = auxiliary_loss + stage_weight * stage_loss
            auxiliary_loss = auxiliary_loss / weight_sum

        total_loss = main_loss + self.auxiliary_weight * auxiliary_loss
        return BaselineLossOutput(
            total_loss=total_loss,
            main_loss=main_loss,
            main_bce_loss=main_bce_loss,
            main_dice_loss=main_dice_loss,
            auxiliary_loss=auxiliary_loss,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the matched MV-TransUNet baseline.")
    parser.add_argument(
        "--config",
        default="config_baseline_matched_topology_seed42.yaml",
        help="Path to the matched baseline YAML configuration.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")
    return config


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(config: Mapping) -> torch.device:
    requested = str(config.get("hardware", {}).get("device", "cuda")).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_training_loaders(
    config: Mapping,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, str]:
    dataset = config["dataset"]
    train_dataset = dataset["train_dataset"]
    patch = config.get("patch_training", {})
    preprocessing = config["preprocessing"]
    loader = config["dataloader"]

    common = dict(
        image_dir=str(train_dataset["image_dir"]).strip(),
        mask_dir=str(train_dataset["mask_dir"]).strip(),
        batch_size=int(loader["batch_size"]),
        validation_ratio=float(dataset.get("validation_ratio", 0.20)),
        num_workers=int(loader.get("num_workers", 0)),
        clahe=bool(preprocessing.get("clahe", {}).get("enabled", True)),
        seed=int(dataset.get("split_seed", config["seed"]["value"])),
        pin_memory=bool(loader.get("pin_memory", True)),
        persistent_workers=bool(loader.get("persistent_workers", False)),
        drop_last=bool(loader.get("drop_last", False)),
    )
    if common["num_workers"] == 0:
        common["persistent_workers"] = False

    if bool(patch.get("enabled", True)):
        patch_size = int(patch.get("patch_size", 256))
        input_size = int(patch.get("model_input_size", 256))
        scale = patch.get("scale_augmentation", {})
        train_loader, validation_loader = build_patch_dataloaders(
            **common,
            patch_size=patch_size,
            model_input_size=input_size,
            patches_per_image=int(patch.get("patches_per_image", 100)),
            validation_stride=int(patch.get("validation_stride", 128)),
            vessel_center_probability=float(patch.get("vessel_center_probability", 0.60)),
            hard_negative_probability=float(patch.get("hard_negative_probability", 0.25)),
            random_probability=float(patch.get("random_probability", 0.15)),
            minimum_vessel_fraction=float(patch.get("minimum_vessel_fraction", 0.01)),
            hard_negative_max_vessel_fraction=float(
                patch.get("hard_negative_max_vessel_fraction", 0.005)
            ),
            hard_negative_candidates=int(patch.get("hard_negative_candidates", 20)),
            max_sampling_attempts=int(patch.get("max_sampling_attempts", 20)),
            include_empty_validation_patches=bool(
                patch.get("include_empty_validation_patches", True)
            ),
            scale_augmentation_enabled=bool(scale.get("enabled", False)),
            scale_augmentation_probability=float(scale.get("probability", 0.0)),
            scale_augmentation_min=float(scale.get("min_scale", 1.0)),
            scale_augmentation_max=float(scale.get("max_scale", 1.0)),
        )
        return (
            train_loader,
            validation_loader,
            f"matched patch training ({patch_size} -> {input_size})",
        )

    image_size = int(preprocessing["image_size"]["height"])
    train_loader, validation_loader = build_dataloaders(
        **common,
        image_size=image_size,
    )
    return (
        train_loader,
        validation_loader,
        f"matched whole-image training ({image_size}x{image_size})",
    )


def build_model(config: Mapping) -> MVTransUNet:
    model = config.get("model", {})
    return MVTransUNet(
        pretrained=bool(model.get("backbone", {}).get("pretrained", True)),
        transformer_channels=int(model.get("transformer", {}).get("embed_dim", 768)),
        vessel_reduction_ratio=int(
            model.get("vessel_attention", {}).get("reduction_ratio", 16)
        ),
        output_channels=int(model.get("output_channels", 1)),
        deep_supervision=bool(model.get("deep_supervision", {}).get("enabled", True)),
        decoder_dropout_rate=float(model.get("decoder", {}).get("dropout_rate", 0.10)),
        vessel_attention_scale_conditioning=bool(
            model.get("vessel_attention", {}).get("scale_conditioning", False)
        ),
    )


def build_criterion(config: Mapping) -> MatchedBaselineLoss:
    loss = config.get("baseline_loss", {})
    return MatchedBaselineLoss(
        bce_weight=float(loss.get("segmentation_bce_weight", 0.50)),
        dice_weight=float(loss.get("segmentation_dice_weight", 0.50)),
        auxiliary_weight=float(loss.get("auxiliary_weight", 0.20)),
        auxiliary_stage_weights=tuple(
            float(value)
            for value in loss.get("auxiliary_stage_weights", [0.25, 0.50, 0.75])
        ),
        use_width_adaptive_weighting=bool(
            loss.get("use_width_adaptive_weighting", False)
        ),
        width_adaptive_reference_width=float(
            loss.get("width_adaptive_reference_width", 3.0)
        ),
        width_adaptive_maximum_weight=float(
            loss.get("width_adaptive_maximum_weight", 5.0)
        ),
        width_adaptive_epsilon=float(
            loss.get("width_adaptive_epsilon", 1e-3)
        ),
    )


def dice_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    prediction = (torch.sigmoid(logits) >= threshold).to(target.dtype)
    dimensions = tuple(range(1, prediction.ndim))
    intersection = (prediction * target).sum(dim=dimensions)
    denominator = prediction.sum(dim=dimensions) + target.sum(dim=dimensions)
    return ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def extract_main_output(outputs: object) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, Mapping) and "main_output" in outputs:
        return outputs["main_output"]
    raise TypeError("Could not extract main segmentation logits from model output.")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MatchedBaselineLoss,
    device: torch.device,
    threshold: float,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    accumulation_steps: int = 1,
    clip_enabled: bool = True,
    clip_max_norm: float = 1.0,
) -> Dict[str, float]:
    training = optimizer is not None
    if training and scaler is None:
        raise ValueError("A GradScaler is required during training.")
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    if len(loader) == 0:
        raise RuntimeError("DataLoader contains zero batches.")

    model.train(training)
    totals = {key: 0.0 for key in LOSS_KEYS}
    totals["dice"] = 0.0
    valid_batches = 0
    skipped_batches = 0

    if training:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad if training else torch.inference_mode
    progress = tqdm(loader, desc="Training" if training else "Validation", leave=True)

    with context():
        for step, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True).float()
            masks = batch["mask"].to(device, non_blocking=True).float()

            if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
                skipped_batches += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)
                continue

            # See the matching comment in src/train_topology.py: RandomPatchRetinalDataset
            # (training) returns a ground-truth per-patch "scale_factor"; GridPatchRetinalDataset
            # (validation) does not, so default to 1.0 (no scale change) there.
            if "scale_factor" in batch:
                scale_ratio = batch["scale_factor"].to(device, non_blocking=True).float()
            else:
                scale_ratio = torch.ones(images.shape[0], device=device, dtype=images.dtype)

            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images, scale_ratio=scale_ratio, return_auxiliary=True)
                loss_output = criterion(outputs, masks)
                losses = loss_output.as_dict(detach=False)
                total_loss = losses["total_loss"]

            main_logits = extract_main_output(outputs)
            finite = torch.isfinite(main_logits).all() and all(
                torch.isfinite(value).all() for value in losses.values()
            )
            if not finite:
                skipped_batches += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if training:
                scaler.scale(total_loss / accumulation_steps).backward()
                should_step = ((step + 1) % accumulation_steps == 0) or (
                    step + 1 == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        clip_max_norm if clip_enabled else float("inf"),
                    )
                    if torch.isfinite(gradient_norm):
                        scaler.step(optimizer)
                    else:
                        skipped_batches += 1
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            batch_dice = dice_score(main_logits.detach(), masks, threshold)
            for key in LOSS_KEYS:
                totals[key] += float(losses[key].detach().item())
            totals["dice"] += float(batch_dice.item())
            valid_batches += 1
            progress.set_postfix(
                total=f"{total_loss.item():.4f}",
                main=f"{losses['main_loss'].item():.4f}",
                aux=f"{losses['auxiliary_loss'].item():.4f}",
                dice=f"{batch_dice.item():.4f}",
                skipped=skipped_batches,
            )

    if valid_batches == 0:
        raise RuntimeError("No valid batches were processed.")
    if skipped_batches > max(1, int(0.05 * len(loader))):
        raise RuntimeError(
            f"Skipped {skipped_batches}/{len(loader)} batches, exceeding the 5% safety limit."
        )

    metrics = {name: value / valid_batches for name, value in totals.items()}
    metrics["skipped_batches"] = float(skipped_batches)
    return metrics


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: GradScaler,
    best_dice: float,
    patience_counter: int,
    config: Mapping,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_dice": best_dice,
            "patience_counter": patience_counter,
            "model_name": "MVTransUNet",
            "experiment": "matched_topology_control",
            "config": dict(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_dice", 0.0)),
        int(checkpoint.get("patience_counter", 0)),
    )


def log_metrics(
    writer: SummaryWriter,
    split: str,
    metrics: Mapping[str, float],
    epoch: int,
) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{split}/{name}", value, epoch)


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config["seed"]["value"])
    seed_everything(seed, bool(config["seed"].get("deterministic", True)))
    device = resolve_device(config)
    amp_enabled = (
        bool(config.get("hardware", {}).get("mixed_precision", True))
        and device.type == "cuda"
    )

    train_loader, validation_loader, mode = build_training_loaders(config)
    model = build_model(config).to(device)
    criterion = build_criterion(config).to(device)

    optimizer_config = config["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("epsilon", 1e-8)),
    )
    scheduler_config = config.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 8)),
        threshold=float(scheduler_config.get("threshold", 1e-4)),
        min_lr=float(scheduler_config.get("min_lr", 1e-6)),
    )
    scaler = GradScaler("cuda", enabled=amp_enabled)

    checkpoint_config = config["checkpoint"]
    checkpoint_directory = Path(checkpoint_config["directory"])
    best_path = checkpoint_directory / checkpoint_config.get(
        "best_model_name", "best_model.pth"
    )
    last_path = checkpoint_directory / checkpoint_config.get(
        "last_model_name", "last_model.pth"
    )

    logging_config = config.get("logging", {})
    writer = None
    if bool(logging_config.get("tensorboard", {}).get("enabled", True)):
        writer = SummaryWriter(str(Path(logging_config["directory"])))

    training_config = config["training"]
    epochs = int(training_config["epochs"])
    accumulation_steps = int(training_config.get("gradient_accumulation_steps", 1))
    clipping = training_config.get("gradient_clipping", {})
    early_stopping = training_config.get("early_stopping", {})
    threshold = float(config.get("validation", {}).get("threshold", 0.5))

    start_epoch, best_dice, patience_counter = 0, 0.0, 0
    if bool(checkpoint_config.get("resume", False)):
        resume_path = (
            best_path
            if bool(checkpoint_config.get("resume_from_best", False))
            else last_path
        )
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path.resolve()}")
        start_epoch, best_dice, patience_counter = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )

    print("=" * 78)
    print("MV-TransUNet Strictly Matched Baseline Training")
    print("=" * 78)
    print("Config:", Path(config_path).resolve())
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    print("Mixed precision:", amp_enabled)
    print("Mode:", mode)
    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(validation_loader.dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Effective batch size:", int(config["dataloader"]["batch_size"]) * accumulation_steps)
    print("Validation threshold:", threshold)
    print(
        "Trainable parameters:",
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}",
    )
    print("Loss: 0.50 BCE + 0.50 Dice, auxiliary weight 0.20")
    print("=" * 78)

    try:
        for epoch in range(start_epoch, epochs):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                threshold=threshold,
                amp_enabled=amp_enabled,
                optimizer=optimizer,
                scaler=scaler,
                accumulation_steps=accumulation_steps,
                clip_enabled=bool(clipping.get("enabled", True)),
                clip_max_norm=float(clipping.get("max_norm", 1.0)),
            )
            validation_metrics = run_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                threshold=threshold,
                amp_enabled=amp_enabled,
            )
            scheduler.step(validation_metrics["dice"])

            current_epoch = epoch + 1
            if writer is not None:
                log_metrics(writer, "train", train_metrics, current_epoch)
                log_metrics(writer, "validation", validation_metrics, current_epoch)
                writer.add_scalar(
                    "optimization/learning_rate",
                    optimizer.param_groups[0]["lr"],
                    current_epoch,
                )

            improved = validation_metrics["dice"] > best_dice
            if improved:
                best_dice = validation_metrics["dice"]
                patience_counter = 0
                if bool(checkpoint_config.get("save_best", True)):
                    save_checkpoint(
                        best_path,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        best_dice,
                        patience_counter,
                        config,
                    )
            else:
                patience_counter += 1

            if bool(checkpoint_config.get("save_last", True)):
                save_checkpoint(
                    last_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_dice,
                    patience_counter,
                    config,
                )

            print(
                f"Epoch {current_epoch:03d}/{epochs} | "
                f"train Dice {train_metrics['dice']:.4f} | "
                f"val Dice {validation_metrics['dice']:.4f} | "
                f"best {best_dice:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}"
            )
            print(
                "  val losses: "
                f"total={validation_metrics['total_loss']:.4f}, "
                f"main={validation_metrics['main_loss']:.4f}, "
                f"BCE={validation_metrics['main_bce_loss']:.4f}, "
                f"DiceLoss={validation_metrics['main_dice_loss']:.4f}, "
                f"aux={validation_metrics['auxiliary_loss']:.4f}"
            )

            if bool(early_stopping.get("enabled", True)) and patience_counter >= int(
                early_stopping.get("patience", 35)
            ):
                print(
                    f"Early stopping after {patience_counter} epochs without "
                    "validation Dice improvement."
                )
                break
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.config)
