"""End-to-end training engine for topology-aware MV-TransUNet."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.mv_transunet_topology import MVTransUNetTopology
from models.topology_losses import MVTransUNetTopologyLoss, TopologyLossOutput
from src.topology_datasets import (
    build_topology_dataloaders,
    build_topology_patch_dataloaders,
)

LOSS_KEYS = (
    "total_loss", "segmentation_loss", "segmentation_bce_loss",
    "segmentation_dice_loss", "cldice_loss", "skeleton_loss",
    "skeleton_bce_loss", "skeleton_dice_loss", "endpoint_loss",
    "endpoint_bce_loss", "endpoint_focal_loss", "junction_loss",
    "junction_bce_loss", "junction_focal_loss", "auxiliary_loss",
    "baseline_loss",
)
TARGET_KEYS = ("mask", "skeleton", "endpoint", "junction")


def load_config(path: str = "config_topology.yaml") -> Dict:
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
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_training_loaders(config: Mapping) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, str]:
    dataset = config["dataset"]
    train_dataset = dataset["train_dataset"]
    topology = config["topology"]
    patch = config.get("patch_training", {})
    preprocessing = config["preprocessing"]
    loader = config["dataloader"]

    common = dict(
        image_dir=str(train_dataset["image_dir"]).strip(),
        mask_dir=str(train_dataset["mask_dir"]).strip(),
        topology_root=str(topology["target_root"]).strip(),
        batch_size=int(loader["batch_size"]),
        validation_ratio=float(dataset.get("validation_ratio", 0.2)),
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
        train_loader, val_loader = build_topology_patch_dataloaders(
            **common,
            patch_size=patch_size,
            model_input_size=input_size,
            patches_per_image=int(patch.get("patches_per_image", 100)),
            validation_stride=int(patch.get("validation_stride", patch_size // 2)),
            vessel_center_probability=float(patch.get("vessel_center_probability", 0.60)),
            hard_negative_probability=float(patch.get("hard_negative_probability", 0.25)),
            random_probability=float(patch.get("random_probability", 0.15)),
            minimum_vessel_fraction=float(patch.get("minimum_vessel_fraction", 0.01)),
            hard_negative_max_vessel_fraction=float(patch.get("hard_negative_max_vessel_fraction", 0.005)),
            hard_negative_candidates=int(patch.get("hard_negative_candidates", 20)),
            max_sampling_attempts=int(patch.get("max_sampling_attempts", 20)),
            include_empty_validation_patches=bool(patch.get("include_empty_validation_patches", True)),
        )
        return train_loader, val_loader, f"topology patch training ({patch_size} -> {input_size})"

    image_size = int(preprocessing["image_size"]["height"])
    train_loader, val_loader = build_topology_dataloaders(**common, image_size=image_size)
    return train_loader, val_loader, f"topology whole-image training ({image_size}x{image_size})"


def build_model(config: Mapping) -> MVTransUNetTopology:
    model = config.get("model", {})
    topology = model.get("topology_branch", {})
    return MVTransUNetTopology(
        pretrained=bool(model.get("backbone", {}).get("pretrained", True)),
        transformer_channels=int(model.get("transformer", {}).get("embed_dim", 768)),
        vessel_reduction_ratio=int(model.get("vessel_attention", {}).get("reduction_ratio", 16)),
        output_channels=int(model.get("output_channels", 1)),
        deep_supervision=bool(model.get("deep_supervision", {}).get("enabled", True)),
        decoder_dropout_rate=float(model.get("decoder", {}).get("dropout_rate", 0.1)),
        topology_channels=int(topology.get("topology_channels", 128)),
        topology_branch_channels=int(topology.get("branch_channels", 32)),
        topology_head_hidden_channels=int(topology.get("head_hidden_channels", 64)),
        topology_refinement_blocks=int(topology.get("refinement_blocks", 2)),
        topology_fusion_dropout=float(topology.get("fusion_dropout", 0.1)),
        topology_head_dropout=float(topology.get("head_dropout", 0.0)),
    )


def build_criterion(config: Mapping) -> MVTransUNetTopologyLoss:
    loss = config.get("topology_loss", {})
    return MVTransUNetTopologyLoss(
        segmentation_weight=float(loss.get("segmentation_weight", 1.0)),
        cldice_weight=float(loss.get("cldice_weight", 0.30)),
        skeleton_weight=float(loss.get("skeleton_weight", 0.30)),
        endpoint_weight=float(loss.get("endpoint_weight", 0.10)),
        junction_weight=float(loss.get("junction_weight", 0.10)),
        auxiliary_weight=float(loss.get("auxiliary_weight", 0.20)),
        baseline_weight=float(loss.get("baseline_weight", 0.10)),
        segmentation_bce_weight=float(loss.get("segmentation_bce_weight", 0.50)),
        segmentation_dice_weight=float(loss.get("segmentation_dice_weight", 0.50)),
        skeleton_bce_weight=float(loss.get("skeleton_bce_weight", 0.50)),
        skeleton_dice_weight=float(loss.get("skeleton_dice_weight", 0.50)),
        node_bce_weight=float(loss.get("node_bce_weight", 0.50)),
        node_focal_weight=float(loss.get("node_focal_weight", 0.50)),
        node_maximum_positive_weight=float(loss.get("node_maximum_positive_weight", 50.0)),
        node_focal_alpha=float(loss.get("node_focal_alpha", 0.75)),
        node_focal_gamma=float(loss.get("node_focal_gamma", 2.0)),
        cldice_iterations=int(loss.get("cldice_iterations", 10)),
        auxiliary_stage_weights=tuple(float(x) for x in loss.get("auxiliary_stage_weights", [0.25, 0.50, 0.75])),
    )


def move_targets(batch: Mapping, device: torch.device) -> Dict[str, torch.Tensor]:
    targets = {}
    for key in TARGET_KEYS:
        if key not in batch:
            raise KeyError(f"Dataset batch is missing required target '{key}'.")
        targets[key] = batch[key].to(device, non_blocking=True).float()
    return targets


def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    prediction = (torch.sigmoid(logits) >= threshold).to(target.dtype)
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction * target).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def loss_to_dict(output: TopologyLossOutput) -> Dict[str, torch.Tensor]:
    values = output.as_dict(detach=False)
    missing = set(LOSS_KEYS) - set(values)
    if missing:
        raise KeyError(f"Topology loss output is missing: {sorted(missing)}")
    return values


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MVTransUNetTopologyLoss,
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
    model.train(training)
    totals = {key: 0.0 for key in LOSS_KEYS}
    totals["dice"] = 0.0
    valid_batches = 0
    skipped_batches = 0

    if training:
        assert scaler is not None
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad if training else torch.inference_mode
    progress = tqdm(loader, desc="Training" if training else "Validation", leave=True)
    with context():
        for step, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True).float()
            targets = move_targets(batch, device)
            all_inputs = [images, *targets.values()]
            if not all(torch.isfinite(tensor).all() for tensor in all_inputs):
                skipped_batches += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)
                continue

            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images, return_topology=True)
                loss_values = loss_to_dict(criterion(outputs, targets))
                total_loss = loss_values["total_loss"]

            finite = all(torch.isfinite(value).all() for value in loss_values.values())
            finite = finite and torch.isfinite(outputs["topology_segmentation_logits"]).all()
            if not finite:
                skipped_batches += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if training:
                scaler.scale(total_loss / accumulation_steps).backward()
                should_step = ((step + 1) % accumulation_steps == 0) or (step + 1 == len(loader))
                if should_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_max_norm if clip_enabled else float("inf")
                    )
                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        skipped_batches += 1
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            batch_dice = dice_score(outputs["topology_segmentation_logits"].detach(), targets["mask"], threshold)
            for key in LOSS_KEYS:
                totals[key] += float(loss_values[key].detach().item())
            totals["dice"] += float(batch_dice.item())
            valid_batches += 1
            progress.set_postfix(total=f"{total_loss.item():.4f}", seg=f"{loss_values['segmentation_loss'].item():.4f}", topo=f"{loss_values['cldice_loss'].item():.4f}", dice=f"{batch_dice.item():.4f}", skipped=skipped_batches)

    if valid_batches == 0:
        raise RuntimeError("No valid batches were processed.")
    if skipped_batches > max(1, int(0.05 * len(loader))):
        raise RuntimeError(f"Skipped {skipped_batches}/{len(loader)} batches, exceeding the 5% safety limit.")
    metrics = {key: value / valid_batches for key, value in totals.items()}
    metrics["skipped_batches"] = float(skipped_batches)
    return metrics


def save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler, scaler, best_dice: float, patience_counter: int, config: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_dice": best_dice,
        "patience_counter": patience_counter,
        "model_name": "MVTransUNetTopology",
        "config": dict(config),
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device) -> Tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_dice", 0.0)), int(checkpoint.get("patience_counter", 0))


def log_metrics(writer: SummaryWriter, split: str, metrics: Mapping[str, float], epoch: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{split}/{name}", value, epoch)


def main(config_path: str = "config_topology.yaml") -> None:
    config = load_config(config_path)
    seed = int(config["seed"]["value"])
    seed_everything(seed, bool(config["seed"].get("deterministic", True)))
    device = resolve_device(config)
    amp_enabled = bool(config.get("hardware", {}).get("mixed_precision", True)) and device.type == "cuda"

    train_loader, val_loader, mode = build_training_loaders(config)
    model = build_model(config).to(device)
    criterion = build_criterion(config).to(device)

    optimizer_cfg = config["optimizer"]
    optimizer = AdamW(model.parameters(), lr=float(optimizer_cfg["learning_rate"]), weight_decay=float(optimizer_cfg["weight_decay"]), betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])), eps=float(optimizer_cfg.get("epsilon", 1e-8)))
    scheduler_cfg = config.get("scheduler", {})
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=float(scheduler_cfg.get("factor", 0.5)), patience=int(scheduler_cfg.get("patience", 8)), threshold=float(scheduler_cfg.get("threshold", 1e-4)), min_lr=float(scheduler_cfg.get("min_lr", 1e-6)))
    scaler = GradScaler("cuda", enabled=amp_enabled)

    checkpoint_cfg = config["checkpoint"]
    checkpoint_dir = Path(checkpoint_cfg["directory"])
    best_path = checkpoint_dir / checkpoint_cfg.get("best_model_name", "best_model.pth")
    last_path = checkpoint_dir / checkpoint_cfg.get("last_model_name", "last_model.pth")
    log_dir = Path(config["logging"]["directory"])
    writer = SummaryWriter(str(log_dir)) if bool(config.get("logging", {}).get("tensorboard", {}).get("enabled", True)) else None

    training_cfg = config["training"]
    epochs = int(training_cfg["epochs"])
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    clipping = training_cfg.get("gradient_clipping", {})
    early = training_cfg.get("early_stopping", {})
    threshold = float(config.get("validation", {}).get("threshold", 0.5))

    start_epoch, best_dice, patience_counter = 0, 0.0, 0
    if bool(checkpoint_cfg.get("resume", False)):
        resume_path = best_path if bool(checkpoint_cfg.get("resume_from_best", False)) else last_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path.resolve()}")
        start_epoch, best_dice, patience_counter = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)

    print("=" * 72)
    print("Topology-Aware MV-TransUNet Training")
    print("=" * 72)
    print("Device:", device)
    if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    print("AMP:", amp_enabled)
    print("Mode:", mode)
    print("Train samples:", len(train_loader.dataset), "Validation samples:", len(val_loader.dataset))
    print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    try:
        for epoch in range(start_epoch, epochs):
            train_metrics = run_epoch(model, train_loader, criterion, device, threshold, amp_enabled, optimizer, scaler, accumulation, bool(clipping.get("enabled", True)), float(clipping.get("max_norm", 1.0)))
            val_metrics = run_epoch(model, val_loader, criterion, device, threshold, amp_enabled)
            scheduler.step(val_metrics["dice"])

            if writer is not None:
                log_metrics(writer, "train", train_metrics, epoch + 1)
                log_metrics(writer, "validation", val_metrics, epoch + 1)
                writer.add_scalar("optimization/learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)

            improved = val_metrics["dice"] > best_dice
            if improved:
                best_dice = val_metrics["dice"]
                patience_counter = 0
                if bool(checkpoint_cfg.get("save_best", True)):
                    save_checkpoint(best_path, epoch, model, optimizer, scheduler, scaler, best_dice, patience_counter, config)
            else:
                patience_counter += 1

            if bool(checkpoint_cfg.get("save_last", True)):
                save_checkpoint(last_path, epoch, model, optimizer, scheduler, scaler, best_dice, patience_counter, config)

            print(f"Epoch {epoch + 1:03d}/{epochs} | train Dice {train_metrics['dice']:.4f} | val Dice {val_metrics['dice']:.4f} | best {best_dice:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  val losses: total={val_metrics['total_loss']:.4f}, seg={val_metrics['segmentation_loss']:.4f}, clDice={val_metrics['cldice_loss']:.4f}, skeleton={val_metrics['skeleton_loss']:.4f}, endpoint={val_metrics['endpoint_loss']:.4f}, junction={val_metrics['junction_loss']:.4f}")

            if bool(early.get("enabled", True)) and patience_counter >= int(early.get("patience", 35)):
                print(f"Early stopping after {patience_counter} epochs without improvement.")
                break
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()