"""
MV-TransUNet Training Engine

Patch-Based / Whole-Image Compatible
Colab GPU Optimized

Features:
- YAML configuration
- Whole-image or vessel-centred patch training
- Deterministic image-level train-validation splitting
- Mixed precision training
- AdamW optimizer
- Cosine annealing scheduler
- Gradient accumulation
- Correct unscaled loss reporting
- Gradient clipping
- Deep-supervision loss support
- TensorBoard logging
- Best-model saving
- Last-checkpoint saving
- Resume training
- Early stopping
- Training and validation Dice

Important:
- Patch training is enabled through `patch_training.enabled` in config.yaml.
- Image-level splitting occurs before patch extraction, preventing leakage.
- Whole-image training remains available for controlled ablation studies.
"""


import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.losses import MVTransUNetLoss
from models.mv_transunet import MVTransUNet
from src.datasets import (
    build_dataloaders,
    build_patch_dataloaders,
)


# ============================================================
# RANDOM SEED
# ============================================================


def seed_everything(
    seed: int,
    deterministic: bool = True,
) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ============================================================
# CONFIGURATION
# ============================================================


def load_config(
    path: str,
) -> Dict:
    """Load a YAML configuration file."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration file is empty or invalid."
        )

    return config


# ============================================================
# MODEL OUTPUT HELPER
# ============================================================


def get_main_output(
    outputs,
) -> torch.Tensor:
    """
    Extract final segmentation logits.

    Deep-supervision training returns a dictionary.
    Evaluation normally returns a tensor.
    """

    if torch.is_tensor(outputs):
        return outputs

    if isinstance(outputs, dict):
        if "main_output" not in outputs:
            raise KeyError(
                "Model output dictionary does not contain 'main_output'."
            )

        return outputs["main_output"]

    raise TypeError(
        "Model output must be a tensor or dictionary."
    )


# ============================================================
# DICE SCORE
# ============================================================


def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute mean binary Dice from raw segmentation logits."""

    probability = torch.sigmoid(
        prediction
    )

    binary_prediction = (
        probability > threshold
    ).to(
        dtype=target.dtype
    )

    dimensions = tuple(
        range(
            1,
            binary_prediction.ndim,
        )
    )

    intersection = torch.sum(
        binary_prediction * target,
        dim=dimensions,
    )

    prediction_sum = torch.sum(
        binary_prediction,
        dim=dimensions,
    )

    target_sum = torch.sum(
        target,
        dim=dimensions,
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        prediction_sum
        + target_sum
        + smooth
    )

    return dice.mean()


# ============================================================
# DATA PIPELINE BUILDER
# ============================================================


def build_training_loaders(
    config: Dict,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    str,
]:
    """
    Build whole-image or patch-based loaders from config.yaml.

    Patch-based mode is selected when:
        patch_training.enabled: true
    """

    dataset_config = config["dataset"]
    dataloader_config = config["dataloader"]
    preprocessing_config = config["preprocessing"]
    patch_config = config.get(
        "patch_training",
        {},
    )

    image_dir = str(
    dataset_config[
        "train_dataset"
    ][
        "image_dir"
    ]
    ).strip()

    mask_dir = str(
        dataset_config[
        "train_dataset"
    ][
        "mask_dir"
    ]
      ).strip()

    image_size = int(
        preprocessing_config[
            "image_size"
        ][
            "height"
        ]
    )

    batch_size = int(
        dataloader_config[
            "batch_size"
        ]
    )

    validation_ratio = float(
        dataset_config.get(
            "validation_ratio",
            0.2,
        )
    )

    split_seed = int(
        dataset_config.get(
            "split_seed",
            config["seed"]["value"],
        )
    )

    num_workers = int(
        dataloader_config.get(
            "num_workers",
            2,
        )
    )

    clahe_enabled = bool(
        preprocessing_config.get(
            "clahe",
            {},
        ).get(
            "enabled",
            True,
        )
    )

    pin_memory = bool(
        dataloader_config.get(
            "pin_memory",
            True,
        )
    )

    persistent_workers = bool(
        dataloader_config.get(
            "persistent_workers",
            False,
        )
    )

    drop_last = bool(
        dataloader_config.get(
            "drop_last",
            False,
        )
    )

    patch_training_enabled = bool(
        patch_config.get(
            "enabled",
            False,
        )
    )

    if patch_training_enabled:
        model_input_size = int(
            patch_config.get(
                "model_input_size",
                image_size,
            )
        )

        patch_size = int(
            patch_config.get(
                "patch_size",
                model_input_size,
            )
        )

        train_loader, validation_loader = (
            build_patch_dataloaders(
                image_dir=image_dir,
                mask_dir=mask_dir,
                patch_size=patch_size,
                model_input_size=model_input_size,
                patches_per_image=int(
                    patch_config.get(
                        "patches_per_image",
                        100,
                    )
                ),
                validation_stride=int(
                    patch_config.get(
                        "validation_stride",
                        max(
                            1,
                            patch_size // 2,
                        ),
                    )
                ),
                batch_size=batch_size,
                validation_ratio=validation_ratio,
                vessel_center_probability=float(
                    patch_config.get(
                        "vessel_center_probability",
                        0.70,
                    )
                ),
                minimum_vessel_fraction=float(
                    patch_config.get(
                        "minimum_vessel_fraction",
                        0.01,
                    )
                ),
                max_sampling_attempts=int(
                    patch_config.get(
                        "max_sampling_attempts",
                        20,
                    )
                ),
                num_workers=num_workers,
                clahe=clahe_enabled,
                seed=split_seed,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=drop_last,
                include_empty_validation_patches=bool(
                    patch_config.get(
                        "include_empty_validation_patches",
                        True,
                    )
                ),
            )
        )

        mode_description = (
            "patch-based "
            f"(patch={patch_size}, input={model_input_size})"
        )

    else:
        train_loader, validation_loader = (
            build_dataloaders(
                image_dir=image_dir,
                mask_dir=mask_dir,
                image_size=image_size,
                batch_size=batch_size,
                validation_ratio=validation_ratio,
                num_workers=num_workers,
                clahe=clahe_enabled,
                seed=split_seed,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=drop_last,
            )
        )

        mode_description = (
            f"whole-image ({image_size}x{image_size})"
        )

    return (
        train_loader,
        validation_loader,
        mode_description,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int,
    gradient_clip_enabled: bool,
    gradient_clip_max_norm: float,
) -> Dict[str, float]:
    """
    Train for one epoch.

    The real loss is reported.
    Only the backward loss is divided for accumulation.
    """

    if accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be at least 1."
        )

    if len(loader) == 0:
        raise RuntimeError(
            "Training DataLoader contains zero batches."
        )

    model.train()

    running_total_loss = 0.0
    running_main_loss = 0.0
    running_auxiliary_loss = 0.0
    running_bce_loss = 0.0
    running_dice_loss = 0.0
    running_boundary_loss = 0.0
    running_dice_score = 0.0

    optimizer.zero_grad(
        set_to_none=True
    )

    progress = tqdm(
        loader,
        desc="Training",
        leave=True,
    )

    for step, batch in enumerate(progress):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        amp_enabled = (
            device.type == "cuda"
        )

        with autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            outputs = model(
                images
            )

            loss_dict = criterion(
                outputs,
                masks,
            )

            total_loss = loss_dict[
                "total_loss"
            ]

            backward_loss = (
                total_loss
                / accumulation_steps
            )

        scaler.scale(
            backward_loss
        ).backward()

        should_update = (
            (step + 1) % accumulation_steps == 0
            or
            step == len(loader) - 1
        )

        if should_update:
            scaler.unscale_(
                optimizer
            )

            if gradient_clip_enabled:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_max_norm,
                )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        main_output = get_main_output(
            outputs
        )

        batch_dice = dice_score(
            main_output.detach(),
            masks,
        )

        total_loss_value = float(
            total_loss.detach().item()
        )

        main_loss_value = float(
            loss_dict.get(
                "main_loss",
                total_loss,
            ).detach().item()
        )

        auxiliary_loss_value = float(
            loss_dict.get(
                "auxiliary_loss",
                total_loss * 0.0,
            ).detach().item()
        )

        bce_loss_value = float(
            loss_dict[
                "bce_loss"
            ].detach().item()
        )

        dice_loss_value = float(
            loss_dict[
                "dice_loss"
            ].detach().item()
        )

        boundary_loss_value = float(
            loss_dict[
                "boundary_loss"
            ].detach().item()
        )

        batch_dice_value = float(
            batch_dice.detach().item()
        )

        running_total_loss += total_loss_value
        running_main_loss += main_loss_value
        running_auxiliary_loss += auxiliary_loss_value
        running_bce_loss += bce_loss_value
        running_dice_loss += dice_loss_value
        running_boundary_loss += boundary_loss_value
        running_dice_score += batch_dice_value

        progress.set_postfix(
            total=f"{total_loss_value:.4f}",
            main=f"{main_loss_value:.4f}",
            aux=f"{auxiliary_loss_value:.4f}",
            dice=f"{batch_dice_value:.4f}",
        )

    number_of_batches = len(
        loader
    )

    return {
        "total_loss": (
            running_total_loss
            / number_of_batches
        ),
        "main_loss": (
            running_main_loss
            / number_of_batches
        ),
        "auxiliary_loss": (
            running_auxiliary_loss
            / number_of_batches
        ),
        "bce_loss": (
            running_bce_loss
            / number_of_batches
        ),
        "dice_loss": (
            running_dice_loss
            / number_of_batches
        ),
        "boundary_loss": (
            running_boundary_loss
            / number_of_batches
        ),
        "dice": (
            running_dice_score
            / number_of_batches
        ),
    }


# ============================================================
# VALIDATION
# ============================================================


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on validation images or grid patches."""

    if len(loader) == 0:
        raise RuntimeError(
            "Validation DataLoader contains zero batches."
        )

    model.eval()

    running_total_loss = 0.0
    running_main_loss = 0.0
    running_auxiliary_loss = 0.0
    running_bce_loss = 0.0
    running_dice_loss = 0.0
    running_boundary_loss = 0.0
    running_dice_score = 0.0

    progress = tqdm(
        loader,
        desc="Validation",
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

            amp_enabled = (
                device.type == "cuda"
            )

            with autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                outputs = model(
                    images
                )

                loss_dict = criterion(
                    outputs,
                    masks,
                )

            main_output = get_main_output(
                outputs
            )

            batch_dice = dice_score(
                main_output,
                masks,
            )

            total_loss_value = float(
                loss_dict[
                    "total_loss"
                ].item()
            )

            main_loss_value = float(
                loss_dict.get(
                    "main_loss",
                    loss_dict["total_loss"],
                ).item()
            )

            auxiliary_loss_value = float(
                loss_dict.get(
                    "auxiliary_loss",
                    loss_dict["total_loss"] * 0.0,
                ).item()
            )

            bce_loss_value = float(
                loss_dict[
                    "bce_loss"
                ].item()
            )

            dice_loss_value = float(
                loss_dict[
                    "dice_loss"
                ].item()
            )

            boundary_loss_value = float(
                loss_dict[
                    "boundary_loss"
                ].item()
            )

            batch_dice_value = float(
                batch_dice.item()
            )

            running_total_loss += total_loss_value
            running_main_loss += main_loss_value
            running_auxiliary_loss += auxiliary_loss_value
            running_bce_loss += bce_loss_value
            running_dice_loss += dice_loss_value
            running_boundary_loss += boundary_loss_value
            running_dice_score += batch_dice_value

            progress.set_postfix(
                loss=f"{total_loss_value:.4f}",
                dice=f"{batch_dice_value:.4f}",
            )

    number_of_batches = len(
        loader
    )

    return {
        "total_loss": (
            running_total_loss
            / number_of_batches
        ),
        "main_loss": (
            running_main_loss
            / number_of_batches
        ),
        "auxiliary_loss": (
            running_auxiliary_loss
            / number_of_batches
        ),
        "bce_loss": (
            running_bce_loss
            / number_of_batches
        ),
        "dice_loss": (
            running_dice_loss
            / number_of_batches
        ),
        "boundary_loss": (
            running_boundary_loss
            / number_of_batches
        ),
        "dice": (
            running_dice_score
            / number_of_batches
        ),
    }


# ============================================================
# CHECKPOINTS
# ============================================================


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    epoch: int,
    best_dice: float,
    patience_counter: int,
    path: str,
) -> None:
    """Save complete training state."""

    checkpoint_path = Path(
        path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_dice": best_dice,
            "patience_counter": patience_counter,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float, int]:
    """
    Restore complete training state.

    Returns:
        next_epoch, best_dice, patience_counter
    """

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state"]
    )

    if "scaler_state" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state"]
        )

    saved_epoch = int(
        checkpoint["epoch"]
    )

    best_dice = float(
        checkpoint.get(
            "best_dice",
            checkpoint.get(
                "dice",
                0.0,
            ),
        )
    )

    patience_counter = int(
        checkpoint.get(
            "patience_counter",
            0,
        )
    )

    return (
        saved_epoch + 1,
        best_dice,
        patience_counter,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    config = load_config(
        "config.yaml"
    )

    seed_value = int(
        config["seed"]["value"]
    )

    deterministic = bool(
        config["seed"].get(
            "deterministic",
            True,
        )
    )

    seed_everything(
        seed=seed_value,
        deterministic=deterministic,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print("=" * 70)
    print("MV-TransUNet Training")
    print("=" * 70)
    print(
        "Device:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "GPU memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_loader,
        validation_loader,
        data_mode,
    ) = build_training_loaders(
        config
    )

    print()
    print("Dataset summary")
    print("-" * 70)
    print(
        "Training mode:",
        data_mode,
    )
    print(
        "Training samples:",
        len(train_loader.dataset),
    )
    print(
        "Validation samples:",
        len(validation_loader.dataset),
    )
    print(
        "Training batches:",
        len(train_loader),
    )
    print(
        "Validation batches:",
        len(validation_loader),
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
        print(
            "Patches per source image:",
            patch_config.get(
                "patches_per_image",
                100,
            ),
        )
        print(
            "Vessel-centred probability:",
            patch_config.get(
                "vessel_center_probability",
                0.70,
            ),
        )
        print(
            "Minimum vessel fraction:",
            patch_config.get(
                "minimum_vessel_fraction",
                0.01,
            ),
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model_config = config.get(
        "model",
        {},
    )

    backbone_config = model_config.get(
        "backbone",
        {},
    )

    deep_supervision_config = model_config.get(
        "deep_supervision",
        {},
    )

    deep_supervision_enabled = bool(
        deep_supervision_config.get(
            "enabled",
            True,
        )
    )

    model = MVTransUNet(
        pretrained=bool(
            backbone_config.get(
                "pretrained",
                True,
            )
        ),
        transformer_channels=int(
            model_config.get(
                "transformer",
                {},
            ).get(
                "embed_dim",
                768,
            )
        ),
        vessel_reduction_ratio=int(
            model_config.get(
                "vessel_attention",
                {},
            ).get(
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
        deep_supervision=deep_supervision_enabled,
    ).to(device)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss_config = config.get(
        "loss",
        {},
    )

    auxiliary_weights = tuple(
        loss_config.get(
            "auxiliary_weights",
            [
                0.10,
                0.20,
                0.30,
            ],
        )
    )

    criterion = MVTransUNetLoss(
        alpha=float(
            loss_config.get(
                "alpha",
                0.4,
            )
        ),
        beta=float(
            loss_config.get(
                "beta",
                0.4,
            )
        ),
        gamma=float(
            loss_config.get(
                "gamma",
                0.2,
            )
        ),
        auxiliary_weights=auxiliary_weights,
    ).to(device)

    # --------------------------------------------------------
    # OPTIMIZER AND SCHEDULER
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=float(
            config[
                "optimizer"
            ][
                "learning_rate"
            ]
        ),
        weight_decay=float(
            config[
                "optimizer"
            ][
                "weight_decay"
            ]
        ),
        betas=tuple(
            config[
                "optimizer"
            ].get(
                "betas",
                [
                    0.9,
                    0.999,
                ],
            )
        ),
        eps=float(
            config[
                "optimizer"
            ].get(
                "epsilon",
                1e-8,
            )
        ),
    )

    epochs = int(
        config[
            "training"
        ][
            "epochs"
        ]
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(
            config.get(
                "scheduler",
                {},
            ).get(
                "min_lr",
                0.0,
            )
        ),
    )

    scaler = GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    accumulation_steps = int(
        config[
            "training"
        ][
            "gradient_accumulation_steps"
        ]
    )

    gradient_clipping_config = config[
        "training"
    ].get(
        "gradient_clipping",
        {},
    )

    gradient_clip_enabled = bool(
        gradient_clipping_config.get(
            "enabled",
            True,
        )
    )

    gradient_clip_max_norm = float(
        gradient_clipping_config.get(
            "max_norm",
            1.0,
        )
    )

    early_stopping_config = config[
        "training"
    ].get(
        "early_stopping",
        {},
    )

    early_stopping_enabled = bool(
        early_stopping_config.get(
            "enabled",
            True,
        )
    )

    early_stopping_patience = int(
        early_stopping_config.get(
            "patience",
            20,
        )
    )

    validation_threshold = float(
        config.get(
            "validation",
            {},
        ).get(
            "threshold",
            0.5,
        )
    )

    del validation_threshold

    # --------------------------------------------------------
    # DIRECTORIES AND LOGGING
    # --------------------------------------------------------

    checkpoint_directory = Path(
        config[
            "checkpoint"
        ][
            "directory"
        ]
    )

    logging_directory = Path(
        config[
            "logging"
        ][
            "directory"
        ]
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = SummaryWriter(
        log_dir=str(
            logging_directory
        )
    )

    print()
    print("Training configuration")
    print("-" * 70)
    print(
        "Epochs:",
        epochs,
    )
    print(
        "Batch size:",
        config[
            "dataloader"
        ][
            "batch_size"
        ],
    )
    print(
        "Gradient accumulation:",
        accumulation_steps,
    )
    print(
        "Effective batch size:",
        int(
            config[
                "dataloader"
            ][
                "batch_size"
            ]
        )
        * accumulation_steps,
    )
    print(
        "Deep supervision:",
        deep_supervision_enabled,
    )
    print(
        "Auxiliary weights:",
        auxiliary_weights,
    )
    print(
        "Checkpoint directory:",
        checkpoint_directory,
    )

    # --------------------------------------------------------
    # CHECKPOINT RESUME
    # --------------------------------------------------------

    best_dice = 0.0
    patience_counter = 0
    start_epoch = 0

    last_checkpoint_path = (
        checkpoint_directory
        / config[
            "checkpoint"
        ].get(
            "last_model_name",
            "last_model.pth",
        )
    )

    best_checkpoint_path = (
        checkpoint_directory
        / config[
            "checkpoint"
        ].get(
            "best_model_name",
            "best_model.pth",
        )
    )

    resume_enabled = bool(
        config[
            "checkpoint"
        ].get(
            "resume",
            False,
        )
    )

    if (
        resume_enabled
        and last_checkpoint_path.exists()
    ):
        print()
        print(
            "Resuming from:",
            last_checkpoint_path,
        )

        (
            start_epoch,
            best_dice,
            patience_counter,
        ) = load_checkpoint(
            path=str(
                last_checkpoint_path
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        print(
            "Next epoch:",
            start_epoch + 1,
        )
        print(
            "Best Dice:",
            best_dice,
        )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    try:
        for epoch in range(
            start_epoch,
            epochs,
        ):
            print()
            print("=" * 70)
            print(
                f"Epoch {epoch + 1}/{epochs}"
            )
            print("=" * 70)

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                accumulation_steps=accumulation_steps,
                gradient_clip_enabled=gradient_clip_enabled,
                gradient_clip_max_norm=gradient_clip_max_norm,
            )

            validation_metrics = validate(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
            )

            scheduler.step()

            learning_rate = float(
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            )

            print()
            print("Epoch results")
            print("-" * 70)
            print(
                "Train total loss:",
                f"{train_metrics['total_loss']:.6f}",
            )
            print(
                "Train main loss:",
                f"{train_metrics['main_loss']:.6f}",
            )
            print(
                "Train auxiliary loss:",
                f"{train_metrics['auxiliary_loss']:.6f}",
            )
            print(
                "Train Dice:",
                f"{train_metrics['dice']:.6f}",
            )
            print(
                "Validation loss:",
                f"{validation_metrics['total_loss']:.6f}",
            )
            print(
                "Validation Dice:",
                f"{validation_metrics['dice']:.6f}",
            )
            print(
                "Learning rate:",
                f"{learning_rate:.10f}",
            )

            # ------------------------------------------------
            # TENSORBOARD LOGGING
            # ------------------------------------------------

            writer.add_scalar(
                "Loss/train_total",
                train_metrics[
                    "total_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Loss/train_main",
                train_metrics[
                    "main_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Loss/train_auxiliary",
                train_metrics[
                    "auxiliary_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Loss/train_bce",
                train_metrics[
                    "bce_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Loss/train_dice",
                train_metrics[
                    "dice_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Loss/train_boundary",
                train_metrics[
                    "boundary_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Dice/train",
                train_metrics[
                    "dice"
                ],
                epoch,
            )

            writer.add_scalar(
                "Loss/validation_total",
                validation_metrics[
                    "total_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Loss/validation_bce",
                validation_metrics[
                    "bce_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Loss/validation_dice",
                validation_metrics[
                    "dice_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Loss/validation_boundary",
                validation_metrics[
                    "boundary_loss"
                ],
                epoch,
            )
            writer.add_scalar(
                "Dice/validation",
                validation_metrics[
                    "dice"
                ],
                epoch,
            )
            writer.add_scalar(
                "LearningRate",
                learning_rate,
                epoch,
            )

            # ------------------------------------------------
            # BEST MODEL TRACKING
            # ------------------------------------------------

            current_validation_dice = (
                validation_metrics[
                    "dice"
                ]
            )

            improved = (
                current_validation_dice
                > best_dice
            )

            if improved:
                best_dice = (
                    current_validation_dice
                )
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_dice=best_dice,
                patience_counter=patience_counter,
                path=str(
                    last_checkpoint_path
                ),
            )

            if improved:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_dice=best_dice,
                    patience_counter=patience_counter,
                    path=str(
                        best_checkpoint_path
                    ),
                )

                print(
                    "Saved new best model."
                )

            writer.flush()

            if (
                early_stopping_enabled
                and patience_counter
                >= early_stopping_patience
            ):
                print()
                print(
                    "Early stopping triggered."
                )
                break

    finally:
        writer.close()

    print()
    print("=" * 70)
    print("Training complete")
    print("=" * 70)
    print(
        "Best validation Dice:",
        f"{best_dice:.6f}",
    )


if __name__ == "__main__":
    main()