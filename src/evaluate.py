"""
MV-TransUNet Evaluation Pipeline

Retinal Vessel Segmentation Framework

Metrics:
    - Dice
    - Sensitivity
    - Specificity
    - Accuracy
    - ROC-AUC

Additional:
    - Thin vessel Dice
    - Skeletonization analysis
    - Cross dataset evaluation

Supported:
    - DRIVE (train/test kept strictly separate -- see prepare_datasets.py)
    - STARE
    - CHASE_DB1
    - HRF

This script reports THREE distinct numbers, and they are not
interchangeable:

1. "DRIVE_internal_validation" -- the same image-level validation
   split used during training (dataset.validation_ratio /
   dataset.split_seed, drawn from dataset.train_dataset). This is a
   cross-check against the "Validation Dice" your training log
   reported; it is NOT a fair literature comparison, because it was
   used for early-stopping checkpoint selection.

2. "DRIVE_test" -- dataset.test_dataset, evaluated only here, never
   touched during training. THIS is the number comparable to published
   DRIVE baselines. Requires prepare_datasets.py to have been re-run
   with DRIVE_train/DRIVE_test kept separate, and config.yaml's
   dataset.test_dataset to point at the real held-out folder.

3. Cross-dataset results (STARE / CHASE_DB1 / HRF) -- generalization
   check, trained on DRIVE only, never seen these datasets at all.
"""


import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from skimage.morphology import skeletonize

from torch.amp import autocast

from models.mv_transunet import MVTransUNet

from src.datasets import (
    build_cross_dataset_loaders,
    build_test_loader,
    create_split_indices,
    get_validation_transform,
    load_sample_pairs,
    GridPatchRetinalDataset,
    RetinalVesselDataset,
)
from src.datasets import create_dataloader


# ============================================================
# CONFIGURATION
# ============================================================

def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    TP = np.logical_and(prediction, target).sum()
    TN = np.logical_and(np.logical_not(prediction), np.logical_not(target)).sum()
    FP = np.logical_and(prediction, np.logical_not(target)).sum()
    FN = np.logical_and(np.logical_not(prediction), target).sum()

    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return {
        "dice": dice,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
    }


# ============================================================
# THIN VESSEL ANALYSIS
# ============================================================

def extract_thin_vessels(mask, threshold=3):
    mask = mask.astype(np.uint8)

    skeleton = skeletonize(mask)

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    vessel_width = distance * 2

    thin_region = vessel_width <= threshold

    thin_vessels = np.logical_and(skeleton, thin_region)

    return thin_vessels


def calculate_thin_vessel_dice(prediction, target):
    target_thin = extract_thin_vessels(target)
    prediction_thin = np.logical_and(prediction, target_thin)
    metric = calculate_metrics(prediction_thin, target_thin)
    return metric["dice"]


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(checkpoint_path, device, config):
    """
    Build MVTransUNet with the EXACT same architecture arguments
    train.py used, read from config.yaml -- not hardcoded defaults.

    A mismatch here (e.g. a different vessel_reduction_ratio or
    deep_supervision flag than what was actually trained) will raise
    a loud state_dict shape-mismatch error rather than silently doing
    the wrong thing, but there is no reason to rely on that safety
    net when the config is right there.
    """

    model_config = config.get("model", {})
    backbone_config = model_config.get("backbone", {})
    deep_supervision_config = model_config.get("deep_supervision", {})

    model = MVTransUNet(
        pretrained=False,  # irrelevant at eval time -- weights are overwritten below
        transformer_channels=int(
            model_config.get("transformer", {}).get("embed_dim", 768)
        ),
        vessel_reduction_ratio=int(
            model_config.get("vessel_attention", {}).get("reduction_ratio", 16)
        ),
        output_channels=int(model_config.get("output_channels", 1)),
        deep_supervision=bool(deep_supervision_config.get("enabled", True)),
        # nn.Dropout2d has no learnable parameters, so this value has
        # zero effect on state_dict loading either way -- matched to
        # train.py purely for architecture-metadata consistency, and
        # because model.eval() (called below) disables Dropout2d
        # regardless of this value.
        decoder_dropout_rate=float(
            model_config.get("decoder", {}).get("dropout_rate", 0.1)
        ),
    )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}\n"
            "Check config.yaml's checkpoint.directory / "
            "checkpoint.best_model_name match where train.py actually "
            "saved it."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        # Matches train.py's load_checkpoint. Newer PyTorch defaults
        # torch.load to weights_only=True, which fails on a full
        # training-state checkpoint dict (optimizer/scheduler state,
        # plain Python ints/floats) -- not just a bare state_dict.
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state"])

    model.to(device)
    model.eval()

    print(
        "Loaded checkpoint:",
        checkpoint_path,
    )
    print(
        "  epoch:", checkpoint.get("epoch", "unknown"),
        "| best_dice recorded at save time:", checkpoint.get("best_dice", "unknown"),
    )

    return model


# ============================================================
# INTERNAL VALIDATION SPLIT RECONSTRUCTION
# ============================================================

def _reconstruct_validation_pairs(config):
    """
    Reconstruct the EXACT image-level validation split train.py used
    -- same source directory, same validation_ratio, same split_seed,
    same create_split_indices function actually imported from your
    real datasets.py (not a reimplementation). Both loader builders
    below share this so they operate on identical validation images,
    differing only in whole-image vs. grid-patch evaluation.
    """

    train_dataset_config = config["dataset"]["train_dataset"]

    pairs = load_sample_pairs(
        train_dataset_config["image_dir"],
        train_dataset_config["mask_dir"],
    )

    _, validation_indices = create_split_indices(
        dataset_size=len(pairs),
        validation_ratio=float(config["dataset"].get("validation_ratio", 0.2)),
        seed=int(config["dataset"].get("split_seed", config["seed"]["value"])),
    )

    validation_pairs = [pairs[i] for i in validation_indices]

    print(
        f"Reconstructed internal validation split: "
        f"{len(validation_pairs)} image(s) "
        f"(out of {len(pairs)} in dataset.train_dataset)."
    )

    return validation_pairs


def build_internal_validation_loader_whole_image(config, image_size, batch_size=1):
    """
    Whole-image evaluation of the reconstructed validation split.

    This is the literature-style metric (one Dice per whole image,
    then averaged) -- NOT what your training log's "Validation Dice"
    reports, since train.py validates on deterministic overlapping
    grid patches (patch_training.validation_stride), not whole
    images. Use build_internal_validation_loader_grid_patches for a
    number directly comparable to the training log instead. Neither
    of these is the headline result -- see DRIVE_test for that.
    """

    validation_pairs = _reconstruct_validation_pairs(config)

    dataset = RetinalVesselDataset(
        image_dir=config["dataset"]["train_dataset"]["image_dir"],
        mask_dir=config["dataset"]["train_dataset"]["mask_dir"],
        samples=validation_pairs,
        transform=get_validation_transform(image_size=image_size),
        clahe=bool(config["preprocessing"]["clahe"].get("enabled", True)),
    )

    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["dataloader"].get("num_workers", 2)),
        pin_memory=bool(config["dataloader"].get("pin_memory", True)),
        persistent_workers=False,
        drop_last=False,
        seed=int(config["dataset"].get("split_seed", config["seed"]["value"])),
    )


def build_internal_validation_loader_grid_patches(config, batch_size=1):
    """
    Grid-patch evaluation of the reconstructed validation split,
    using the SAME GridPatchRetinalDataset class and the SAME
    patch_training.patch_size / model_input_size / validation_stride
    values train.py used. This is the number that should match your
    training log's per-epoch "Validation Dice" (up to CLAHE/transform
    determinism), because it is the same methodology, not an
    approximation of it.
    """

    patch_config = config.get("patch_training", {})

    if not bool(patch_config.get("enabled", False)):
        raise ValueError(
            "patch_training.enabled is false -- grid-patch validation "
            "reconstruction only applies to patch-based training runs. "
            "Use build_internal_validation_loader_whole_image instead."
        )

    validation_pairs = _reconstruct_validation_pairs(config)

    model_input_size = int(
        patch_config.get("model_input_size", 256)
    )
    patch_size = int(
        patch_config.get("patch_size", model_input_size)
    )
    validation_stride = int(
        patch_config.get("validation_stride", max(1, patch_size // 2))
    )

    dataset = GridPatchRetinalDataset(
        samples=validation_pairs,
        patch_size=patch_size,
        model_input_size=model_input_size,
        stride=validation_stride,
        transform=get_validation_transform(image_size=model_input_size),
        clahe=bool(config["preprocessing"]["clahe"].get("enabled", True)),
        include_empty_patches=bool(
            patch_config.get("include_empty_validation_patches", True)
        ),
        minimum_vessel_fraction=0.0,
    )

    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["dataloader"].get("num_workers", 2)),
        pin_memory=bool(config["dataloader"].get("pin_memory", True)),
        persistent_workers=False,
        drop_last=False,
        seed=int(config["dataset"].get("split_seed", config["seed"]["value"])),
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def predict_with_tta(model, images, device):
    """
    Flip-based test-time augmentation.

    Averages sigmoid probabilities over the identity image and its
    horizontal, vertical, and horizontal+vertical flips, each
    un-flipped back to the original orientation before averaging.
    Free at inference (no retraining, no architecture change) and
    well precedented for exactly this kind of small remaining-gap
    problem. Expect roughly +1-2 Dice points; treat as a cheap,
    low-risk lever to try before anything requiring retraining.
    """

    flip_configs = [
        (None,),
        (2,),       # vertical flip  (height axis)
        (3,),       # horizontal flip (width axis)
        (2, 3),     # both
    ]

    accumulated_probability = None

    with autocast(device_type=device.type, enabled=device.type == "cuda"):
        for dims in flip_configs:
            if dims == (None,):
                flipped_input = images
            else:
                flipped_input = torch.flip(images, dims=dims)

            output = model(flipped_input)

            probability = torch.sigmoid(output)

            if dims != (None,):
                probability = torch.flip(probability, dims=dims)

            probability = probability.float()

            accumulated_probability = (
                probability
                if accumulated_probability is None
                else accumulated_probability + probability
            )

    return accumulated_probability / len(flip_configs)


def sweep_threshold(model, loader, device, thresholds=None, use_tta=False):
    """
    Sweep binarization thresholds on a validation loader and return
    the Dice-maximizing threshold.

    IMPORTANT: only ever call this on a validation split, never on
    DRIVE_test. The returned threshold should then be applied as a
    FIXED value when evaluating DRIVE_test -- tuning the threshold
    directly on test data would invalidate the comparison to
    published baselines just as surely as training on it would.
    """

    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.05)

    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Threshold sweep (forward pass)"):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].numpy()

            if use_tta:
                probability = predict_with_tta(model, images, device)
            else:
                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    probability = torch.sigmoid(model(images))

            all_probabilities.append(probability.cpu().numpy())
            all_targets.append(targets)

    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    best_threshold = 0.5
    best_dice = -1.0

    print("\nThreshold sweep (validation split only):")

    for threshold in thresholds:
        dice_values = []

        for probability, target in zip(all_probabilities, all_targets):
            prediction = (probability.squeeze() > threshold)
            dice_values.append(
                calculate_metrics(prediction, target.squeeze())["dice"]
            )

        mean_dice = float(np.mean(dice_values))

        print(f"  threshold={threshold:.2f} -> Dice={mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = float(threshold)

    print(f"Selected threshold: {best_threshold:.2f} (validation Dice={best_dice:.4f})")

    return best_threshold


def evaluate_dataset(model, loader, device, threshold=0.5, save_directory=None, use_tta=False):
    results = []
    all_probabilities = []
    all_targets = []

    if save_directory:
        os.makedirs(save_directory, exist_ok=True)

    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc="Evaluating")):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].numpy()

            with autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                if use_tta:
                    probabilities = predict_with_tta(model, images, device)
                else:
                    outputs = model(images)
                    probabilities = torch.sigmoid(outputs)

            probabilities = probabilities.cpu().numpy()

            predictions = probabilities > threshold

            for sample_id, (pred, target, probability) in enumerate(
                zip(predictions, masks, probabilities)
            ):
                pred = pred.squeeze()
                target = target.squeeze()
                probability = probability.squeeze()

                metrics = calculate_metrics(pred, target)

                metrics["thin_dice"] = calculate_thin_vessel_dice(pred, target)

                results.append(metrics)

                all_probabilities.extend(probability.flatten())
                all_targets.extend(target.flatten())

                if save_directory:
                    output_path = os.path.join(
                        save_directory,
                        f"prediction_{index}_{sample_id}.png",
                    )

                    cv2.imwrite(
                        output_path,
                        (pred.astype(np.uint8) * 255),
                    )

    if len(results) == 0:
        raise RuntimeError("evaluate_dataset received an empty loader.")

    final_metrics = {}

    for key in results[0]:
        final_metrics[key] = float(
            np.mean([item[key] for item in results])
        )

    try:
        final_metrics["auc"] = float(
            roc_auc_score(all_targets, all_probabilities)
        )
    except ValueError:
        final_metrics["auc"] = 0.0

    return final_metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        yaml.dump(results, file, sort_keys=False)


# ============================================================
# MAIN
# ============================================================

def main():
    config = load_config("config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Evaluation device:", device)

    # --------------------------------------------------------
    # RESOLUTION -- must match what training actually validated at
    # --------------------------------------------------------
    #
    # preprocessing.image_size is only correct when patch_training is
    # disabled. With patch training enabled (as in the current
    # config), the resolution actually used for whole-image
    # validation during training is patch_training.model_input_size.
    # Evaluating at the wrong resolution silently produces a
    # different, non-comparable number.

    patch_config = config.get("patch_training", {})

    if bool(patch_config.get("enabled", False)):
        image_size = int(
            patch_config.get(
                "model_input_size",
                config["preprocessing"]["image_size"]["height"],
            )
        )
        print(f"patch_training.enabled=true -> evaluating at model_input_size={image_size}")
    else:
        image_size = int(config["preprocessing"]["image_size"]["height"])
        print(f"patch_training.enabled=false -> evaluating at preprocessing.image_size={image_size}")

    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))

    # --------------------------------------------------------
    # CHECKPOINT -- read from config, not a hardcoded relative path
    # --------------------------------------------------------

    checkpoint_directory = Path(config["checkpoint"]["directory"])

    checkpoint_path = checkpoint_directory / config["checkpoint"].get(
        "best_model_name", "best_model.pth"
    )

    model = load_model(checkpoint_path, device, config)

    experiment_output_dir = Path(
        config.get("experiment", {}).get("output_directory", "./experiments")
    )

    all_results = {}

    # --------------------------------------------------------
    # 1. INTERNAL VALIDATION SPLIT (two methodologies, both cross-checks)
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Evaluating DRIVE_internal_validation_grid_patches")
    print("(SAME methodology as your training log's 'Validation Dice':")
    print(" deterministic overlapping grid patches, not whole images.")
    print(" Use this to confirm evaluate.py reproduces train.py's number.")
    print(" NOT a fair literature comparison -- used for checkpoint selection.)")
    print("=" * 60)

    if bool(patch_config.get("enabled", False)):
        grid_validation_loader = build_internal_validation_loader_grid_patches(
            config=config,
        )

        all_results["DRIVE_internal_validation_grid_patches"] = evaluate_dataset(
            model,
            grid_validation_loader,
            device,
            threshold=threshold,
            save_directory=None,
        )
    else:
        print("patch_training.enabled=false -- skipping, no grid patches were used in training.")

    print("\n" + "=" * 60)
    print("Evaluating DRIVE_internal_validation (whole-image)")
    print("(literature-style metric: one Dice per whole image, then averaged.")
    print(" Still the SAME potentially-leaky split as above until")
    print(" DRIVE_train/DRIVE_test separation is confirmed -- see warning below.")
    print(" NOT a fair literature comparison, cross-check only)")
    print("=" * 60)

    internal_validation_loader = build_internal_validation_loader_whole_image(
        config=config,
        image_size=image_size,
    )

    all_results["DRIVE_internal_validation"] = evaluate_dataset(
        model,
        internal_validation_loader,
        device,
        threshold=threshold,
        save_directory=str(
            experiment_output_dir / "predictions" / "DRIVE_internal_validation"
        ),
    )

    all_results["DRIVE_internal_validation_TTA"] = evaluate_dataset(
        model,
        internal_validation_loader,
        device,
        threshold=threshold,
        save_directory=None,
        use_tta=True,
    )

    # Tune the binarization threshold on validation ONLY, using TTA
    # predictions (since that is what will actually be used on
    # DRIVE_test below). Never call sweep_threshold on DRIVE_test.
    tuned_threshold = sweep_threshold(
        model,
        internal_validation_loader,
        device,
        use_tta=True,
    )

    # --------------------------------------------------------
    # 2. TRUE HELD-OUT DRIVE TEST SET (the number that matters)
    # --------------------------------------------------------

    test_dataset_config = config["dataset"].get("test_dataset")

    if test_dataset_config is None:
        print(
            "\nWARNING: config.yaml has no dataset.test_dataset entry. "
            "Skipping the official DRIVE test-set evaluation -- this "
            "means you currently have NO number comparable to "
            "published DRIVE baselines. Re-run the corrected "
            "prepare_datasets.py (DRIVE_train/DRIVE_test kept "
            "separate) and add dataset.test_dataset to config.yaml."
        )
    else:
        print("\n" + "=" * 60)
        print("Evaluating DRIVE_test (official held-out set)")
        print("=" * 60)

        test_loader = build_test_loader(
            image_dir=test_dataset_config["image_dir"],
            mask_dir=test_dataset_config["mask_dir"],
            image_size=image_size,
            clahe=bool(config["preprocessing"]["clahe"].get("enabled", True)),
        )

        all_results["DRIVE_test"] = evaluate_dataset(
            model,
            test_loader,
            device,
            threshold=threshold,
            save_directory=str(experiment_output_dir / "predictions" / "DRIVE_test"),
        )

        all_results["DRIVE_test_TTA"] = evaluate_dataset(
            model,
            test_loader,
            device,
            threshold=threshold,
            save_directory=None,
            use_tta=True,
        )

        all_results["DRIVE_test_TTA_tuned_threshold"] = evaluate_dataset(
            model,
            test_loader,
            device,
            threshold=tuned_threshold,
            save_directory=str(
                experiment_output_dir / "predictions" / "DRIVE_test_TTA_tuned_threshold"
            ),
            use_tta=True,
        )

    # --------------------------------------------------------
    # 3. CROSS-DATASET GENERALIZATION
    # --------------------------------------------------------

    external_datasets = config["dataset"].get("external_datasets", {})

    if external_datasets:
        cross_dataset_loaders = build_cross_dataset_loaders(
            datasets_config=external_datasets,
            image_size=image_size,
            clahe=bool(config["preprocessing"]["clahe"].get("enabled", True)),
        )

        for name, loader in cross_dataset_loaders.items():
            print("\n" + "=" * 60)
            print(f"Evaluating {name} (cross-dataset generalization)")
            print("=" * 60)

            all_results[name] = evaluate_dataset(
                model,
                loader,
                device,
                threshold=threshold,
                save_directory=str(experiment_output_dir / "predictions" / name),
            )
@torch.no_grad()
def evaluate_reconstructed_grid_patches(
    model,
    config,
    device,
    threshold=0.5,
):
    """
    Whole-image evaluation using deterministic grid patches.

    This reconstructs full retinal predictions from the same
    GridPatchRetinalDataset used during validation.

    This is different from:
    - patch Dice validation
    - direct whole-image resizing

    It evaluates the model in the distribution it was trained on.
    """

    patch_config = config.get(
        "patch_training",
        {}
    )

    if not bool(
        patch_config.get(
            "enabled",
            False
        )
    ):
        raise ValueError(
            "Patch training must be enabled "
            "for reconstructed evaluation."
        )

    validation_pairs = _reconstruct_validation_pairs(
        config
    )

    patch_size = int(
        patch_config.get(
            "patch_size",
            256
        )
    )

    model_input_size = int(
        patch_config.get(
            "model_input_size",
            256
        )
    )

    stride = int(
        patch_config.get(
            "validation_stride",
            patch_size // 2
        )
    )


    dataset = GridPatchRetinalDataset(
        samples=validation_pairs,
        patch_size=patch_size,
        model_input_size=model_input_size,
        stride=stride,
        transform=get_validation_transform(
            image_size=model_input_size
        ),
        clahe=bool(
            config["preprocessing"]["clahe"].get(
                "enabled",
                True
            )
        ),
        include_empty_patches=True,
        minimum_vessel_fraction=0.0,
    )


    loader = create_dataloader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        seed=int(
            config["dataset"].get(
                "split_seed",
                config["seed"]["value"]
            )
        ),
    )


    model.eval()


    reconstructed = {}
    targets = {}


    for batch in tqdm(
        loader,
        desc="Reconstructing validation images",
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        outputs = torch.sigmoid(
            model(images)
        )


        output = outputs.squeeze().cpu().numpy()


        image_index = int(
            batch["source_image_index"].item()
        )

        top = int(
            batch["top"].item()
        )

        left = int(
            batch["left"].item()
        )


        if image_index not in reconstructed:

            height = (
                validation_pairs[image_index][1]
            )

            mask = load_binary_mask(
                height
            )

            h, w = mask.shape


            reconstructed[image_index] = {
                "prediction":
                    np.zeros(
                        (h,w),
                        dtype=np.float32
                    ),

                "count":
                    np.zeros(
                        (h,w),
                        dtype=np.float32
                    ),
            }

            targets[image_index] = mask


        prediction = reconstructed[image_index][
            "prediction"
        ]

        count = reconstructed[image_index][
            "count"
        ]


        ph, pw = output.shape


        prediction[
            top:top+ph,
            left:left+pw
        ] += output


        count[
            top:top+ph,
            left:left+pw
        ] += 1



    results = []


    for image_index in reconstructed:

        prediction = (
            reconstructed[image_index]["prediction"]
            /
            np.maximum(
                reconstructed[image_index]["count"],
                1e-8
            )
        )


        binary_prediction = (
            prediction > threshold
        )


        metric = calculate_metrics(
            binary_prediction,
            targets[image_index]
        )

        results.append(metric)


    mean_dice = np.mean(
        [
            item["dice"]
            for item in results
        ]
    )


    print("="*60)
    print(
        "DRIVE reconstructed validation"
    )
    print("="*60)

    print(
        f"Images evaluated: {len(results)}"
    )

    print(
        f"Dice: {mean_dice:.4f}"
    )


    return {
        "dice": mean_dice,
        "image_results": results,
    }

    # --------------------------------------------------------
    # SAVE + REPORT
    # --------------------------------------------------------

    results_path = experiment_output_dir / "evaluation_results.yaml"

    save_results(all_results, str(results_path))

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)

    for dataset_name, metrics in all_results.items():
        print("\n", dataset_name)

        for key, value in metrics.items():
            print(key, ":", round(value, 4))

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()