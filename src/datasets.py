"""
MV-TransUNet Dataset Pipeline

Retinal Vessel Segmentation Framework

Supported datasets:
- DRIVE
- STARE
- CHASE_DB1
- HRF

Features:
- Automatic image-mask matching
- Deterministic train-validation splitting
- Separate training and validation transforms
- CLAHE enhancement
- Albumentations augmentation
- PyTorch Dataset and DataLoader
- Colab GPU optimization
- Reproducible DataLoader workers
- Cross-dataset evaluation support
"""


import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".ppm",
    ".gif",
}


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set global random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker deterministically.
    """

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# CLAHE ENHANCEMENT
# ============================================================

class CLAHEEnhancement:
    """
    Apply CLAHE to the luminance channel of an RGB image.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: int = 8,
    ) -> None:
        self.clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(
                int(tile_grid_size),
                int(tile_grid_size),
            ),
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError(
                "CLAHE received an empty image."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "CLAHE expects an RGB image with three channels."
            )

        lab_image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2LAB,
        )

        l_channel, a_channel, b_channel = cv2.split(
            lab_image
        )

        enhanced_l_channel = self.clahe.apply(
            l_channel
        )

        enhanced_lab = cv2.merge(
            [
                enhanced_l_channel,
                a_channel,
                b_channel,
            ]
        )

        enhanced_rgb = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2RGB,
        )

        return enhanced_rgb


# ============================================================
# AUGMENTATION PIPELINES
# ============================================================

def get_train_transform(
    image_size: int = 256,
) -> A.Compose:
    """
    Training augmentation pipeline.
    """

    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
            ),

            A.HorizontalFlip(
                p=0.5,
            ),

            A.VerticalFlip(
                p=0.5,
            ),

            A.Rotate(
                limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),

            A.ElasticTransform(
                alpha=120,
                sigma=6,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.3,
            ),

            A.RandomBrightnessContrast(
                p=0.5,
            ),

            A.Normalize(
                mean=(
                    0.485,
                    0.456,
                    0.406,
                ),
                std=(
                    0.229,
                    0.224,
                    0.225,
                ),
            ),

            ToTensorV2(),
        ]
    )


def get_validation_transform(
    image_size: int = 256,
) -> A.Compose:
    """
    Validation and test preprocessing pipeline.
    """

    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
            ),

            A.Normalize(
                mean=(
                    0.485,
                    0.456,
                    0.406,
                ),
                std=(
                    0.229,
                    0.224,
                    0.225,
                ),
            ),

            ToTensorV2(),
        ]
    )


# ============================================================
# FILE DISCOVERY
# ============================================================

def get_files(
    directory: Union[str, Path],
) -> List[Path]:
    """
    Return all supported image files in a directory.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory.resolve()}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory.resolve()}"
        )

    files = [
        file
        for file in directory.iterdir()
        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]

    return sorted(
        files,
        key=lambda path: path.name.lower(),
    )


# ============================================================
# FILE-NAME NORMALIZATION
# ============================================================

def normalize_filename(
    filename: Union[str, Path],
) -> str:
    """
    Normalize dataset-specific image and mask names.
    """

    name = Path(filename).stem.lower()

    replacements = [
        "_training_mask",
        "_test_mask",
        "_training",
        "_test",
        "_manual1",
        "_manual",
        "_mask",
        "_1stho",
        "_2ndho",
        ".ah",
        ".vk",
    ]

    for item in replacements:
        name = name.replace(
            item,
            "",
        )

    return name.strip()


# ============================================================
# IMAGE-MASK MATCHING
# ============================================================

def match_image_masks(
    image_files: Sequence[Path],
    mask_files: Sequence[Path],
) -> List[Tuple[Path, Path]]:
    """
    Match images and masks using normalized filenames.
    """

    if len(image_files) == 0:
        raise RuntimeError(
            "No image files were found."
        )

    if len(mask_files) == 0:
        raise RuntimeError(
            "No mask files were found."
        )

    mask_dictionary: Dict[str, Path] = {}

    for mask_path in mask_files:
        key = normalize_filename(
            mask_path.name
        )

        # Keep the first matching annotation deterministically.
        if key not in mask_dictionary:
            mask_dictionary[key] = mask_path

    pairs: List[Tuple[Path, Path]] = []
    unmatched_images: List[str] = []

    for image_path in image_files:
        key = normalize_filename(
            image_path.name
        )

        mask_path = mask_dictionary.get(
            key
        )

        if mask_path is not None:
            pairs.append(
                (
                    image_path,
                    mask_path,
                )
            )
        else:
            unmatched_images.append(
                image_path.name
            )

    if len(pairs) == 0:
        raise RuntimeError(
            "No matching image-mask pairs were found."
        )

    if unmatched_images:
        print(
            f"Warning: {len(unmatched_images)} images "
            "did not have matching masks."
        )

    return pairs


# ============================================================
# RETINAL VESSEL DATASET
# ============================================================

class RetinalVesselDataset(Dataset):
    """
    PyTorch dataset for retinal vessel segmentation.
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        transform: Optional[A.Compose] = None,
        clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ) -> None:
        super().__init__()

        self.image_dir = Path(
            image_dir
        )

        self.mask_dir = Path(
            mask_dir
        )

        self.transform = transform

        image_files = get_files(
            self.image_dir
        )

        mask_files = get_files(
            self.mask_dir
        )

        self.samples = match_image_masks(
            image_files,
            mask_files,
        )

        self.clahe = None

        if clahe:
            self.clahe = CLAHEEnhancement(
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )

    def __len__(self) -> int:
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Union[torch.Tensor, str]]:
        image_path, mask_path = self.samples[index]

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        if self.clahe is not None:
            image = self.clahe(
                image
            )

        mask = (
            mask > 127
        ).astype(
            np.float32
        )

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image_tensor = transformed[
                "image"
            ]

            mask_tensor = transformed[
                "mask"
            ]

        else:
            image_tensor = torch.from_numpy(
                image.transpose(
                    2,
                    0,
                    1,
                )
            ).float()

            # Scale unnormalized images to [0, 1].
            image_tensor = (
                image_tensor / 255.0
            )

            mask_tensor = torch.from_numpy(
                mask
            ).float()

        if not torch.is_tensor(
            image_tensor
        ):
            image_tensor = torch.as_tensor(
                image_tensor
            )

        if not torch.is_tensor(
            mask_tensor
        ):
            mask_tensor = torch.as_tensor(
                mask_tensor
            )

        image_tensor = image_tensor.float()
        mask_tensor = mask_tensor.float()

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(
                0
            )

        elif (
            mask_tensor.ndim == 3
            and mask_tensor.shape[-1] == 1
        ):
            mask_tensor = mask_tensor.permute(
                2,
                0,
                1,
            )

        mask_tensor = (
            mask_tensor > 0.5
        ).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(
                image_path
            ),
            "mask_path": str(
                mask_path
            ),
        }


# ============================================================
# DETERMINISTIC SPLIT INDICES
# ============================================================

def create_split_indices(
    dataset_size: int,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Create deterministic training and validation indices.
    """

    if dataset_size < 2:
        raise ValueError(
            "At least two samples are required for "
            "a train-validation split."
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    validation_size = int(
        dataset_size * validation_ratio
    )

    validation_size = max(
        1,
        validation_size,
    )

    validation_size = min(
        validation_size,
        dataset_size - 1,
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    shuffled_indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    validation_indices = shuffled_indices[
        :validation_size
    ]

    training_indices = shuffled_indices[
        validation_size:
    ]

    return (
        training_indices,
        validation_indices,
    )


# ============================================================
# DATASET SPLITTING
# ============================================================

def create_train_validation_split(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Split two separate dataset instances using identical indices.

    Separate dataset instances are necessary because training and
    validation use different transforms.
    """

    if len(train_dataset) != len(
        validation_dataset
    ):
        raise ValueError(
            "Training and validation dataset instances "
            "must contain the same samples."
        )

    training_indices, validation_indices = (
        create_split_indices(
            dataset_size=len(
                train_dataset
            ),
            validation_ratio=validation_ratio,
            seed=seed,
        )
    )

    training_subset = Subset(
        train_dataset,
        training_indices,
    )

    validation_subset = Subset(
        validation_dataset,
        validation_indices,
    )

    return (
        training_subset,
        validation_subset,
    )


# ============================================================
# DATALOADER CREATION
# ============================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    """
    Create a reproducible PyTorch DataLoader.
    """

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    use_persistent_workers = (
        persistent_workers
        and num_workers > 0
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(
            pin_memory
            and torch.cuda.is_available()
        ),
        persistent_workers=use_persistent_workers,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


# ============================================================
# TRAIN AND VALIDATION PIPELINE
# ============================================================

def build_dataloaders(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    image_size: int = 256,
    batch_size: int = 8,
    validation_ratio: float = 0.2,
    num_workers: int = 2,
    clahe: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build deterministic DRIVE training and validation loaders.

    The two subsets use separate dataset objects so that:
    - training keeps augmentation;
    - validation uses deterministic preprocessing only.
    """

    training_dataset_full = RetinalVesselDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_train_transform(
            image_size=image_size
        ),
        clahe=clahe,
    )

    validation_dataset_full = RetinalVesselDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_validation_transform(
            image_size=image_size
        ),
        clahe=clahe,
    )

    train_dataset, validation_dataset = (
        create_train_validation_split(
            train_dataset=training_dataset_full,
            validation_dataset=validation_dataset_full,
            validation_ratio=validation_ratio,
            seed=seed,
        )
    )

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        seed=seed,
    )

    validation_loader = create_dataloader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
        seed=seed,
    )

    return (
        train_loader,
        validation_loader,
    )


# ============================================================
# TEST / EVALUATION LOADER
# ============================================================

def build_test_loader(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 2,
    clahe: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a deterministic test DataLoader.
    """

    dataset = RetinalVesselDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_validation_transform(
            image_size=image_size
        ),
        clahe=clahe,
    )

    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False,
        seed=seed,
    )


# ============================================================
# CROSS-DATASET LOADERS
# ============================================================

def build_cross_dataset_loaders(
    datasets_config: Union[
        Sequence[Dict],
        Dict[str, Dict],
    ],
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 2,
    clahe: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Build test loaders for STARE, CHASE_DB1, and HRF.

    Supports either:
    1. A list of dictionaries with a `name` field.
    2. A dictionary keyed by dataset name.
    """

    loaders: Dict[str, DataLoader] = {}

    if isinstance(
        datasets_config,
        dict,
    ):
        dataset_entries = [
            {
                "name": dataset_name,
                **dataset_config,
            }
            for dataset_name, dataset_config
            in datasets_config.items()
        ]

    else:
        dataset_entries = list(
            datasets_config
        )

    for dataset_config in dataset_entries:
        name = dataset_config[
            "name"
        ]

        print(
            f"Loading evaluation dataset: {name}"
        )

        loaders[name] = build_test_loader(
            image_dir=dataset_config[
                "image_dir"
            ],
            mask_dir=dataset_config[
                "mask_dir"
            ],
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            clahe=clahe,
            seed=seed,
        )

    return loaders


# ============================================================
# DATASET STATISTICS
# ============================================================

def dataset_statistics(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
) -> Dict[str, int]:
    """
    Print and return basic dataset statistics.
    """

    dataset = RetinalVesselDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=None,
        clahe=False,
    )

    statistics = {
        "samples": len(
            dataset
        ),
        "images": len(
            get_files(image_dir)
        ),
        "masks": len(
            get_files(mask_dir)
        ),
    }

    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(
        "Matched samples:",
        statistics["samples"],
    )
    print(
        "Image files:",
        statistics["images"],
    )
    print(
        "Mask files:",
        statistics["masks"],
    )
    print(
        "Image directory:",
        Path(image_dir).resolve(),
    )
    print(
        "Mask directory:",
        Path(mask_dir).resolve(),
    )
    print("=" * 60)

    return statistics


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_dataset(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    image_size: int = 256,
) -> bool:
    """
    Verify that one dataset sample loads correctly.
    """

    print("\nChecking dataset...")

    try:
        dataset = RetinalVesselDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            transform=get_validation_transform(
                image_size=image_size
            ),
            clahe=True,
        )

        sample = dataset[0]

        print("Dataset OK")
        print(
            "Number of samples:",
            len(dataset),
        )
        print(
            "Image shape:",
            sample["image"].shape,
        )
        print(
            "Mask shape:",
            sample["mask"].shape,
        )
        print(
            "Image dtype:",
            sample["image"].dtype,
        )
        print(
            "Mask dtype:",
            sample["mask"].dtype,
        )
        print(
            "Mask values:",
            torch.unique(
                sample["mask"]
            ).tolist(),
        )

        return True

    except Exception as error:
        print("Dataset Error:")
        print(error)

        return False


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print(
        "MV-TransUNet Dataset Pipeline Test"
    )

    test_image_dir = Path(
        "./datasets_processed/DRIVE/images"
    )

    test_mask_dir = Path(
        "./datasets_processed/DRIVE/masks"
    )

    if (
        test_image_dir.exists()
        and test_mask_dir.exists()
    ):
        verify_dataset(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            image_size=256,
        )

        train_loader, validation_loader = (
            build_dataloaders(
                image_dir=test_image_dir,
                mask_dir=test_mask_dir,
                image_size=256,
                batch_size=8,
                validation_ratio=0.2,
                num_workers=2,
                clahe=True,
                seed=42,
            )
        )

        print()
        print("=" * 60)
        print("DataLoader Test")
        print("=" * 60)
        print(
            "Training samples:",
            len(train_loader.dataset),
        )
        print(
            "Validation samples:",
            len(
                validation_loader.dataset
            ),
        )
        print(
            "Training batches:",
            len(train_loader),
        )
        print(
            "Validation batches:",
            len(validation_loader),
        )

        training_batch = next(
            iter(train_loader)
        )

        validation_batch = next(
            iter(validation_loader)
        )

        print(
            "Training image batch:",
            training_batch["image"].shape,
        )
        print(
            "Training mask batch:",
            training_batch["mask"].shape,
        )
        print(
            "Validation image batch:",
            validation_batch["image"].shape,
        )
        print(
            "Validation mask batch:",
            validation_batch["mask"].shape,
        )

    else:
        print(
            "Dataset path does not exist."
        )
        print(
            "Run prepare_datasets.py first."
        )