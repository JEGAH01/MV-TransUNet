"""
Topology-aware dataset pipeline for MV-TransUNetTopology.

This module preserves the original image-level split and patch-sampling
strategy while adding aligned supervision for:

- vessel masks;
- skeleton maps;
- endpoint maps;
- junction maps.

Critical alignment rule
-----------------------
The image, vessel mask, skeleton, endpoint map, and junction map receive
exactly the same geometric augmentation.

Photometric transformations affect only the retinal image.

Expected topology cache structure
---------------------------------
topology_root/
    skeletons/
    endpoints/
    junctions/

Each topology target must share the normalized filename key of its vessel
mask. File extensions may differ.

Example
-------
datasets_topology/DRIVE_train/
    skeletons/
        21_manual1.png
    endpoints/
        21_manual1.png
    junctions/
        21_manual1.png
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.datasets import (
    CLAHEEnhancement,
    IMAGE_EXTENSIONS,
    create_dataloader,
    generate_grid_positions,
    load_binary_mask,
    load_rgb_image,
    load_sample_pairs,
    normalize_filename,
    split_sample_pairs,
)


PathLike = Union[str, Path]
SamplePair = Tuple[Path, Path]


# ============================================================
# TOPOLOGY SAMPLE RECORD
# ============================================================


@dataclass(frozen=True)
class TopologySample:
    """Paths belonging to one retinal image."""

    image_path: Path
    mask_path: Path
    skeleton_path: Path
    endpoint_path: Path
    junction_path: Path


# ============================================================
# FILE MATCHING
# ============================================================


def get_target_files(
    directory: PathLike,
) -> List[Path]:
    """Return supported target files from one directory."""

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Topology target directory not found: "
            f"{directory.resolve()}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory.resolve()}"
        )

    files = [
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]

    if not files:
        raise RuntimeError(
            f"No topology target files found in "
            f"{directory.resolve()}."
        )

    return sorted(
        files,
        key=lambda path: path.name.lower(),
    )


def build_target_dictionary(
    target_dir: PathLike,
    target_name: str,
) -> Dict[str, Path]:
    """Index topology target files by normalized filename."""

    target_dictionary: Dict[str, Path] = {}

    for path in get_target_files(target_dir):
        key = normalize_filename(path.name)

        if key in target_dictionary:
            raise RuntimeError(
                f"Duplicate normalized {target_name} key "
                f"'{key}' for:\n"
                f"  {target_dictionary[key]}\n"
                f"  {path}"
            )

        target_dictionary[key] = path

    return target_dictionary


def load_topology_samples(
    image_dir: PathLike,
    mask_dir: PathLike,
    topology_root: PathLike,
) -> List[TopologySample]:
    """
    Match images, vessel masks, skeletons, endpoints, and junctions.
    """

    topology_root = Path(topology_root)

    skeleton_dictionary = build_target_dictionary(
        topology_root / "skeletons",
        "skeleton",
    )

    endpoint_dictionary = build_target_dictionary(
        topology_root / "endpoints",
        "endpoint",
    )

    junction_dictionary = build_target_dictionary(
        topology_root / "junctions",
        "junction",
    )

    image_mask_pairs = load_sample_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
    )

    samples: List[TopologySample] = []
    missing_records: List[str] = []

    for image_path, mask_path in image_mask_pairs:
        key = normalize_filename(mask_path.name)

        skeleton_path = skeleton_dictionary.get(key)
        endpoint_path = endpoint_dictionary.get(key)
        junction_path = junction_dictionary.get(key)

        missing_targets = []

        if skeleton_path is None:
            missing_targets.append("skeleton")

        if endpoint_path is None:
            missing_targets.append("endpoint")

        if junction_path is None:
            missing_targets.append("junction")

        if missing_targets:
            missing_records.append(
                f"{mask_path.name}: "
                f"{', '.join(missing_targets)}"
            )
            continue

        samples.append(
            TopologySample(
                image_path=image_path,
                mask_path=mask_path,
                skeleton_path=skeleton_path,
                endpoint_path=endpoint_path,
                junction_path=junction_path,
            )
        )

    if missing_records:
        preview = "\n".join(
            missing_records[:10]
        )

        raise RuntimeError(
            "Topology targets are missing for one or more masks.\n"
            f"{preview}\n"
            f"Total incomplete samples: {len(missing_records)}"
        )

    if not samples:
        raise RuntimeError(
            "No complete topology-aware samples were matched."
        )

    return samples


# ============================================================
# TRANSFORMS
# ============================================================


def get_topology_train_transform(
    image_size: int = 256,
) -> A.Compose:
    """
    Build joint image/topology training augmentation.

    All four binary targets are declared as masks, which means geometric
    transformations use nearest-neighbour semantics and photometric
    transformations are not applied to them.
    """

    if image_size < 1:
        raise ValueError(
            "image_size must be positive."
        )

    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),

            A.HorizontalFlip(
                p=0.5,
            ),

            A.VerticalFlip(
                p=0.5,
            ),

            A.Rotate(
                limit=30,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),

            A.ElasticTransform(
                alpha=120,
                sigma=6,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.3,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.5,
            ),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),

            ToTensorV2(),
        ],
        additional_targets={
            "skeleton": "mask",
            "endpoint": "mask",
            "junction": "mask",
        },
    )


def get_topology_validation_transform(
    image_size: int = 256,
) -> A.Compose:
    """Build deterministic aligned validation transformation."""

    if image_size < 1:
        raise ValueError(
            "image_size must be positive."
        )

    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),

            ToTensorV2(),
        ],
        additional_targets={
            "skeleton": "mask",
            "endpoint": "mask",
            "junction": "mask",
        },
    )


# ============================================================
# ARRAY AND TENSOR UTILITIES
# ============================================================


def ensure_same_spatial_shape(
    arrays: Dict[str, np.ndarray],
    sample_name: str,
) -> None:
    """Verify that all topology targets align before augmentation."""

    shapes = {
        name: value.shape[:2]
        for name, value in arrays.items()
    }

    unique_shapes = set(
        shapes.values()
    )

    if len(unique_shapes) != 1:
        raise RuntimeError(
            f"Spatial mismatch for sample '{sample_name}': "
            f"{shapes}"
        )


def binary_tensor(
    value: Union[
        torch.Tensor,
        np.ndarray,
    ],
) -> torch.Tensor:
    """Convert one transformed binary map to [1,H,W] float tensor."""

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    value = value.float()

    if value.ndim == 2:
        value = value.unsqueeze(0)

    elif (
        value.ndim == 3
        and value.shape[-1] == 1
    ):
        value = value.permute(
            2,
            0,
            1,
        )

    if value.ndim != 3:
        raise RuntimeError(
            "Binary target must become a three-dimensional "
            f"[C,H,W] tensor, received {tuple(value.shape)}."
        )

    if value.shape[0] != 1:
        raise RuntimeError(
            "Binary topology targets must contain one channel."
        )

    return (
        value > 0.5
    ).float()


def apply_topology_transform(
    image: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    endpoint: np.ndarray,
    junction: np.ndarray,
    transform: Optional[A.Compose],
) -> Dict[str, torch.Tensor]:
    """Apply one synchronized transformation to all targets."""

    ensure_same_spatial_shape(
        {
            "image": image,
            "mask": mask,
            "skeleton": skeleton,
            "endpoint": endpoint,
            "junction": junction,
        },
        sample_name="current sample",
    )

    if transform is None:
        image_tensor = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float() / 255.0

        return {
            "image": image_tensor,
            "mask": binary_tensor(mask),
            "skeleton": binary_tensor(skeleton),
            "endpoint": binary_tensor(endpoint),
            "junction": binary_tensor(junction),
        }

    transformed = transform(
        image=image,
        mask=mask,
        skeleton=skeleton,
        endpoint=endpoint,
        junction=junction,
    )

    image_tensor = transformed["image"]

    if not torch.is_tensor(image_tensor):
        image_tensor = torch.as_tensor(
            image_tensor
        )

    return {
        "image": image_tensor.float(),
        "mask": binary_tensor(
            transformed["mask"]
        ),
        "skeleton": binary_tensor(
            transformed["skeleton"]
        ),
        "endpoint": binary_tensor(
            transformed["endpoint"]
        ),
        "junction": binary_tensor(
            transformed["junction"]
        ),
    }


def load_sample_arrays(
    sample: TopologySample,
) -> Dict[str, np.ndarray]:
    """Load an image and all four aligned binary targets."""

    arrays = {
        "image": load_rgb_image(
            sample.image_path
        ),
        "mask": load_binary_mask(
            sample.mask_path
        ),
        "skeleton": load_binary_mask(
            sample.skeleton_path
        ),
        "endpoint": load_binary_mask(
            sample.endpoint_path
        ),
        "junction": load_binary_mask(
            sample.junction_path
        ),
    }

    ensure_same_spatial_shape(
        arrays,
        sample_name=sample.image_path.name,
    )

    return arrays


def pad_topology_arrays(
    arrays: Dict[str, np.ndarray],
    minimum_height: int,
    minimum_width: int,
) -> Dict[str, np.ndarray]:
    """
    Pad image reflectively and all supervision maps with zero.
    """

    height, width = arrays["mask"].shape

    pad_bottom = max(
        0,
        minimum_height - height,
    )
    pad_right = max(
        0,
        minimum_width - width,
    )

    if (
        pad_bottom == 0
        and pad_right == 0
    ):
        return arrays

    padded = {
        "image": cv2.copyMakeBorder(
            arrays["image"],
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
    }

    for target_name in (
        "mask",
        "skeleton",
        "endpoint",
        "junction",
    ):
        padded[target_name] = cv2.copyMakeBorder(
            arrays[target_name],
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

    return padded


def crop_topology_patch(
    arrays: Dict[str, np.ndarray],
    top: int,
    left: int,
    patch_height: int,
    patch_width: int,
) -> Dict[str, np.ndarray]:
    """Crop identical spatial coordinates from every array."""

    cropped = {
        name: value[
            top:top + patch_height,
            left:left + patch_width,
        ]
        for name, value in arrays.items()
    }

    expected_shape = (
        patch_height,
        patch_width,
    )

    for name, value in cropped.items():
        if value.shape[:2] != expected_shape:
            raise RuntimeError(
                f"Invalid {name} patch shape "
                f"{value.shape[:2]}, expected {expected_shape}."
            )

    return cropped


def center_to_top_left(
    center_y: int,
    center_x: int,
    image_height: int,
    image_width: int,
    patch_height: int,
    patch_width: int,
) -> Tuple[int, int]:
    """Convert a selected center pixel to valid patch coordinates."""

    top = int(
        center_y - patch_height // 2
    )
    left = int(
        center_x - patch_width // 2
    )

    top = min(
        max(top, 0),
        image_height - patch_height,
    )

    left = min(
        max(left, 0),
        image_width - patch_width,
    )

    return top, left


# ============================================================
# WHOLE-IMAGE DATASET
# ============================================================


class TopologyRetinalDataset(Dataset):
    """Whole-image topology-aware retinal dataset."""

    def __init__(
        self,
        samples: Sequence[TopologySample],
        transform: Optional[A.Compose] = None,
        clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ) -> None:
        super().__init__()

        self.samples = list(samples)

        if not self.samples:
            raise ValueError(
                "TopologyRetinalDataset received no samples."
            )

        self.transform = transform

        self.clahe = (
            CLAHEEnhancement(
                clip_limit=clahe_clip_limit,
                tile_grid_size=(
                    clahe_tile_grid_size
                ),
            )
            if clahe
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[
        str,
        Union[torch.Tensor, str],
    ]:
        sample = self.samples[index]

        arrays = load_sample_arrays(
            sample
        )

        if self.clahe is not None:
            arrays["image"] = self.clahe(
                arrays["image"]
            )

        tensors = apply_topology_transform(
            image=arrays["image"],
            mask=arrays["mask"],
            skeleton=arrays["skeleton"],
            endpoint=arrays["endpoint"],
            junction=arrays["junction"],
            transform=self.transform,
        )

        return {
            **tensors,
            "image_path": str(
                sample.image_path
            ),
            "mask_path": str(
                sample.mask_path
            ),
            "skeleton_path": str(
                sample.skeleton_path
            ),
            "endpoint_path": str(
                sample.endpoint_path
            ),
            "junction_path": str(
                sample.junction_path
            ),
        }


# ============================================================
# RANDOM PATCH DATASET
# ============================================================


class RandomPatchTopologyDataset(Dataset):
    """
    Random topology-aware patch dataset.

    Patch selection is based on the vessel mask. After coordinates are
    selected, the exact same crop is taken from every topology target.
    """

    def __init__(
        self,
        samples: Sequence[TopologySample],
        patch_size: Union[
            int,
            Tuple[int, int],
        ] = 256,
        model_input_size: int = 256,
        patches_per_image: int = 100,
        vessel_center_probability: float = 0.60,
        hard_negative_probability: float = 0.25,
        random_probability: float = 0.15,
        minimum_vessel_fraction: float = 0.01,
        hard_negative_max_vessel_fraction: float = 0.005,
        hard_negative_candidates: int = 20,
        max_sampling_attempts: int = 20,
        transform: Optional[A.Compose] = None,
        clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ) -> None:
        super().__init__()

        self.samples = list(samples)

        if not self.samples:
            raise ValueError(
                "RandomPatchTopologyDataset received no samples."
            )

        if isinstance(
            patch_size,
            int,
        ):
            self.patch_height = int(
                patch_size
            )
            self.patch_width = int(
                patch_size
            )
        else:
            self.patch_height = int(
                patch_size[0]
            )
            self.patch_width = int(
                patch_size[1]
            )

        if (
            self.patch_height < 1
            or self.patch_width < 1
        ):
            raise ValueError(
                "Patch dimensions must be positive."
            )

        probabilities = (
            float(vessel_center_probability),
            float(hard_negative_probability),
            float(random_probability),
        )

        if not np.isclose(
            sum(probabilities),
            1.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Sampling probabilities must sum to one. "
                f"Received {sum(probabilities):.6f}."
            )

        if any(
            probability < 0.0
            for probability in probabilities
        ):
            raise ValueError(
                "Sampling probabilities cannot be negative."
            )

        if patches_per_image < 1:
            raise ValueError(
                "patches_per_image must be at least one."
            )

        self.model_input_size = int(
            model_input_size
        )
        self.patches_per_image = int(
            patches_per_image
        )

        self.vessel_center_probability = float(
            vessel_center_probability
        )
        self.hard_negative_probability = float(
            hard_negative_probability
        )
        self.random_probability = float(
            random_probability
        )

        self.minimum_vessel_fraction = float(
            minimum_vessel_fraction
        )

        self.hard_negative_max_vessel_fraction = float(
            hard_negative_max_vessel_fraction
        )

        self.hard_negative_candidates = int(
            hard_negative_candidates
        )

        self.max_sampling_attempts = int(
            max_sampling_attempts
        )

        self.transform = (
            transform
            if transform is not None
            else get_topology_train_transform(
                self.model_input_size
            )
        )

        self.clahe = (
            CLAHEEnhancement(
                clip_limit=clahe_clip_limit,
                tile_grid_size=(
                    clahe_tile_grid_size
                ),
            )
            if clahe
            else None
        )

    def __len__(self) -> int:
        return (
            len(self.samples)
            * self.patches_per_image
        )

    def _random_coordinates(
        self,
        mask: np.ndarray,
    ) -> Tuple[int, int]:
        """Sample valid uniform patch coordinates."""

        max_top = (
            mask.shape[0]
            - self.patch_height
        )
        max_left = (
            mask.shape[1]
            - self.patch_width
        )

        top = int(
            np.random.randint(
                0,
                max_top + 1,
            )
        )

        left = int(
            np.random.randint(
                0,
                max_left + 1,
            )
        )

        return top, left

    def _vessel_coordinates(
        self,
        mask: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Select a patch centered near a vessel pixel."""

        vessel_coordinates = np.argwhere(
            mask > 0.5
        )

        if vessel_coordinates.shape[0] == 0:
            return None

        selected_index = int(
            np.random.randint(
                0,
                vessel_coordinates.shape[0],
            )
        )

        center_y, center_x = (
            vessel_coordinates[selected_index]
        )

        return center_to_top_left(
            center_y=int(center_y),
            center_x=int(center_x),
            image_height=mask.shape[0],
            image_width=mask.shape[1],
            patch_height=self.patch_height,
            patch_width=self.patch_width,
        )

    @staticmethod
    def _edge_energy(
        image_patch: np.ndarray,
    ) -> float:
        """Calculate normalized Sobel edge energy."""

        grayscale = cv2.cvtColor(
            image_patch,
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32)

        gradient_x = cv2.Sobel(
            grayscale,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )

        gradient_y = cv2.Sobel(
            grayscale,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        magnitude = cv2.magnitude(
            gradient_x,
            gradient_y,
        )

        return float(
            magnitude.mean() / 255.0
        )

    def _choose_vessel_patch(
        self,
        arrays: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        float,
        float,
    ]:
        """Try to find a patch with sufficient vessel content."""

        best_patch = None
        best_fraction = -1.0
        best_energy = 0.0

        for _ in range(
            self.max_sampling_attempts
        ):
            coordinates = self._vessel_coordinates(
                arrays["mask"]
            )

            if coordinates is None:
                coordinates = self._random_coordinates(
                    arrays["mask"]
                )

            top, left = coordinates

            patch = crop_topology_patch(
                arrays=arrays,
                top=top,
                left=left,
                patch_height=self.patch_height,
                patch_width=self.patch_width,
            )

            vessel_fraction = float(
                patch["mask"].mean()
            )

            edge_energy = self._edge_energy(
                patch["image"]
            )

            if vessel_fraction > best_fraction:
                best_patch = patch
                best_fraction = vessel_fraction
                best_energy = edge_energy

            if (
                vessel_fraction
                >= self.minimum_vessel_fraction
            ):
                return (
                    patch,
                    vessel_fraction,
                    edge_energy,
                )

        if best_patch is None:
            raise RuntimeError(
                "Vessel patch sampling failed."
            )

        return (
            best_patch,
            best_fraction,
            best_energy,
        )

    def _choose_hard_negative_patch(
        self,
        arrays: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        float,
        float,
    ]:
        """Select a low-vessel, high-texture patch."""

        eligible_candidates = []
        fallback_candidates = []

        for _ in range(
            self.hard_negative_candidates
        ):
            top, left = self._random_coordinates(
                arrays["mask"]
            )

            patch = crop_topology_patch(
                arrays=arrays,
                top=top,
                left=left,
                patch_height=self.patch_height,
                patch_width=self.patch_width,
            )

            vessel_fraction = float(
                patch["mask"].mean()
            )

            edge_energy = self._edge_energy(
                patch["image"]
            )

            fallback_candidates.append(
                (
                    vessel_fraction,
                    -edge_energy,
                    patch,
                    edge_energy,
                )
            )

            if (
                vessel_fraction
                <= self.hard_negative_max_vessel_fraction
            ):
                eligible_candidates.append(
                    (
                        edge_energy,
                        patch,
                        vessel_fraction,
                    )
                )

        if eligible_candidates:
            (
                edge_energy,
                patch,
                vessel_fraction,
            ) = max(
                eligible_candidates,
                key=lambda item: item[0],
            )

            return (
                patch,
                vessel_fraction,
                edge_energy,
            )

        (
            vessel_fraction,
            _,
            patch,
            edge_energy,
        ) = min(
            fallback_candidates,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

        return (
            patch,
            vessel_fraction,
            edge_energy,
        )

    def _choose_random_patch(
        self,
        arrays: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        float,
        float,
    ]:
        """Select one uniform random patch."""

        top, left = self._random_coordinates(
            arrays["mask"]
        )

        patch = crop_topology_patch(
            arrays=arrays,
            top=top,
            left=left,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
        )

        return (
            patch,
            float(patch["mask"].mean()),
            self._edge_energy(
                patch["image"]
            ),
        )

    def _select_patch(
        self,
        arrays: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        str,
        float,
        float,
    ]:
        """Choose vessel, hard-negative, or random sampling."""

        random_value = float(
            np.random.random()
        )

        hard_negative_boundary = (
            self.vessel_center_probability
            + self.hard_negative_probability
        )

        if (
            random_value
            < self.vessel_center_probability
        ):
            (
                patch,
                vessel_fraction,
                edge_energy,
            ) = self._choose_vessel_patch(
                arrays
            )

            sample_type = "vessel"

        elif (
            random_value
            < hard_negative_boundary
        ):
            (
                patch,
                vessel_fraction,
                edge_energy,
            ) = self._choose_hard_negative_patch(
                arrays
            )

            sample_type = "hard_negative"

        else:
            (
                patch,
                vessel_fraction,
                edge_energy,
            ) = self._choose_random_patch(
                arrays
            )

            sample_type = "random"

        return (
            patch,
            sample_type,
            vessel_fraction,
            edge_energy,
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[
        str,
        Union[
            torch.Tensor,
            str,
            int,
            float,
            bool,
        ],
    ]:
        image_index = (
            index // self.patches_per_image
        ) % len(self.samples)

        sample = self.samples[image_index]

        arrays = load_sample_arrays(
            sample
        )

        arrays = pad_topology_arrays(
            arrays=arrays,
            minimum_height=self.patch_height,
            minimum_width=self.patch_width,
        )

        if self.clahe is not None:
            arrays["image"] = self.clahe(
                arrays["image"]
            )

        (
            patch,
            sample_type,
            vessel_fraction,
            edge_energy,
        ) = self._select_patch(
            arrays
        )

        tensors = apply_topology_transform(
            image=patch["image"],
            mask=patch["mask"],
            skeleton=patch["skeleton"],
            endpoint=patch["endpoint"],
            junction=patch["junction"],
            transform=self.transform,
        )

        return {
            **tensors,
            "image_path": str(
                sample.image_path
            ),
            "mask_path": str(
                sample.mask_path
            ),
            "source_image_index": image_index,
            "patch_index": index,
            "sample_type": sample_type,
            "vessel_fraction": vessel_fraction,
            "edge_energy": edge_energy,
            "vessel_centred": (
                sample_type == "vessel"
            ),
            "hard_negative": (
                sample_type
                == "hard_negative"
            ),
        }


# ============================================================
# GRID VALIDATION DATASET
# ============================================================


class GridPatchTopologyDataset(Dataset):
    """Deterministic grid-patch topology validation dataset."""

    def __init__(
        self,
        samples: Sequence[TopologySample],
        patch_size: Union[
            int,
            Tuple[int, int],
        ] = 256,
        model_input_size: int = 256,
        stride: int = 128,
        transform: Optional[A.Compose] = None,
        clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        include_empty_patches: bool = True,
        minimum_vessel_fraction: float = 0.0,
    ) -> None:
        super().__init__()

        self.samples = list(samples)

        if not self.samples:
            raise ValueError(
                "GridPatchTopologyDataset received no samples."
            )

        if isinstance(
            patch_size,
            int,
        ):
            self.patch_height = int(
                patch_size
            )
            self.patch_width = int(
                patch_size
            )
        else:
            self.patch_height = int(
                patch_size[0]
            )
            self.patch_width = int(
                patch_size[1]
            )

        self.model_input_size = int(
            model_input_size
        )
        self.stride = int(stride)

        self.include_empty_patches = bool(
            include_empty_patches
        )

        self.minimum_vessel_fraction = float(
            minimum_vessel_fraction
        )

        self.transform = (
            transform
            if transform is not None
            else get_topology_validation_transform(
                self.model_input_size
            )
        )

        self.clahe = (
            CLAHEEnhancement(
                clip_limit=clahe_clip_limit,
                tile_grid_size=(
                    clahe_tile_grid_size
                ),
            )
            if clahe
            else None
        )

        self.patch_records: List[
            Tuple[int, int, int]
        ] = []

        self._build_patch_records()

    def _build_patch_records(
        self,
    ) -> None:
        """Precompute deterministic validation patch positions."""

        for sample_index, sample in enumerate(
            self.samples
        ):
            mask = load_binary_mask(
                sample.mask_path
            )

            dummy_arrays = {
                "image": np.zeros(
                    (
                        mask.shape[0],
                        mask.shape[1],
                        3,
                    ),
                    dtype=np.uint8,
                ),
                "mask": mask,
                "skeleton": np.zeros_like(
                    mask
                ),
                "endpoint": np.zeros_like(
                    mask
                ),
                "junction": np.zeros_like(
                    mask
                ),
            }

            dummy_arrays = pad_topology_arrays(
                arrays=dummy_arrays,
                minimum_height=self.patch_height,
                minimum_width=self.patch_width,
            )

            padded_mask = dummy_arrays[
                "mask"
            ]

            positions = generate_grid_positions(
                image_height=padded_mask.shape[0],
                image_width=padded_mask.shape[1],
                patch_height=self.patch_height,
                patch_width=self.patch_width,
                stride=self.stride,
            )

            for top, left in positions:
                mask_patch = padded_mask[
                    top:top + self.patch_height,
                    left:left + self.patch_width,
                ]

                vessel_fraction = float(
                    mask_patch.mean()
                )

                if (
                    self.include_empty_patches
                    or vessel_fraction
                    >= self.minimum_vessel_fraction
                ):
                    self.patch_records.append(
                        (
                            sample_index,
                            top,
                            left,
                        )
                    )

        if not self.patch_records:
            raise RuntimeError(
                "No topology validation patches were generated."
            )

    def __len__(self) -> int:
        return len(
            self.patch_records
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[
        str,
        Union[
            torch.Tensor,
            str,
            int,
            float,
        ],
    ]:
        (
            sample_index,
            top,
            left,
        ) = self.patch_records[index]

        sample = self.samples[
            sample_index
        ]

        arrays = load_sample_arrays(
            sample
        )

        arrays = pad_topology_arrays(
            arrays=arrays,
            minimum_height=self.patch_height,
            minimum_width=self.patch_width,
        )

        if self.clahe is not None:
            arrays["image"] = self.clahe(
                arrays["image"]
            )

        patch = crop_topology_patch(
            arrays=arrays,
            top=top,
            left=left,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
        )

        tensors = apply_topology_transform(
            image=patch["image"],
            mask=patch["mask"],
            skeleton=patch["skeleton"],
            endpoint=patch["endpoint"],
            junction=patch["junction"],
            transform=self.transform,
        )

        return {
            **tensors,
            "image_path": str(
                sample.image_path
            ),
            "mask_path": str(
                sample.mask_path
            ),
            "source_image_index": sample_index,
            "patch_index": index,
            "top": top,
            "left": left,
            "vessel_fraction": float(
                patch["mask"].mean()
            ),
        }


# ============================================================
# SPLITTING
# ============================================================


def split_topology_samples(
    samples: Sequence[TopologySample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[
    List[TopologySample],
    List[TopologySample],
]:
    """Create deterministic image-level topology sample splits."""

    if len(samples) < 2:
        raise ValueError(
            "At least two samples are required."
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between zero and one."
        )

    sample_pairs: List[SamplePair] = [
        (
            sample.image_path,
            sample.mask_path,
        )
        for sample in samples
    ]

    training_pairs, validation_pairs = (
        split_sample_pairs(
            samples=sample_pairs,
            validation_ratio=validation_ratio,
            seed=seed,
        )
    )

    sample_dictionary = {
        normalize_filename(
            sample.mask_path.name
        ): sample
        for sample in samples
    }

    training_samples = [
        sample_dictionary[
            normalize_filename(
                mask_path.name
            )
        ]
        for _, mask_path in training_pairs
    ]

    validation_samples = [
        sample_dictionary[
            normalize_filename(
                mask_path.name
            )
        ]
        for _, mask_path in validation_pairs
    ]

    return (
        training_samples,
        validation_samples,
    )


# ============================================================
# DATALOADER BUILDERS
# ============================================================


def build_topology_dataloaders(
    image_dir: PathLike,
    mask_dir: PathLike,
    topology_root: PathLike,
    image_size: int = 256,
    batch_size: int = 4,
    validation_ratio: float = 0.2,
    num_workers: int = 2,
    clahe: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build topology-aware whole-image loaders."""

    samples = load_topology_samples(
        image_dir=image_dir,
        mask_dir=mask_dir,
        topology_root=topology_root,
    )

    (
        training_samples,
        validation_samples,
    ) = split_topology_samples(
        samples=samples,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    training_dataset = TopologyRetinalDataset(
        samples=training_samples,
        transform=get_topology_train_transform(
            image_size
        ),
        clahe=clahe,
    )

    validation_dataset = TopologyRetinalDataset(
        samples=validation_samples,
        transform=get_topology_validation_transform(
            image_size
        ),
        clahe=clahe,
    )

    training_loader = create_dataloader(
        dataset=training_dataset,
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
        training_loader,
        validation_loader,
    )


def build_topology_patch_dataloaders(
    image_dir: PathLike,
    mask_dir: PathLike,
    topology_root: PathLike,
    patch_size: int = 256,
    model_input_size: int = 256,
    patches_per_image: int = 100,
    validation_stride: int = 128,
    batch_size: int = 4,
    validation_ratio: float = 0.2,
    vessel_center_probability: float = 0.60,
    hard_negative_probability: float = 0.25,
    random_probability: float = 0.15,
    minimum_vessel_fraction: float = 0.01,
    hard_negative_max_vessel_fraction: float = 0.005,
    hard_negative_candidates: int = 20,
    max_sampling_attempts: int = 20,
    num_workers: int = 2,
    clahe: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
    include_empty_validation_patches: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build topology-aware random-train/grid-validation loaders."""

    samples = load_topology_samples(
        image_dir=image_dir,
        mask_dir=mask_dir,
        topology_root=topology_root,
    )

    (
        training_samples,
        validation_samples,
    ) = split_topology_samples(
        samples=samples,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    training_dataset = RandomPatchTopologyDataset(
        samples=training_samples,
        patch_size=patch_size,
        model_input_size=model_input_size,
        patches_per_image=patches_per_image,
        vessel_center_probability=(
            vessel_center_probability
        ),
        hard_negative_probability=(
            hard_negative_probability
        ),
        random_probability=random_probability,
        minimum_vessel_fraction=(
            minimum_vessel_fraction
        ),
        hard_negative_max_vessel_fraction=(
            hard_negative_max_vessel_fraction
        ),
        hard_negative_candidates=(
            hard_negative_candidates
        ),
        max_sampling_attempts=(
            max_sampling_attempts
        ),
        transform=get_topology_train_transform(
            model_input_size
        ),
        clahe=clahe,
    )

    validation_dataset = GridPatchTopologyDataset(
        samples=validation_samples,
        patch_size=patch_size,
        model_input_size=model_input_size,
        stride=validation_stride,
        transform=(
            get_topology_validation_transform(
                model_input_size
            )
        ),
        clahe=clahe,
        include_empty_patches=(
            include_empty_validation_patches
        ),
        minimum_vessel_fraction=0.0,
    )

    training_loader = create_dataloader(
        dataset=training_dataset,
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
        training_loader,
        validation_loader,
    )


# ============================================================
# PIPELINE SMOKE TEST
# ============================================================


def verify_topology_batch(
    batch: Dict[str, object],
    expected_size: int,
    batch_name: str,
) -> None:
    """Validate one topology-aware DataLoader batch."""

    required_keys = {
        "image",
        "mask",
        "skeleton",
        "endpoint",
        "junction",
    }

    missing_keys = (
        required_keys
        - set(batch.keys())
    )

    if missing_keys:
        raise KeyError(
            f"{batch_name} is missing: "
            f"{sorted(missing_keys)}"
        )

    image = batch["image"]

    if not torch.is_tensor(image):
        raise TypeError(
            f"{batch_name} image must be a tensor."
        )

    batch_size = image.shape[0]

    expected_image_shape = (
        batch_size,
        3,
        expected_size,
        expected_size,
    )

    if tuple(image.shape) != expected_image_shape:
        raise RuntimeError(
            f"{batch_name} image shape is "
            f"{tuple(image.shape)}, expected "
            f"{expected_image_shape}."
        )

    for target_name in (
        "mask",
        "skeleton",
        "endpoint",
        "junction",
    ):
        target = batch[target_name]

        expected_target_shape = (
            batch_size,
            1,
            expected_size,
            expected_size,
        )

        if not torch.is_tensor(target):
            raise TypeError(
                f"{target_name} must be a tensor."
            )

        if tuple(
            target.shape
        ) != expected_target_shape:
            raise RuntimeError(
                f"{batch_name} {target_name} shape is "
                f"{tuple(target.shape)}, expected "
                f"{expected_target_shape}."
            )

        unique_values = torch.unique(
            target
        )

        valid_values = torch.all(
            (
                unique_values == 0.0
            )
            | (
                unique_values == 1.0
            )
        )

        if not bool(valid_values):
            raise RuntimeError(
                f"{batch_name} {target_name} is not binary: "
                f"{unique_values.tolist()}"
            )

def resolve_drive_training_paths() -> Tuple[Path, Path]:
    """
    Resolve the DRIVE training image and vessel-mask directories.

    Supported raw layouts:
        datasets/DRIVE/training/image
        datasets/DRIVE/training/1st_manual

        datasets/DRIVE/training/images
        datasets/DRIVE/training/1st_manual

    Supported processed layout:
        datasets_processed/DRIVE/images
        datasets_processed/DRIVE/masks

    The official raw DRIVE training layout is preferred to avoid mixing
    training and test images from a combined processed directory.
    """

    candidate_layouts = [
        (
            Path(
                "./datasets/DRIVE/training/image"
            ),
            Path(
                "./datasets/DRIVE/training/1st_manual"
            ),
            "raw DRIVE layout using 'image'",
        ),
        (
            Path(
                "./datasets/DRIVE/training/images"
            ),
            Path(
                "./datasets/DRIVE/training/1st_manual"
            ),
            "raw DRIVE layout using 'images'",
        ),
        (
            Path(
                "./datasets_processed/DRIVE/images"
            ),
            Path(
                "./datasets_processed/DRIVE/masks"
            ),
            "processed DRIVE layout",
        ),
    ]

    for (
        image_dir,
        mask_dir,
        layout_name,
    ) in candidate_layouts:
        if (
            image_dir.is_dir()
            and mask_dir.is_dir()
        ):
            print(
                f"Using {layout_name}:"
            )
            print(
                f"  Images: {image_dir.resolve()}"
            )
            print(
                f"  Masks:  {mask_dir.resolve()}"
            )

            return (
                image_dir,
                mask_dir,
            )

    searched_locations = "\n".join(
        (
            f"  Layout: {layout_name}\n"
            f"  Images: {image_dir.resolve()}\n"
            f"  Masks:  {mask_dir.resolve()}"
        )
        for (
            image_dir,
            mask_dir,
            layout_name,
        ) in candidate_layouts
    )

    raise FileNotFoundError(
        "Could not find a valid DRIVE training layout.\n"
        "Searched:\n"
        f"{searched_locations}"
    )

def run_topology_dataset_smoke_test() -> None:
    """
    Run a DRIVE topology-loader smoke test.

    Dataset layout selection:
    - datasets_processed uses images/ and masks/
    - raw datasets uses training/images/ and training/1st_manual/
    """

    (
        image_dir,
        mask_dir,
    ) = resolve_drive_training_paths()

    topology_root = Path(
        "./datasets_topology/DRIVE_train"
    )

    if not topology_root.is_dir():
        raise FileNotFoundError(
            "DRIVE topology directory was not found: "
            f"{topology_root.resolve()}"
        )

    model_input_size = 256

    training_loader, validation_loader = (
        build_topology_patch_dataloaders(
            image_dir=image_dir,
            mask_dir=mask_dir,
            topology_root=topology_root,
            patch_size=256,
            model_input_size=model_input_size,
            patches_per_image=4,
            validation_stride=128,
            batch_size=2,
            validation_ratio=0.2,
            vessel_center_probability=0.60,
            hard_negative_probability=0.25,
            random_probability=0.15,
            minimum_vessel_fraction=0.01,
            hard_negative_max_vessel_fraction=0.005,
            hard_negative_candidates=5,
            max_sampling_attempts=10,
            num_workers=0,
            clahe=True,
            seed=42,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
            include_empty_validation_patches=True,
        )
    )

    training_batch = next(
        iter(training_loader)
    )

    validation_batch = next(
        iter(validation_loader)
    )

    verify_topology_batch(
        batch=training_batch,
        expected_size=model_input_size,
        batch_name="Training batch",
    )

    verify_topology_batch(
        batch=validation_batch,
        expected_size=model_input_size,
        batch_name="Validation batch",
    )

    print("=" * 72)
    print(
        "Topology Dataset Pipeline Smoke Test"
    )
    print("=" * 72)

    print(
        "Training image samples:",
        len(
            training_loader.dataset.samples
        ),
    )

    print(
        "Validation image samples:",
        len(
            validation_loader.dataset.samples
        ),
    )

    print(
        "Training patch samples:",
        len(training_loader.dataset),
    )

    print(
        "Validation patch samples:",
        len(validation_loader.dataset),
    )

    print()

    for name in (
        "image",
        "mask",
        "skeleton",
        "endpoint",
        "junction",
    ):
        print(
            f"Training {name:12s}",
            tuple(
                training_batch[name].shape
            ),
        )

    print()

    print(
        "Training sample types:",
        list(
            training_batch[
                "sample_type"
            ]
        ),
    )

    print(
        "Training vessel fractions:",
        training_batch[
            "vessel_fraction"
        ].tolist(),
    )

    print()

    print("=" * 72)
    print(
        "Topology file matching: PASSED"
    )
    print(
        "Image-level split: PASSED"
    )
    print(
        "Aligned patch extraction: PASSED"
    )
    print(
        "Joint geometric augmentation: PASSED"
    )
    print(
        "Binary target preservation: PASSED"
    )
    print("=" * 72)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    run_topology_dataset_smoke_test()