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
- Whole-image training and evaluation loaders
- Patch-based training with vessel-centred oversampling
- Texture-aware hard-negative patch mining
- Deterministic image-level train-validation splitting
- Deterministic grid-based validation patches
- Separate training and validation transforms
- CLAHE enhancement
- Albumentations augmentation
- Reproducible PyTorch DataLoader workers
- Cross-dataset evaluation support

Important protocol rule:
Training and validation are split at IMAGE level before patches are created.
This prevents patches from the same retinal image appearing in both sets.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset


PathLike = Union[str, Path]
SamplePair = Tuple[Path, Path]


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".gif"
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CLAHEEnhancement:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8) -> None:
        self.clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("CLAHE received an empty image.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("CLAHE expects an RGB image with three channels.")
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        enhanced_l_channel = self.clahe.apply(l_channel)
        enhanced_lab = cv2.merge([enhanced_l_channel, a_channel, b_channel])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def get_train_transform(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=6,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.3,
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_validation_transform(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_files(directory: PathLike) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory.resolve()}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory: {directory.resolve()}")
    files = [
        file for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.name.lower())


def normalize_filename(filename: PathLike) -> str:
    name = Path(filename).stem.lower()
    replacements = [
        "_training_mask", "_test_mask", "_training", "_test",
        "_manual1", "_manual", "_mask", "_1stho", "_2ndho",
        ".ah", ".vk",
    ]
    for item in replacements:
        name = name.replace(item, "")
    return name.strip()


def match_image_masks(
    image_files: Sequence[Path],
    mask_files: Sequence[Path],
) -> List[SamplePair]:
    if len(image_files) == 0:
        raise RuntimeError("No image files were found.")
    if len(mask_files) == 0:
        raise RuntimeError("No mask files were found.")

    mask_dictionary: Dict[str, Path] = {}
    for mask_path in mask_files:
        key = normalize_filename(mask_path.name)
        if key not in mask_dictionary:
            mask_dictionary[key] = mask_path

    pairs: List[SamplePair] = []
    unmatched_images: List[str] = []
    for image_path in image_files:
        key = normalize_filename(image_path.name)
        mask_path = mask_dictionary.get(key)
        if mask_path is None:
            unmatched_images.append(image_path.name)
        else:
            pairs.append((image_path, mask_path))

    if len(pairs) == 0:
        raise RuntimeError("No matching image-mask pairs were found.")
    if unmatched_images:
        print(f"Warning: {len(unmatched_images)} images did not have matching masks.")
    return pairs


def load_sample_pairs(image_dir: PathLike, mask_dir: PathLike) -> List[SamplePair]:
    return match_image_masks(get_files(image_dir), get_files(mask_dir))


def load_rgb_image(image_path: PathLike) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_binary_mask(mask_path: PathLike) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    return (mask > 127).astype(np.float32)


def apply_transform(
    image: np.ndarray,
    mask: np.ndarray,
    transform: Optional[A.Compose],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if transform is not None:
        transformed = transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]
    else:
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float()

    if not torch.is_tensor(image_tensor):
        image_tensor = torch.as_tensor(image_tensor)
    if not torch.is_tensor(mask_tensor):
        mask_tensor = torch.as_tensor(mask_tensor)

    image_tensor = image_tensor.float()
    mask_tensor = mask_tensor.float()

    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    elif mask_tensor.ndim == 3 and mask_tensor.shape[-1] == 1:
        mask_tensor = mask_tensor.permute(2, 0, 1)

    mask_tensor = (mask_tensor > 0.5).float()
    return image_tensor, mask_tensor


def pad_to_minimum_size(
    image: np.ndarray,
    mask: np.ndarray,
    minimum_height: int,
    minimum_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_bottom = max(0, minimum_height - height)
    pad_right = max(0, minimum_width - width)
    if pad_bottom == 0 and pad_right == 0:
        return image, mask

    image = cv2.copyMakeBorder(
        image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101
    )
    mask = cv2.copyMakeBorder(
        mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    return image, mask


def crop_patch(
    image: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    patch_height: int,
    patch_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    image_patch = image[top:top + patch_height, left:left + patch_width]
    mask_patch = mask[top:top + patch_height, left:left + patch_width]
    if image_patch.shape[:2] != (patch_height, patch_width):
        raise RuntimeError("Image patch extraction produced an invalid shape.")
    if mask_patch.shape[:2] != (patch_height, patch_width):
        raise RuntimeError("Mask patch extraction produced an invalid shape.")
    return image_patch, mask_patch


def center_to_top_left(
    center_y: int,
    center_x: int,
    image_height: int,
    image_width: int,
    patch_height: int,
    patch_width: int,
) -> Tuple[int, int]:
    top = int(center_y - patch_height // 2)
    left = int(center_x - patch_width // 2)
    top = min(max(top, 0), image_height - patch_height)
    left = min(max(left, 0), image_width - patch_width)
    return top, left


def generate_grid_positions(
    image_height: int,
    image_width: int,
    patch_height: int,
    patch_width: int,
    stride: int,
) -> List[Tuple[int, int]]:
    if stride < 1:
        raise ValueError("Patch stride must be at least 1.")

    max_top = max(0, image_height - patch_height)
    max_left = max(0, image_width - patch_width)

    top_positions = list(range(0, max_top + 1, stride)) or [0]
    left_positions = list(range(0, max_left + 1, stride)) or [0]

    if top_positions[-1] != max_top:
        top_positions.append(max_top)
    if left_positions[-1] != max_left:
        left_positions.append(max_left)

    return [(top, left) for top in top_positions for left in left_positions]


class RetinalVesselDataset(Dataset):
    """Whole-image retinal vessel segmentation dataset."""

    def __init__(
        self,
        image_dir: PathLike,
        mask_dir: PathLike,
        transform: Optional[A.Compose] = None,
        clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        samples: Optional[Sequence[SamplePair]] = None,
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.samples = (
            load_sample_pairs(self.image_dir, self.mask_dir)
            if samples is None else list(samples)
        )
        self.clahe = (
            CLAHEEnhancement(clahe_clip_limit, clahe_tile_grid_size)
            if clahe else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_path, mask_path = self.samples[index]
        image = load_rgb_image(image_path)
        mask = load_binary_mask(mask_path)
        if self.clahe is not None:
            image = self.clahe(image)
        image_tensor, mask_tensor = apply_transform(image, mask, self.transform)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


class RandomPatchRetinalDataset(Dataset):
    """
    Random patch dataset with vessel-centred, hard-negative, and
    uniform-random sampling.

    The three sampling probabilities must sum to 1.0.
    """

    def __init__(
        self,
        samples: Sequence[SamplePair],
        patch_size: Union[int, Tuple[int, int]] = 256,
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
            raise ValueError("RandomPatchRetinalDataset received no samples.")

        if isinstance(patch_size, int):
            self.patch_height = int(patch_size)
            self.patch_width = int(patch_size)
        else:
            self.patch_height = int(patch_size[0])
            self.patch_width = int(patch_size[1])

        if self.patch_height < 1 or self.patch_width < 1:
            raise ValueError("Patch dimensions must be positive.")
        if patches_per_image < 1:
            raise ValueError("patches_per_image must be at least 1.")

        probabilities = (
            float(vessel_center_probability),
            float(hard_negative_probability),
            float(random_probability),
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError(
                "All patch-sampling probabilities must be between 0 and 1."
            )
        if not np.isclose(sum(probabilities), 1.0, atol=1e-6):
            raise ValueError(
                "Patch-sampling probabilities must sum to 1.0. "
                f"Received {sum(probabilities):.6f}."
            )

        if not 0.0 <= minimum_vessel_fraction <= 1.0:
            raise ValueError(
                "minimum_vessel_fraction must be between 0 and 1."
            )
        if not 0.0 <= hard_negative_max_vessel_fraction <= 1.0:
            raise ValueError(
                "hard_negative_max_vessel_fraction must be between 0 and 1."
            )
        if hard_negative_candidates < 1:
            raise ValueError("hard_negative_candidates must be at least 1.")
        if max_sampling_attempts < 1:
            raise ValueError("max_sampling_attempts must be at least 1.")

        self.model_input_size = int(model_input_size)
        self.patches_per_image = int(patches_per_image)
        self.vessel_center_probability = float(vessel_center_probability)
        self.hard_negative_probability = float(hard_negative_probability)
        self.random_probability = float(random_probability)
        self.minimum_vessel_fraction = float(minimum_vessel_fraction)
        self.hard_negative_max_vessel_fraction = float(
            hard_negative_max_vessel_fraction
        )
        self.hard_negative_candidates = int(hard_negative_candidates)
        self.max_sampling_attempts = int(max_sampling_attempts)

        self.transform = (
            transform
            if transform is not None
            else get_train_transform(self.model_input_size)
        )
        self.clahe = (
            CLAHEEnhancement(
                clahe_clip_limit,
                clahe_tile_grid_size,
            )
            if clahe
            else None
        )

        # RAM Caching: Pre-load all source images and masks into RAM
        self.cached_images = [load_rgb_image(p[0]) for p in self.samples]
        self.cached_masks = [load_binary_mask(p[1]) for p in self.samples]

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def _sample_random_coordinates(
        self,
        image_height: int,
        image_width: int,
    ) -> Tuple[int, int]:
        max_top = image_height - self.patch_height
        max_left = image_width - self.patch_width
        top = int(np.random.randint(0, max_top + 1))
        left = int(np.random.randint(0, max_left + 1))
        return top, left

    def _sample_vessel_coordinates(
        self,
        mask: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        vessel_coordinates = np.argwhere(mask > 0.5)
        if vessel_coordinates.shape[0] == 0:
            return None

        coordinate_index = int(
            np.random.randint(0, vessel_coordinates.shape[0])
        )
        center_y, center_x = vessel_coordinates[coordinate_index]

        return center_to_top_left(
            center_y=int(center_y),
            center_x=int(center_x),
            image_height=mask.shape[0],
            image_width=mask.shape[1],
            patch_height=self.patch_height,
            patch_width=self.patch_width,
        )

    @staticmethod
    def _calculate_edge_energy(
        image_patch: np.ndarray,
    ) -> float:
        """Calculate normalized Sobel-gradient energy."""

        grayscale = cv2.cvtColor(
            image_patch,
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32)

        sobel_x = cv2.Sobel(
            grayscale,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
        sobel_y = cv2.Sobel(
            grayscale,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        return float(magnitude.mean() / 255.0)

    def _sample_vessel_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        best_patch = None
        best_vessel_fraction = -1.0

        for _ in range(self.max_sampling_attempts):
            coordinates = self._sample_vessel_coordinates(mask)
            if coordinates is None:
                coordinates = self._sample_random_coordinates(
                    mask.shape[0],
                    mask.shape[1],
                )

            top, left = coordinates
            image_patch, mask_patch = crop_patch(
                image,
                mask,
                top,
                left,
                self.patch_height,
                self.patch_width,
            )
            vessel_fraction = float(mask_patch.mean())
            edge_energy = self._calculate_edge_energy(image_patch)

            if vessel_fraction > best_vessel_fraction:
                best_patch = (
                    image_patch,
                    mask_patch,
                    vessel_fraction,
                    edge_energy,
                )
                best_vessel_fraction = vessel_fraction

            if vessel_fraction >= self.minimum_vessel_fraction:
                return (
                    image_patch,
                    mask_patch,
                    vessel_fraction,
                    edge_energy,
                )

        if best_patch is None:
            raise RuntimeError("Vessel-centred patch sampling failed.")
        return best_patch

    def _sample_hard_negative_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        eligible_candidates = []
        fallback_candidates = []

        for _ in range(self.hard_negative_candidates):
            top, left = self._sample_random_coordinates(
                mask.shape[0],
                mask.shape[1],
            )
            image_patch, mask_patch = crop_patch(
                image,
                mask,
                top,
                left,
                self.patch_height,
                self.patch_width,
            )
            vessel_fraction = float(mask_patch.mean())
            edge_energy = self._calculate_edge_energy(image_patch)

            fallback_candidates.append(
                (
                    vessel_fraction,
                    -edge_energy,
                    image_patch,
                    mask_patch,
                )
            )

            if (
                vessel_fraction
                <= self.hard_negative_max_vessel_fraction
            ):
                eligible_candidates.append(
                    (
                        edge_energy,
                        image_patch,
                        mask_patch,
                        vessel_fraction,
                    )
                )

        if eligible_candidates:
            edge_energy, image_patch, mask_patch, vessel_fraction = max(
                eligible_candidates,
                key=lambda item: item[0],
            )
            return (
                image_patch,
                mask_patch,
                vessel_fraction,
                edge_energy,
            )

        vessel_fraction, negative_edge_energy, image_patch, mask_patch = min(
            fallback_candidates,
            key=lambda item: (item[0], item[1]),
        )
        return (
            image_patch,
            mask_patch,
            vessel_fraction,
            -negative_edge_energy,
        )

    def _sample_uniform_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        top, left = self._sample_random_coordinates(
            mask.shape[0],
            mask.shape[1],
        )
        image_patch, mask_patch = crop_patch(
            image,
            mask,
            top,
            left,
            self.patch_height,
            self.patch_width,
        )
        vessel_fraction = float(mask_patch.mean())
        edge_energy = self._calculate_edge_energy(image_patch)
        return image_patch, mask_patch, vessel_fraction, edge_energy

    def _select_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, str, float, float]:
        random_value = float(np.random.random())
        hard_negative_boundary = (
            self.vessel_center_probability
            + self.hard_negative_probability
        )

        if random_value < self.vessel_center_probability:
            image_patch, mask_patch, vessel_fraction, edge_energy = (
                self._sample_vessel_patch(image, mask)
            )
            sample_type = "vessel"
        elif random_value < hard_negative_boundary:
            image_patch, mask_patch, vessel_fraction, edge_energy = (
                self._sample_hard_negative_patch(image, mask)
            )
            sample_type = "hard_negative"
        else:
            image_patch, mask_patch, vessel_fraction, edge_energy = (
                self._sample_uniform_patch(image, mask)
            )
            sample_type = "random"

        return (
            image_patch,
            mask_patch,
            sample_type,
            vessel_fraction,
            edge_energy,
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[
        str,
        Union[torch.Tensor, str, int, float, bool],
    ]:
        image_index = (
            index // self.patches_per_image
        ) % len(self.samples)

        image_path, mask_path = self.samples[image_index]
        
        # Directly extract from RAM memory instead of loading from disk
        image = self.cached_images[image_index]
        mask = self.cached_masks[image_index]

        image, mask = pad_to_minimum_size(
            image,
            mask,
            self.patch_height,
            self.patch_width,
        )

        if self.clahe is not None:
            image = self.clahe(image)

        (
            image_patch,
            mask_patch,
            sample_type,
            vessel_fraction,
            edge_energy,
        ) = self._select_patch(image, mask)

        image_tensor, mask_tensor = apply_transform(
            image_patch,
            mask_patch,
            self.transform,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "source_image_index": image_index,
            "patch_index": index,
            "sample_type": sample_type,
            "vessel_fraction": vessel_fraction,
            "edge_energy": edge_energy,
            "vessel_centred": sample_type == "vessel",
            "hard_negative": sample_type == "hard_negative",
        }


class GridPatchRetinalDataset(Dataset):
    """Deterministic overlapping validation-patch dataset."""

    def __init__(
        self,
        samples: Sequence[SamplePair],
        patch_size: Union[int, Tuple[int, int]] = 256,
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
        if len(self.samples) == 0:
            raise ValueError("GridPatchRetinalDataset received no samples.")

        if isinstance(patch_size, int):
            self.patch_height = patch_size
            self.patch_width = patch_size
        else:
            self.patch_height, self.patch_width = patch_size

        self.model_input_size = int(model_input_size)
        self.stride = int(stride)
        self.include_empty_patches = bool(include_empty_patches)
        self.minimum_vessel_fraction = float(minimum_vessel_fraction)
        self.transform = transform or get_validation_transform(self.model_input_size)
        self.clahe = (
            CLAHEEnhancement(clahe_clip_limit, clahe_tile_grid_size)
            if clahe else None
        )
        
        # RAM Caching: Pre-load all source images and masks into RAM
        self.cached_images = [load_rgb_image(p[0]) for p in self.samples]
        self.cached_masks = [load_binary_mask(p[1]) for p in self.samples]

        self.patch_records: List[Tuple[int, int, int]] = []
        self._build_patch_records()

    def _build_patch_records(self) -> None:
        for sample_index in range(len(self.samples)):
            mask = self.cached_masks[sample_index]
            dummy_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            dummy_image, mask = pad_to_minimum_size(
                dummy_image, mask, self.patch_height, self.patch_width
            )
            del dummy_image

            positions = generate_grid_positions(
                mask.shape[0], mask.shape[1],
                self.patch_height, self.patch_width, self.stride
            )

            for top, left in positions:
                mask_patch = mask[
                    top:top + self.patch_height,
                    left:left + self.patch_width,
                ]
                vessel_fraction = float(mask_patch.mean())
                if self.include_empty_patches or vessel_fraction >= self.minimum_vessel_fraction:
                    self.patch_records.append((sample_index, top, left))

        if len(self.patch_records) == 0:
            raise RuntimeError("No validation patches were generated.")

    def __len__(self) -> int:
        return len(self.patch_records)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, int, float]]:
        sample_index, top, left = self.patch_records[index]
        image_path, mask_path = self.samples[sample_index]

        # Directly extract from RAM memory instead of loading from disk
        image = self.cached_images[sample_index]
        mask = self.cached_masks[sample_index]
        
        image, mask = pad_to_minimum_size(
            image, mask, self.patch_height, self.patch_width
        )

        if self.clahe is not None:
            image = self.clahe(image)

        image_patch, mask_patch = crop_patch(
            image, mask, top, left, self.patch_height, self.patch_width
        )
        vessel_fraction = float(mask_patch.mean())
        image_tensor, mask_tensor = apply_transform(
            image_patch, mask_patch, self.transform
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "source_image_index": sample_index,
            "patch_index": index,
            "top": top,
            "left": left,
            "vessel_fraction": vessel_fraction,
        }


def create_split_indices(
    dataset_size: int,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    if dataset_size < 2:
        raise ValueError("At least two images are required for a split.")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    validation_size = int(dataset_size * validation_ratio)
    validation_size = max(1, validation_size)
    validation_size = min(validation_size, dataset_size - 1)

    generator = torch.Generator()
    generator.manual_seed(seed)
    shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()

    validation_indices = shuffled_indices[:validation_size]
    training_indices = shuffled_indices[validation_size:]
    return training_indices, validation_indices


def split_sample_pairs(
    samples: Sequence[SamplePair],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[SamplePair], List[SamplePair]]:
    training_indices, validation_indices = create_split_indices(
        len(samples), validation_ratio, seed
    )
    training_samples = [samples[index] for index in training_indices]
    validation_samples = [samples[index] for index in validation_indices]
    return training_samples, validation_samples


def create_train_validation_split(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    if len(train_dataset) != len(validation_dataset):
        raise ValueError(
            "Training and validation dataset instances must contain the same samples."
        )
    training_indices, validation_indices = create_split_indices(
        len(train_dataset), validation_ratio, seed
    )
    return Subset(train_dataset, training_indices), Subset(validation_dataset, validation_indices)


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
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    use_persistent_workers = persistent_workers and num_workers > 0

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=use_persistent_workers,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloaders(
    image_dir: PathLike,
    mask_dir: PathLike,
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
    """Original whole-image loaders, retained for ablations."""

    samples = load_sample_pairs(image_dir, mask_dir)
    training_samples, validation_samples = split_sample_pairs(
        samples, validation_ratio, seed
    )

    training_dataset = RetinalVesselDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        samples=training_samples,
        transform=get_train_transform(image_size),
        clahe=clahe,
    )
    validation_dataset = RetinalVesselDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        samples=validation_samples,
        transform=get_validation_transform(image_size),
        clahe=clahe,
    )

    train_loader = create_dataloader(
        training_dataset, batch_size, True, num_workers,
        pin_memory, persistent_workers, drop_last, seed
    )
    validation_loader = create_dataloader(
        validation_dataset, batch_size, False, num_workers,
        pin_memory, persistent_workers, False, seed
    )
    return train_loader, validation_loader


def build_patch_dataloaders(
    image_dir: PathLike,
    mask_dir: PathLike,
    patch_size: int = 256,
    model_input_size: int = 256,
    patches_per_image: int = 100,
    validation_stride: int = 128,
    batch_size: int = 8,
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
    """
    Build patch-based training and deterministic validation loaders.

    The image-level split occurs before patch generation, preventing leakage.
    Suggested first controlled experiment:
      patch_size=256, model_input_size=256, patches_per_image=100,
      vessel_center_probability=0.60, validation_stride=128.
    """

    samples = load_sample_pairs(image_dir, mask_dir)
    training_samples, validation_samples = split_sample_pairs(
        samples, validation_ratio, seed
    )

    training_dataset = RandomPatchRetinalDataset(
        samples=training_samples,
        patch_size=patch_size,
        model_input_size=model_input_size,
        patches_per_image=patches_per_image,
        vessel_center_probability=vessel_center_probability,
        hard_negative_probability=hard_negative_probability,
        random_probability=random_probability,
        minimum_vessel_fraction=minimum_vessel_fraction,
        hard_negative_max_vessel_fraction=(
            hard_negative_max_vessel_fraction
        ),
        hard_negative_candidates=hard_negative_candidates,
        max_sampling_attempts=max_sampling_attempts,
        transform=get_train_transform(model_input_size),
        clahe=clahe,
    )

    validation_dataset = GridPatchRetinalDataset(
        samples=validation_samples,
        patch_size=patch_size,
        model_input_size=model_input_size,
        stride=validation_stride,
        transform=get_validation_transform(model_input_size),
        clahe=clahe,
        include_empty_patches=include_empty_validation_patches,
        minimum_vessel_fraction=0.0,
    )

    train_loader = create_dataloader(
        training_dataset, batch_size, True, num_workers,
        pin_memory, persistent_workers, drop_last, seed
    )
    validation_loader = create_dataloader(
        validation_dataset, batch_size, False, num_workers,
        pin_memory, persistent_workers, False, seed
    )
    return train_loader, validation_loader


def build_all_training_patch_loader(
    image_dir: PathLike,
    mask_dir: PathLike,
    patch_size: int = 256,
    model_input_size: int = 256,
    patches_per_image: int = 100,
    vessel_center_probability: float = 0.60,
    hard_negative_probability: float = 0.25,
    random_probability: float = 0.15,
    minimum_vessel_fraction: float = 0.01,
    hard_negative_max_vessel_fraction: float = 0.005,
    hard_negative_candidates: int = 20,
    max_sampling_attempts: int = 20,
    batch_size: int = 8,
    num_workers: int = 2,
    clahe: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a patch loader from all official DRIVE training images."""

    samples = load_sample_pairs(image_dir, mask_dir)
    dataset = RandomPatchRetinalDataset(
        samples=samples,
        patch_size=patch_size,
        model_input_size=model_input_size,
        patches_per_image=patches_per_image,
        vessel_center_probability=vessel_center_probability,
        hard_negative_probability=hard_negative_probability,
        random_probability=random_probability,
        minimum_vessel_fraction=minimum_vessel_fraction,
        hard_negative_max_vessel_fraction=(
            hard_negative_max_vessel_fraction
        ),
        hard_negative_candidates=hard_negative_candidates,
        max_sampling_attempts=max_sampling_attempts,
        transform=get_train_transform(model_input_size),
        clahe=clahe,
    )
    return create_dataloader(
        dataset, batch_size, True, num_workers,
        pin_memory, persistent_workers, drop_last, seed
    )


def build_test_loader(
    image_dir: PathLike,
    mask_dir: PathLike,
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 2,
    clahe: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = RetinalVesselDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_validation_transform(image_size),
        clahe=clahe,
    )
    return create_dataloader(
        dataset, batch_size, False, num_workers,
        pin_memory, False, False, seed
    )


def build_cross_dataset_loaders(
    datasets_config: Union[Sequence[Dict], Dict[str, Dict]],
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 2,
    clahe: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    if isinstance(datasets_config, dict):
        dataset_entries = [
            {"name": dataset_name, **dataset_config}
            for dataset_name, dataset_config in datasets_config.items()
        ]
    else:
        dataset_entries = list(datasets_config)

    for dataset_config in dataset_entries:
        name = dataset_config["name"]
        print(f"Loading evaluation dataset: {name}")
        loaders[name] = build_test_loader(
            image_dir=dataset_config["image_dir"],
            mask_dir=dataset_config["mask_dir"],
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            clahe=clahe,
            seed=seed,
        )
    return loaders


def dataset_statistics(image_dir: PathLike, mask_dir: PathLike) -> Dict[str, int]:
    samples = load_sample_pairs(image_dir, mask_dir)
    statistics = {
        "samples": len(samples),
        "images": len(get_files(image_dir)),
        "masks": len(get_files(mask_dir)),
    }
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print("Matched samples:", statistics["samples"])
    print("Image files:", statistics["images"])
    print("Mask files:", statistics["masks"])
    print("Image directory:", Path(image_dir).resolve())
    print("Mask directory:", Path(mask_dir).resolve())
    print("=" * 60)
    return statistics


def verify_dataset(
    image_dir: PathLike,
    mask_dir: PathLike,
    image_size: int = 256,
) -> bool:
    print("\nChecking whole-image dataset...")
    try:
        dataset = RetinalVesselDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            transform=get_validation_transform(image_size),
            clahe=True,
        )
        sample = dataset[0]
        print("Dataset OK")
        print("Number of samples:", len(dataset))
        print("Image shape:", sample["image"].shape)
        print("Mask shape:", sample["mask"].shape)
        print("Image dtype:", sample["image"].dtype)
        print("Mask dtype:", sample["mask"].dtype)
        print("Mask values:", torch.unique(sample["mask"]).tolist())
        return True
    except Exception as error:
        print("Dataset Error:")
        print(error)
        return False


def verify_patch_pipeline(
    image_dir: PathLike,
    mask_dir: PathLike,
    patch_size: int = 256,
    model_input_size: int = 256,
    patches_per_image: int = 10,
    validation_stride: int = 128,
    batch_size: int = 4,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> bool:
    print("\nChecking patch pipeline...")
    try:
        train_loader, validation_loader = build_patch_dataloaders(
            image_dir=image_dir,
            mask_dir=mask_dir,
            patch_size=patch_size,
            model_input_size=model_input_size,
            patches_per_image=patches_per_image,
            validation_stride=validation_stride,
            batch_size=batch_size,
            validation_ratio=validation_ratio,
            vessel_center_probability=0.60,
            hard_negative_probability=0.25,
            random_probability=0.15,
            minimum_vessel_fraction=0.01,
            hard_negative_max_vessel_fraction=0.005,
            hard_negative_candidates=20,
            num_workers=0,
            clahe=True,
            seed=seed,
        )
        train_batch = next(iter(train_loader))
        validation_batch = next(iter(validation_loader))
        print("Patch pipeline OK")
        print("Training patch samples:", len(train_loader.dataset))
        print("Validation patch samples:", len(validation_loader.dataset))
        print("Training batch image shape:", train_batch["image"].shape)
        print("Training batch mask shape:", train_batch["mask"].shape)
        print("Validation batch image shape:", validation_batch["image"].shape)
        print("Validation batch mask shape:", validation_batch["mask"].shape)
        print("Training sample types:", train_batch["sample_type"])
        print("Training vessel fractions:", train_batch["vessel_fraction"])
        print("Training edge energies:", train_batch["edge_energy"])
        return True
    except Exception as error:
        print("Patch Pipeline Error:")
        print(error)
        return False


if __name__ == "__main__":
    print("MV-TransUNet Dataset Pipeline Test")

    test_image_dir = Path("./datasets_processed/DRIVE/images")
    test_mask_dir = Path("./datasets_processed/DRIVE/masks")

    if test_image_dir.exists() and test_mask_dir.exists():
        verify_dataset(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            image_size=256,
        )
        verify_patch_pipeline(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            patch_size=256,
            model_input_size=256,
            patches_per_image=10,
            validation_stride=128,
            batch_size=4,
            validation_ratio=0.2,
            seed=42,
        )
    else:
        print("Dataset path does not exist.")
        print("Run prepare_datasets.py first.")