"""
Topology-aware multi-task loss for MV-TransUNetTopology.

The loss supervises:

1. Topology-aware vessel segmentation
2. Soft centerline consistency through clDice
3. Explicit skeleton prediction
4. Endpoint prediction
5. Junction prediction
6. Original decoder deep-supervision outputs
7. Optional original baseline segmentation output

All model inputs must be raw logits. Sigmoid is applied internally.

Expected model output dictionary
--------------------------------
{
    "main_output": Tensor[B, 1, H, W],
    "original_main_output": Tensor[B, 1, H, W],
    "topology_segmentation_logits": Tensor[B, 1, H, W],
    "skeleton_logits": Tensor[B, 1, H, W],
    "endpoint_logits": Tensor[B, 1, H, W],
    "junction_logits": Tensor[B, 1, H, W],
    "auxiliary_outputs": List[Tensor[B, 1, H, W]],
}

Expected target dictionary
--------------------------
{
    "mask": Tensor[B, 1, H, W],
    "skeleton": Tensor[B, 1, H, W],
    "endpoint": Tensor[B, 1, H, W],
    "junction": Tensor[B, 1, H, W],
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# STRUCTURED LOSS OUTPUT
# ============================================================


@dataclass
class TopologyLossOutput:
    """
    Structured loss result.

    total_loss is used for backward propagation. All other fields are
    detached scalar tensors for logging.
    """

    total_loss: torch.Tensor
    segmentation_loss: torch.Tensor
    segmentation_bce_loss: torch.Tensor
    segmentation_dice_loss: torch.Tensor
    cldice_loss: torch.Tensor
    skeleton_loss: torch.Tensor
    skeleton_bce_loss: torch.Tensor
    skeleton_dice_loss: torch.Tensor
    endpoint_loss: torch.Tensor
    endpoint_bce_loss: torch.Tensor
    endpoint_focal_loss: torch.Tensor
    junction_loss: torch.Tensor
    junction_bce_loss: torch.Tensor
    junction_focal_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    baseline_loss: torch.Tensor

    def as_dict(
        self,
        detach: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return individual losses in a logging-friendly dictionary."""

        values = {
            "total_loss": self.total_loss,
            "segmentation_loss": self.segmentation_loss,
            "segmentation_bce_loss": self.segmentation_bce_loss,
            "segmentation_dice_loss": self.segmentation_dice_loss,
            "cldice_loss": self.cldice_loss,
            "skeleton_loss": self.skeleton_loss,
            "skeleton_bce_loss": self.skeleton_bce_loss,
            "skeleton_dice_loss": self.skeleton_dice_loss,
            "endpoint_loss": self.endpoint_loss,
            "endpoint_bce_loss": self.endpoint_bce_loss,
            "endpoint_focal_loss": self.endpoint_focal_loss,
            "junction_loss": self.junction_loss,
            "junction_bce_loss": self.junction_bce_loss,
            "junction_focal_loss": self.junction_focal_loss,
            "auxiliary_loss": self.auxiliary_loss,
            "baseline_loss": self.baseline_loss,
        }

        if detach:
            return {
                name: value.detach()
                for name, value in values.items()
            }

        return values


# ============================================================
# VALIDATION UTILITIES
# ============================================================


def validate_binary_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
) -> None:
    """Validate one prediction or target tensor."""

    if not torch.is_tensor(tensor):
        raise TypeError(
            f"{tensor_name} must be a torch.Tensor."
        )

    if tensor.ndim != 4:
        raise ValueError(
            f"{tensor_name} must have shape [B, C, H, W], "
            f"received {tuple(tensor.shape)}."
        )

    if tensor.shape[1] != 1:
        raise ValueError(
            f"{tensor_name} must contain one channel, "
            f"received {tensor.shape[1]}."
        )

    if not tensor.is_floating_point():
        raise TypeError(
            f"{tensor_name} must use a floating-point dtype."
        )


def prepare_binary_target(
    target: torch.Tensor,
    reference: torch.Tensor,
    target_name: str,
) -> torch.Tensor:
    """
    Convert, resize, and clamp one binary target.

    Nearest-neighbour resizing is used because skeleton and node targets
    must not be blurred by bilinear interpolation.
    """

    if not torch.is_tensor(target):
        raise TypeError(
            f"{target_name} must be a torch.Tensor."
        )

    if target.ndim == 3:
        target = target.unsqueeze(1)

    if target.ndim != 4:
        raise ValueError(
            f"{target_name} must have shape [B, 1, H, W] "
            f"or [B, H, W], received {tuple(target.shape)}."
        )

    if target.shape[1] != 1:
        raise ValueError(
            f"{target_name} must contain one channel."
        )

    target = target.to(
        device=reference.device,
        dtype=reference.dtype,
    )

    if target.shape[-2:] != reference.shape[-2:]:
        target = F.interpolate(
            target,
            size=reference.shape[-2:],
            mode="nearest",
        )

    return target.clamp(
        min=0.0,
        max=1.0,
    )


# ============================================================
# DICE LOSS
# ============================================================


class SoftDiceLoss(nn.Module):
    """Binary Dice loss computed from logits."""

    def __init__(
        self,
        smooth: float = 1.0,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()

        if smooth < 0.0:
            raise ValueError(
                "smooth must be non-negative."
            )

        if epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive."
            )

        self.smooth = smooth
        self.epsilon = epsilon

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate mean soft Dice loss."""

        validate_binary_tensor(
            logits,
            "Dice logits",
        )

        targets = prepare_binary_target(
            target=targets,
            reference=logits,
            target_name="Dice targets",
        )

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(
            start_dim=1
        )
        targets = targets.flatten(
            start_dim=1
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice_score = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
            + self.epsilon
        )

        return (
            1.0 - dice_score
        ).mean()


# ============================================================
# DYNAMIC POSITIVE WEIGHTED BCE
# ============================================================


class DynamicWeightedBCELoss(nn.Module):
    """
    BCE loss with batch-dependent positive-class weighting.

    Endpoint and junction maps are highly sparse. Without positive weighting,
    predicting background everywhere can dominate optimization.

    The positive weight is calculated as:

        negative_pixels / positive_pixels

    and clamped to avoid unstable gradients in nearly empty batches.
    """

    def __init__(
        self,
        minimum_positive_weight: float = 1.0,
        maximum_positive_weight: float = 50.0,
        epsilon: float = 1.0,
    ) -> None:
        super().__init__()

        if minimum_positive_weight <= 0.0:
            raise ValueError(
                "minimum_positive_weight must be positive."
            )

        if (
            maximum_positive_weight
            < minimum_positive_weight
        ):
            raise ValueError(
                "maximum_positive_weight must be greater than or "
                "equal to minimum_positive_weight."
            )

        if epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive."
            )

        self.minimum_positive_weight = (
            minimum_positive_weight
        )
        self.maximum_positive_weight = (
            maximum_positive_weight
        )
        self.epsilon = epsilon

    def calculate_positive_weight(
        self,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate a detached positive-class weight."""

        positive_pixels = targets.sum()
        total_pixels = torch.tensor(
            targets.numel(),
            dtype=targets.dtype,
            device=targets.device,
        )

        negative_pixels = (
            total_pixels - positive_pixels
        )

        positive_weight = (
            negative_pixels
            / (
                positive_pixels
                + self.epsilon
            )
        )

        positive_weight = positive_weight.clamp(
            min=self.minimum_positive_weight,
            max=self.maximum_positive_weight,
        )

        return positive_weight.detach()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate dynamically weighted BCE-with-logits loss."""

        validate_binary_tensor(
            logits,
            "Weighted BCE logits",
        )

        targets = prepare_binary_target(
            target=targets,
            reference=logits,
            target_name="Weighted BCE targets",
        )

        positive_weight = (
            self.calculate_positive_weight(
                targets
            )
        )

        return F.binary_cross_entropy_with_logits(
            input=logits,
            target=targets,
            pos_weight=positive_weight,
        )


# ============================================================
# BINARY FOCAL LOSS
# ============================================================


class BinaryFocalLoss(nn.Module):
    """
    Numerically stable binary focal loss from logits.

    Alpha balances positive and negative examples.
    Gamma focuses learning on difficult pixels.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be in [0, 1]."
            )

        if gamma < 0.0:
            raise ValueError(
                "gamma must be non-negative."
            )

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate binary focal loss."""

        validate_binary_tensor(
            logits,
            "Focal logits",
        )

        targets = prepare_binary_target(
            target=targets,
            reference=logits,
            target_name="Focal targets",
        )

        binary_cross_entropy = (
            F.binary_cross_entropy_with_logits(
                input=logits,
                target=targets,
                reduction="none",
            )
        )

        probabilities = torch.sigmoid(logits)

        probability_of_correct_class = (
            probabilities * targets
            + (1.0 - probabilities)
            * (1.0 - targets)
        )

        alpha_factor = (
            self.alpha * targets
            + (1.0 - self.alpha)
            * (1.0 - targets)
        )

        modulation_factor = (
            1.0
            - probability_of_correct_class
        ).pow(self.gamma)

        focal_loss = (
            alpha_factor
            * modulation_factor
            * binary_cross_entropy
        )

        return focal_loss.mean()


# ============================================================
# SOFT MORPHOLOGICAL SKELETONIZATION
# ============================================================


def soft_erode(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable approximation of binary erosion.

    Horizontal and vertical kernels are combined with a 3x3 erosion.
    """

    erosion_vertical = -F.max_pool2d(
        -image,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0),
    )

    erosion_horizontal = -F.max_pool2d(
        -image,
        kernel_size=(1, 3),
        stride=1,
        padding=(0, 1),
    )

    erosion_square = -F.max_pool2d(
        -image,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    return torch.minimum(
        torch.minimum(
            erosion_vertical,
            erosion_horizontal,
        ),
        erosion_square,
    )


def soft_dilate(
    image: torch.Tensor,
) -> torch.Tensor:
    """Differentiable approximation of binary dilation."""

    return F.max_pool2d(
        image,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def soft_open(
    image: torch.Tensor,
) -> torch.Tensor:
    """Differentiable morphological opening."""

    return soft_dilate(
        soft_erode(image)
    )


def soft_skeletonize(
    image: torch.Tensor,
    iterations: int = 10,
) -> torch.Tensor:
    """
    Approximate a binary skeleton through iterative soft morphology.

    Parameters
    ----------
    image:
        Probability tensor in [0, 1].

    iterations:
        Number of erosion/skeletonization iterations. Ten iterations are
        adequate for 256x256 vessel patches without excessive compute.
    """

    if iterations < 1:
        raise ValueError(
            "iterations must be at least one."
        )

    image = image.clamp(
        min=0.0,
        max=1.0,
    )

    opened_image = soft_open(image)

    skeleton = F.relu(
        image - opened_image
    )

    working_image = image

    for _ in range(iterations - 1):
        working_image = soft_erode(
            working_image
        )

        opened_image = soft_open(
            working_image
        )

        delta = F.relu(
            working_image
            - opened_image
        )

        skeleton = (
            skeleton
            + F.relu(
                delta
                - skeleton * delta
            )
        )

    return skeleton


# ============================================================
# CLDICE LOSS
# ============================================================


class SoftClDiceLoss(nn.Module):
    """
    Differentiable centerline Dice loss.

    Topology precision:
        predicted skeleton inside the target vessel mask.

    Topology sensitivity:
        target skeleton covered by the predicted vessel mask.
    """

    def __init__(
        self,
        iterations: int = 10,
        smooth: float = 1.0,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()

        if iterations < 1:
            raise ValueError(
                "iterations must be at least one."
            )

        self.iterations = iterations
        self.smooth = smooth
        self.epsilon = epsilon

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate soft clDice loss."""

        validate_binary_tensor(
            logits,
            "clDice logits",
        )

        targets = prepare_binary_target(
            target=targets,
            reference=logits,
            target_name="clDice targets",
        )

        probabilities = torch.sigmoid(logits)

        predicted_skeleton = soft_skeletonize(
            probabilities,
            iterations=self.iterations,
        )

        target_skeleton = soft_skeletonize(
            targets,
            iterations=self.iterations,
        )

        topology_precision = (
            (
                predicted_skeleton
                * targets
            ).sum(dim=(1, 2, 3))
            + self.smooth
        ) / (
            predicted_skeleton.sum(
                dim=(1, 2, 3)
            )
            + self.smooth
            + self.epsilon
        )

        topology_sensitivity = (
            (
                target_skeleton
                * probabilities
            ).sum(dim=(1, 2, 3))
            + self.smooth
        ) / (
            target_skeleton.sum(
                dim=(1, 2, 3)
            )
            + self.smooth
            + self.epsilon
        )

        cldice_score = (
            2.0
            * topology_precision
            * topology_sensitivity
        ) / (
            topology_precision
            + topology_sensitivity
            + self.epsilon
        )

        return (
            1.0 - cldice_score
        ).mean()


# ============================================================
# COMPOSITE BINARY LOSSES
# ============================================================


class SegmentationCompositeLoss(nn.Module):
    """BCE plus Dice for the primary vessel mask."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = SoftDiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return total segmentation loss, BCE, and Dice."""

        targets = prepare_binary_target(
            target=targets,
            reference=logits,
            target_name="Segmentation targets",
        )

        bce_loss = (
            F.binary_cross_entropy_with_logits(
                logits,
                targets,
            )
        )

        dice_loss = self.dice_loss(
            logits,
            targets,
        )

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return (
            total_loss,
            bce_loss,
            dice_loss,
        )


class SparseNodeLoss(nn.Module):
    """
    Composite loss for endpoint and junction prediction.

    Dynamic weighted BCE maintains positive-pixel gradients, while focal
    loss suppresses easy background pixels.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        maximum_positive_weight: float = 50.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()

        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

        self.weighted_bce = DynamicWeightedBCELoss(
            minimum_positive_weight=1.0,
            maximum_positive_weight=(
                maximum_positive_weight
            ),
        )

        self.focal_loss = BinaryFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return total node loss, weighted BCE, and focal loss."""

        bce_loss = self.weighted_bce(
            logits,
            targets,
        )

        focal_loss = self.focal_loss(
            logits,
            targets,
        )

        total_loss = (
            self.bce_weight * bce_loss
            + self.focal_weight * focal_loss
        )

        return (
            total_loss,
            bce_loss,
            focal_loss,
        )


# ============================================================
# COMPLETE MULTI-TASK TOPOLOGY LOSS
# ============================================================


class MVTransUNetTopologyLoss(nn.Module):
    """
    Complete multi-task objective for MVTransUNetTopology.

    Initial loss weights are deliberately conservative. Segmentation remains
    the dominant task while auxiliary topology tasks shape the representation.
    """

    REQUIRED_OUTPUT_KEYS = {
        "topology_segmentation_logits",
        "skeleton_logits",
        "endpoint_logits",
        "junction_logits",
    }

    REQUIRED_TARGET_KEYS = {
        "mask",
        "skeleton",
        "endpoint",
        "junction",
    }

    def __init__(
        self,
        segmentation_weight: float = 1.0,
        cldice_weight: float = 0.3,
        skeleton_weight: float = 0.3,
        endpoint_weight: float = 0.1,
        junction_weight: float = 0.1,
        auxiliary_weight: float = 0.2,
        baseline_weight: float = 0.1,
        segmentation_bce_weight: float = 0.5,
        segmentation_dice_weight: float = 0.5,
        skeleton_bce_weight: float = 0.5,
        skeleton_dice_weight: float = 0.5,
        node_bce_weight: float = 0.5,
        node_focal_weight: float = 0.5,
        node_maximum_positive_weight: float = 50.0,
        node_focal_alpha: float = 0.75,
        node_focal_gamma: float = 2.0,
        cldice_iterations: int = 10,
        auxiliary_stage_weights: tuple[
            float,
            float,
            float,
        ] = (
            0.25,
            0.50,
            0.75,
        ),
    ) -> None:
        super().__init__()

        scalar_weights = {
            "segmentation_weight": segmentation_weight,
            "cldice_weight": cldice_weight,
            "skeleton_weight": skeleton_weight,
            "endpoint_weight": endpoint_weight,
            "junction_weight": junction_weight,
            "auxiliary_weight": auxiliary_weight,
            "baseline_weight": baseline_weight,
        }

        for name, value in scalar_weights.items():
            if value < 0.0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        if len(auxiliary_stage_weights) != 3:
            raise ValueError(
                "auxiliary_stage_weights must contain exactly "
                "three values."
            )

        if any(
            weight < 0.0
            for weight in auxiliary_stage_weights
        ):
            raise ValueError(
                "Auxiliary stage weights must be non-negative."
            )

        self.segmentation_weight = segmentation_weight
        self.cldice_weight = cldice_weight
        self.skeleton_weight = skeleton_weight
        self.endpoint_weight = endpoint_weight
        self.junction_weight = junction_weight
        self.auxiliary_weight = auxiliary_weight
        self.baseline_weight = baseline_weight

        self.auxiliary_stage_weights = (
            auxiliary_stage_weights
        )

        self.segmentation_loss = (
            SegmentationCompositeLoss(
                bce_weight=(
                    segmentation_bce_weight
                ),
                dice_weight=(
                    segmentation_dice_weight
                ),
            )
        )

        self.cldice_loss = SoftClDiceLoss(
            iterations=cldice_iterations,
        )

        self.skeleton_loss = (
            SegmentationCompositeLoss(
                bce_weight=skeleton_bce_weight,
                dice_weight=skeleton_dice_weight,
            )
        )

        self.endpoint_loss = SparseNodeLoss(
            bce_weight=node_bce_weight,
            focal_weight=node_focal_weight,
            maximum_positive_weight=(
                node_maximum_positive_weight
            ),
            focal_alpha=node_focal_alpha,
            focal_gamma=node_focal_gamma,
        )

        self.junction_loss = SparseNodeLoss(
            bce_weight=node_bce_weight,
            focal_weight=node_focal_weight,
            maximum_positive_weight=(
                node_maximum_positive_weight
            ),
            focal_alpha=node_focal_alpha,
            focal_gamma=node_focal_gamma,
        )

    @staticmethod
    def _validate_mapping_keys(
        mapping: Mapping,
        required_keys: set[str],
        mapping_name: str,
    ) -> None:
        """Check required dictionary entries."""

        missing_keys = (
            required_keys
            - set(mapping.keys())
        )

        if missing_keys:
            raise KeyError(
                f"{mapping_name} is missing required keys: "
                f"{sorted(missing_keys)}"
            )

    def _calculate_auxiliary_loss(
        self,
        auxiliary_outputs: object,
        mask_target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate weighted deep-supervision segmentation loss."""

        if self.auxiliary_weight == 0.0:
            return mask_target.new_zeros(())

        if auxiliary_outputs is None:
            return mask_target.new_zeros(())

        if not isinstance(
            auxiliary_outputs,
            (list, tuple),
        ):
            raise TypeError(
                "auxiliary_outputs must be a list or tuple."
            )

        if len(auxiliary_outputs) != 3:
            raise ValueError(
                "Expected exactly three auxiliary outputs."
            )

        weighted_loss = mask_target.new_zeros(())
        weight_sum = 0.0

        for stage_weight, auxiliary_logits in zip(
            self.auxiliary_stage_weights,
            auxiliary_outputs,
        ):
            auxiliary_segmentation_loss, _, _ = (
                self.segmentation_loss(
                    auxiliary_logits,
                    mask_target,
                )
            )

            weighted_loss = (
                weighted_loss
                + stage_weight
                * auxiliary_segmentation_loss
            )

            weight_sum += stage_weight

        if weight_sum <= 0.0:
            return mask_target.new_zeros(())

        return weighted_loss / weight_sum

    def forward(
        self,
        outputs: Mapping[str, object],
        targets: Mapping[str, torch.Tensor],
    ) -> TopologyLossOutput:
        """
        Calculate the complete topology-aware loss.

        Returns a structured object rather than only one scalar so that
        every component can be logged independently.
        """

        if not isinstance(outputs, Mapping):
            raise TypeError(
                "outputs must be a dictionary-like mapping."
            )

        if not isinstance(targets, Mapping):
            raise TypeError(
                "targets must be a dictionary-like mapping."
            )

        self._validate_mapping_keys(
            mapping=outputs,
            required_keys=self.REQUIRED_OUTPUT_KEYS,
            mapping_name="Model outputs",
        )

        self._validate_mapping_keys(
            mapping=targets,
            required_keys=self.REQUIRED_TARGET_KEYS,
            mapping_name="Targets",
        )

        segmentation_logits = outputs[
            "topology_segmentation_logits"
        ]
        skeleton_logits = outputs[
            "skeleton_logits"
        ]
        endpoint_logits = outputs[
            "endpoint_logits"
        ]
        junction_logits = outputs[
            "junction_logits"
        ]

        if not torch.is_tensor(
            segmentation_logits
        ):
            raise TypeError(
                "topology_segmentation_logits must be a tensor."
            )

        mask_target = prepare_binary_target(
            target=targets["mask"],
            reference=segmentation_logits,
            target_name="Mask target",
        )

        skeleton_target = prepare_binary_target(
            target=targets["skeleton"],
            reference=skeleton_logits,
            target_name="Skeleton target",
        )

        endpoint_target = prepare_binary_target(
            target=targets["endpoint"],
            reference=endpoint_logits,
            target_name="Endpoint target",
        )

        junction_target = prepare_binary_target(
            target=targets["junction"],
            reference=junction_logits,
            target_name="Junction target",
        )

        (
            segmentation_loss,
            segmentation_bce_loss,
            segmentation_dice_loss,
        ) = self.segmentation_loss(
            segmentation_logits,
            mask_target,
        )

        cldice_loss = self.cldice_loss(
            segmentation_logits,
            mask_target,
        )

        (
            skeleton_loss,
            skeleton_bce_loss,
            skeleton_dice_loss,
        ) = self.skeleton_loss(
            skeleton_logits,
            skeleton_target,
        )

        (
            endpoint_loss,
            endpoint_bce_loss,
            endpoint_focal_loss,
        ) = self.endpoint_loss(
            endpoint_logits,
            endpoint_target,
        )

        (
            junction_loss,
            junction_bce_loss,
            junction_focal_loss,
        ) = self.junction_loss(
            junction_logits,
            junction_target,
        )

        auxiliary_loss = (
            self._calculate_auxiliary_loss(
                auxiliary_outputs=outputs.get(
                    "auxiliary_outputs"
                ),
                mask_target=mask_target,
            )
        )

        baseline_loss = mask_target.new_zeros(())

        original_main_output = outputs.get(
            "original_main_output"
        )

        if (
            self.baseline_weight > 0.0
            and original_main_output is not None
        ):
            if not torch.is_tensor(
                original_main_output
            ):
                raise TypeError(
                    "original_main_output must be a tensor."
                )

            baseline_loss, _, _ = (
                self.segmentation_loss(
                    original_main_output,
                    mask_target,
                )
            )

        total_loss = (
            self.segmentation_weight
            * segmentation_loss
            + self.cldice_weight
            * cldice_loss
            + self.skeleton_weight
            * skeleton_loss
            + self.endpoint_weight
            * endpoint_loss
            + self.junction_weight
            * junction_loss
            + self.auxiliary_weight
            * auxiliary_loss
            + self.baseline_weight
            * baseline_loss
        )

        return TopologyLossOutput(
            total_loss=total_loss,
            segmentation_loss=segmentation_loss,
            segmentation_bce_loss=(
                segmentation_bce_loss
            ),
            segmentation_dice_loss=(
                segmentation_dice_loss
            ),
            cldice_loss=cldice_loss,
            skeleton_loss=skeleton_loss,
            skeleton_bce_loss=skeleton_bce_loss,
            skeleton_dice_loss=skeleton_dice_loss,
            endpoint_loss=endpoint_loss,
            endpoint_bce_loss=endpoint_bce_loss,
            endpoint_focal_loss=endpoint_focal_loss,
            junction_loss=junction_loss,
            junction_bce_loss=junction_bce_loss,
            junction_focal_loss=junction_focal_loss,
            auxiliary_loss=auxiliary_loss,
            baseline_loss=baseline_loss,
        )


# ============================================================
# SMOKE-TEST UTILITIES
# ============================================================


def draw_line_target(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create simple artificial vessel topology targets."""

    mask = torch.zeros(
        batch_size,
        1,
        height,
        width,
        device=device,
    )

    skeleton = torch.zeros_like(mask)
    endpoint = torch.zeros_like(mask)
    junction = torch.zeros_like(mask)

    center_y = height // 2
    left_x = width // 4
    right_x = 3 * width // 4
    junction_x = width // 2
    branch_top = height // 4

    mask[
        :,
        :,
        center_y - 2:center_y + 3,
        left_x:right_x + 1,
    ] = 1.0

    mask[
        :,
        :,
        branch_top:center_y + 1,
        junction_x - 2:junction_x + 3,
    ] = 1.0

    skeleton[
        :,
        :,
        center_y,
        left_x:right_x + 1,
    ] = 1.0

    skeleton[
        :,
        :,
        branch_top:center_y + 1,
        junction_x,
    ] = 1.0

    endpoint[
        :,
        :,
        center_y,
        left_x,
    ] = 1.0

    endpoint[
        :,
        :,
        center_y,
        right_x,
    ] = 1.0

    endpoint[
        :,
        :,
        branch_top,
        junction_x,
    ] = 1.0

    junction[
        :,
        :,
        center_y,
        junction_x,
    ] = 1.0

    return {
        "mask": mask,
        "skeleton": skeleton,
        "endpoint": endpoint,
        "junction": junction,
    }


def run_loss_smoke_test() -> None:
    """Test forward calculation, numerical stability, and gradients."""

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    batch_size = 2
    height = 256
    width = 256

    criterion = MVTransUNetTopologyLoss(
        segmentation_weight=1.0,
        cldice_weight=0.3,
        skeleton_weight=0.3,
        endpoint_weight=0.1,
        junction_weight=0.1,
        auxiliary_weight=0.2,
        baseline_weight=0.1,
        cldice_iterations=10,
    ).to(device)

    def random_logits() -> torch.Tensor:
        return torch.randn(
            batch_size,
            1,
            height,
            width,
            device=device,
            requires_grad=True,
        )

    outputs = {
        "main_output": random_logits(),
        "topology_segmentation_logits": (
            random_logits()
        ),
        "original_main_output": random_logits(),
        "skeleton_logits": random_logits(),
        "endpoint_logits": random_logits(),
        "junction_logits": random_logits(),
        "auxiliary_outputs": [
            random_logits(),
            random_logits(),
            random_logits(),
        ],
    }

    targets = draw_line_target(
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
    )

    loss_output = criterion(
        outputs=outputs,
        targets=targets,
    )

    if not torch.isfinite(
        loss_output.total_loss
    ):
        raise RuntimeError(
            "Total loss is NaN or infinite."
        )

    loss_output.total_loss.backward()

    prediction_keys = [
        "topology_segmentation_logits",
        "original_main_output",
        "skeleton_logits",
        "endpoint_logits",
        "junction_logits",
    ]

    for prediction_key in prediction_keys:
        prediction = outputs[
            prediction_key
        ]

        if prediction.grad is None:
            raise RuntimeError(
                f"No gradient reached {prediction_key}."
            )

        if not torch.isfinite(
            prediction.grad
        ).all():
            raise RuntimeError(
                f"Non-finite gradient found in "
                f"{prediction_key}."
            )

    for index, auxiliary_output in enumerate(
        outputs["auxiliary_outputs"],
        start=1,
    ):
        if auxiliary_output.grad is None:
            raise RuntimeError(
                f"No gradient reached auxiliary output "
                f"{index}."
            )

    print("=" * 72)
    print("MV-TransUNet Topology Loss Smoke Test")
    print("=" * 72)
    print("Device:", device)
    print()

    for name, value in (
        loss_output.as_dict(
            detach=True
        ).items()
    ):
        print(
            f"{name:32s}",
            f"{value.item():.6f}",
        )

    print()
    print("=" * 72)
    print("Finite loss values: PASSED")
    print("Gradient propagation: PASSED")
    print("Sparse node supervision: PASSED")
    print("Soft clDice calculation: PASSED")
    print("=" * 72)


if __name__ == "__main__":
    run_loss_smoke_test()