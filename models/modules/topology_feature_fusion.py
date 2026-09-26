"""
Topology Feature Fusion Module for MV-TransUNet++.

The module receives the final decoder feature map and produces:

1. A refined shared topology-aware feature representation
2. A vessel segmentation logit map
3. A skeleton logit map
4. An endpoint logit map
5. A junction logit map

The prediction heads remain lightweight. Most representation learning occurs
inside the shared Topology Feature Fusion Module (TFFM).

Important
---------
All returned maps are logits. Do not apply sigmoid inside the module during
training because PyTorch losses such as BCEWithLogitsLoss expect logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# OUTPUT STRUCTURE
# ============================================================


@dataclass
class TopologyHeadOutput:
    """
    Structured output of the topology-aware prediction module.

    Attributes
    ----------
    segmentation_logits:
        Binary vessel-segmentation logits with shape [B, 1, H, W].

    skeleton_logits:
        Vessel-centerline logits with shape [B, 1, H, W].

    endpoint_logits:
        Vessel endpoint logits with shape [B, 1, H, W].

    junction_logits:
        Vessel junction logits with shape [B, 1, H, W].

    shared_topology_features:
        Refined decoder representation shared by every prediction head.
    """

    segmentation_logits: torch.Tensor
    skeleton_logits: torch.Tensor
    endpoint_logits: torch.Tensor
    junction_logits: torch.Tensor
    shared_topology_features: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return the structured outputs as a dictionary."""

        return {
            "segmentation_logits": self.segmentation_logits,
            "skeleton_logits": self.skeleton_logits,
            "endpoint_logits": self.endpoint_logits,
            "junction_logits": self.junction_logits,
            "shared_topology_features": (
                self.shared_topology_features
            ),
        }


# ============================================================
# NORMALIZATION
# ============================================================


def build_group_norm(
    channels: int,
    maximum_groups: int = 8,
) -> nn.GroupNorm:
    """
    Build GroupNorm with a valid group count.

    GroupNorm is preferred over BatchNorm for retinal-vessel segmentation
    because medical-image models commonly train with small batch sizes.
    """

    if channels <= 0:
        raise ValueError(
            f"channels must be positive, received {channels}."
        )

    group_count = min(
        maximum_groups,
        channels,
    )

    while channels % group_count != 0:
        group_count -= 1

    return nn.GroupNorm(
        num_groups=group_count,
        num_channels=channels,
    )


# ============================================================
# BASIC CONVOLUTION BLOCK
# ============================================================


class ConvNormActivation(nn.Module):
    """
    Convolution followed by GroupNorm and GELU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd to preserve spatial dimensions."
            )

        padding = (
            dilation * (kernel_size - 1) // 2
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            build_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""

        return self.block(inputs)


# ============================================================
# MULTI-SCALE TOPOLOGY CONTEXT
# ============================================================


class MultiScaleTopologyContext(nn.Module):
    """
    Extract vessel context at multiple receptive-field scales.

    Retinal vessels contain:

    - small capillaries requiring local detail;
    - medium branches requiring neighbourhood context;
    - major vessels requiring wider spatial context.

    Parallel dilated convolutions allow the topology module to examine all
    three scales without reducing spatial resolution.
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
    ) -> None:
        super().__init__()

        if branch_channels <= 0:
            raise ValueError(
                "branch_channels must be positive."
            )

        self.local_branch = ConvNormActivation(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=3,
            dilation=1,
        )

        self.medium_branch = ConvNormActivation(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=3,
            dilation=2,
        )

        self.wide_branch = ConvNormActivation(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=3,
            dilation=4,
        )

        self.pointwise_branch = ConvNormActivation(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=1,
            dilation=1,
        )

    @property
    def output_channels(self) -> int:
        """Number of channels produced after branch concatenation."""

        return (
            self.local_branch.block[0].out_channels
            * 4
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Extract and concatenate multi-scale topology features."""

        local_features = self.local_branch(inputs)
        medium_features = self.medium_branch(inputs)
        wide_features = self.wide_branch(inputs)
        pointwise_features = self.pointwise_branch(inputs)

        return torch.cat(
            [
                local_features,
                medium_features,
                wide_features,
                pointwise_features,
            ],
            dim=1,
        )


# ============================================================
# CHANNEL ATTENTION
# ============================================================


class ChannelTopologyAttention(nn.Module):
    """
    Channel attention for selecting topology-relevant feature channels.

    This is a lightweight squeeze-and-excitation mechanism. It allows the
    network to increase the contribution of channels useful for vessel
    continuity, branching, and centerline representation.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()

        hidden_channels = max(
            channels // reduction_ratio,
            8,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.attention = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Reweight channels according to global topology context."""

        channel_weights = self.attention(
            self.pool(inputs)
        )

        return inputs * channel_weights


# ============================================================
# SPATIAL ATTENTION
# ============================================================


class SpatialTopologyAttention(nn.Module):
    """
    Spatial attention for emphasizing vessel-like locations.

    Average and maximum channel projections are combined to form a
    spatial importance map.
    """

    def __init__(
        self,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        if kernel_size not in (3, 5, 7):
            raise ValueError(
                "Spatial attention kernel_size must be 3, 5, or 7."
            )

        padding = kernel_size // 2

        self.attention = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Reweight spatial locations."""

        average_projection = torch.mean(
            inputs,
            dim=1,
            keepdim=True,
        )

        maximum_projection, _ = torch.max(
            inputs,
            dim=1,
            keepdim=True,
        )

        spatial_descriptor = torch.cat(
            [
                average_projection,
                maximum_projection,
            ],
            dim=1,
        )

        spatial_weights = self.attention(
            spatial_descriptor
        )

        return inputs * spatial_weights


# ============================================================
# RESIDUAL TOPOLOGY REFINEMENT
# ============================================================


class ResidualTopologyBlock(nn.Module):
    """
    Residual feature-refinement block.

    Depthwise convolution captures local vessel geometry at low cost, while
    pointwise convolution mixes semantic channels.
    """

    def __init__(
        self,
        channels: int,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "dropout_probability must be in [0, 1)."
            )

        self.depthwise = ConvNormActivation(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            dilation=1,
            groups=channels,
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=False,
            ),
            build_group_norm(channels),
            nn.GELU(),
            nn.Dropout2d(
                p=dropout_probability
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply residual topology refinement."""

        residual = inputs

        features = self.depthwise(inputs)
        features = self.pointwise(features)

        return features + residual


# ============================================================
# TOPOLOGY FEATURE FUSION MODULE
# ============================================================


class TopologyFeatureFusionModule(nn.Module):
    """
    Build a shared topology-aware representation from decoder features.

    Processing sequence
    -------------------
    Decoder features
        -> channel projection
        -> multi-scale spatial context
        -> feature fusion
        -> channel attention
        -> spatial attention
        -> residual topology refinement
        -> shared topology representation

    The output is shared by all prediction heads.
    """

    def __init__(
        self,
        in_channels: int,
        topology_channels: int = 128,
        branch_channels: int = 32,
        number_of_refinement_blocks: int = 2,
        dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive."
            )

        if topology_channels <= 0:
            raise ValueError(
                "topology_channels must be positive."
            )

        if branch_channels <= 0:
            raise ValueError(
                "branch_channels must be positive."
            )

        if number_of_refinement_blocks < 1:
            raise ValueError(
                "number_of_refinement_blocks must be at least 1."
            )

        self.input_projection = ConvNormActivation(
            in_channels=in_channels,
            out_channels=topology_channels,
            kernel_size=1,
        )

        self.multi_scale_context = MultiScaleTopologyContext(
            in_channels=topology_channels,
            branch_channels=branch_channels,
        )

        multi_scale_channels = (
            self.multi_scale_context.output_channels
        )

        self.context_fusion = ConvNormActivation(
            in_channels=(
                topology_channels
                + multi_scale_channels
            ),
            out_channels=topology_channels,
            kernel_size=1,
        )

        self.channel_attention = ChannelTopologyAttention(
            channels=topology_channels,
            reduction_ratio=8,
        )

        self.spatial_attention = SpatialTopologyAttention(
            kernel_size=7,
        )

        self.refinement_blocks = nn.Sequential(
            *[
                ResidualTopologyBlock(
                    channels=topology_channels,
                    dropout_probability=(
                        dropout_probability
                    ),
                )
                for _ in range(
                    number_of_refinement_blocks
                )
            ]
        )

        self.output_normalization = build_group_norm(
            topology_channels
        )

        self.output_activation = nn.GELU()

    def forward(
        self,
        decoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce shared topology-aware decoder features.

        Parameters
        ----------
        decoder_features:
            Tensor with shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Shared topology representation with shape
            [B, topology_channels, H, W].
        """

        if decoder_features.ndim != 4:
            raise ValueError(
                "decoder_features must have shape [B, C, H, W], "
                f"received {tuple(decoder_features.shape)}."
            )

        projected_features = self.input_projection(
            decoder_features
        )

        multi_scale_features = self.multi_scale_context(
            projected_features
        )

        fused_features = self.context_fusion(
            torch.cat(
                [
                    projected_features,
                    multi_scale_features,
                ],
                dim=1,
            )
        )

        attended_features = self.channel_attention(
            fused_features
        )

        attended_features = self.spatial_attention(
            attended_features
        )

        # Preserve the original projected decoder representation so that
        # topology refinement cannot erase useful segmentation information.
        shared_features = (
            attended_features
            + projected_features
        )

        shared_features = self.refinement_blocks(
            shared_features
        )

        shared_features = self.output_normalization(
            shared_features
        )

        return self.output_activation(
            shared_features
        )


# ============================================================
# TASK-SPECIFIC PREDICTION HEAD
# ============================================================


class TopologyPredictionHead(nn.Module):
    """
    Lightweight task-specific prediction head.

    Each head receives the same shared topology features but learns a
    task-specific transformation before producing one binary logit map.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive."
            )

        self.head = nn.Sequential(
            ConvNormActivation(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
            ),
            nn.Dropout2d(
                p=dropout_probability
            ),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Produce one binary prediction-logit map."""

        return self.head(features)


# ============================================================
# COMPLETE TOPOLOGY-AWARE OUTPUT MODULE
# ============================================================


class TopologyAwarePredictionModule(nn.Module):
    """
    Complete TFFM with segmentation and topology prediction heads.

    This module is intended to replace the current final segmentation head
    in MV-TransUNet++ while preserving the existing encoder, transformer,
    skip connections, and decoder.
    """

    def __init__(
        self,
        decoder_channels: int,
        topology_channels: int = 128,
        branch_channels: int = 32,
        head_hidden_channels: int = 64,
        number_of_refinement_blocks: int = 2,
        fusion_dropout_probability: float = 0.1,
        head_dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        self.topology_fusion = TopologyFeatureFusionModule(
            in_channels=decoder_channels,
            topology_channels=topology_channels,
            branch_channels=branch_channels,
            number_of_refinement_blocks=(
                number_of_refinement_blocks
            ),
            dropout_probability=(
                fusion_dropout_probability
            ),
        )

        self.segmentation_head = TopologyPredictionHead(
            in_channels=topology_channels,
            hidden_channels=head_hidden_channels,
            dropout_probability=(
                head_dropout_probability
            ),
        )

        self.skeleton_head = TopologyPredictionHead(
            in_channels=topology_channels,
            hidden_channels=head_hidden_channels,
            dropout_probability=(
                head_dropout_probability
            ),
        )

        self.endpoint_head = TopologyPredictionHead(
            in_channels=topology_channels,
            hidden_channels=head_hidden_channels,
            dropout_probability=(
                head_dropout_probability
            ),
        )

        self.junction_head = TopologyPredictionHead(
            in_channels=topology_channels,
            hidden_channels=head_hidden_channels,
            dropout_probability=(
                head_dropout_probability
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize newly introduced convolution and normalization layers.

        Kaiming initialization is suitable for the convolutional feature
        transformations, while prediction biases begin at zero.
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        decoder_features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> TopologyHeadOutput:
        """
        Produce segmentation and topology logits.

        Parameters
        ----------
        decoder_features:
            Final decoder feature map with shape [B, C, H, W].

        output_size:
            Optional final spatial size. Use this when the decoder output
            resolution differs from the original input resolution.

        Returns
        -------
        TopologyHeadOutput
            Structured prediction logits and shared features.
        """

        shared_features = self.topology_fusion(
            decoder_features
        )

        segmentation_logits = self.segmentation_head(
            shared_features
        )

        skeleton_logits = self.skeleton_head(
            shared_features
        )

        endpoint_logits = self.endpoint_head(
            shared_features
        )

        junction_logits = self.junction_head(
            shared_features
        )

        if output_size is not None:
            segmentation_logits = F.interpolate(
                segmentation_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

            skeleton_logits = F.interpolate(
                skeleton_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

            endpoint_logits = F.interpolate(
                endpoint_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

            junction_logits = F.interpolate(
                junction_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return TopologyHeadOutput(
            segmentation_logits=segmentation_logits,
            skeleton_logits=skeleton_logits,
            endpoint_logits=endpoint_logits,
            junction_logits=junction_logits,
            shared_topology_features=shared_features,
        )


# ============================================================
# SMOKE TEST
# ============================================================


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Count trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def run_smoke_test() -> None:
    """Run a standalone shape and gradient smoke test."""

    torch.manual_seed(42)

    batch_size = 2
    decoder_channels = 64
    decoder_height = 128
    decoder_width = 128

    model = TopologyAwarePredictionModule(
        decoder_channels=decoder_channels,
        topology_channels=128,
        branch_channels=32,
        head_hidden_channels=64,
        number_of_refinement_blocks=2,
        fusion_dropout_probability=0.1,
        head_dropout_probability=0.0,
    )

    decoder_features = torch.randn(
        batch_size,
        decoder_channels,
        decoder_height,
        decoder_width,
        requires_grad=True,
    )

    outputs = model(
        decoder_features=decoder_features,
        output_size=(256, 256),
    )

    expected_prediction_shape = (
        batch_size,
        1,
        256,
        256,
    )

    expected_feature_shape = (
        batch_size,
        128,
        decoder_height,
        decoder_width,
    )

    assert (
        outputs.segmentation_logits.shape
        == expected_prediction_shape
    )

    assert (
        outputs.skeleton_logits.shape
        == expected_prediction_shape
    )

    assert (
        outputs.endpoint_logits.shape
        == expected_prediction_shape
    )

    assert (
        outputs.junction_logits.shape
        == expected_prediction_shape
    )

    assert (
        outputs.shared_topology_features.shape
        == expected_feature_shape
    )

    test_loss = (
        outputs.segmentation_logits.mean()
        + outputs.skeleton_logits.mean()
        + outputs.endpoint_logits.mean()
        + outputs.junction_logits.mean()
    )

    test_loss.backward()

    if decoder_features.grad is None:
        raise RuntimeError(
            "Gradient did not propagate to decoder features."
        )

    print("=" * 70)
    print("TopologyAwarePredictionModule smoke test passed")
    print("=" * 70)
    print(
        "Input decoder features:  ",
        tuple(decoder_features.shape),
    )
    print(
        "Segmentation logits:      ",
        tuple(outputs.segmentation_logits.shape),
    )
    print(
        "Skeleton logits:          ",
        tuple(outputs.skeleton_logits.shape),
    )
    print(
        "Endpoint logits:          ",
        tuple(outputs.endpoint_logits.shape),
    )
    print(
        "Junction logits:          ",
        tuple(outputs.junction_logits.shape),
    )
    print(
        "Shared topology features: ",
        tuple(
            outputs.shared_topology_features.shape
        ),
    )
    print(
        "Trainable parameters:     ",
        f"{count_trainable_parameters(model):,}",
    )
    print(
        "Gradient propagation:      PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()