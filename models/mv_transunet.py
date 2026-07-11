"""
MV-TransUNet Full Architecture

Boundary-Aware Hybrid CNN-Vision Transformer
for Retinal Vessel Segmentation

Components:
1. ResNet50 + Vision Transformer encoder
2. Vessel Attention Module
3. Multi-scale progressive decoder
4. Optional deep supervision

Input:
    Tensor with shape:
    B x 3 x H x W

Training output with deep supervision enabled:
    {
        "main_output": Tensor[B, 1, H, W],
        "auxiliary_outputs": [
            Tensor[B, 1, H, W],
            Tensor[B, 1, H, W],
            Tensor[B, 1, H, W]
        ],
        "decoder_features": [
            Tensor,
            Tensor,
            Tensor,
            Tensor
        ]
    }

Validation and inference output:
    Tensor with shape:
    B x 1 x H x W
"""


from typing import Dict, List, Union

import torch
import torch.nn as nn

from models.backbones.transunet_base import TransUNetBackbone
from models.modules.decoder import MultiScaleDecoder
from models.modules.vessel_attention import VesselAttentionModule


# ============================================================
# TYPE DEFINITIONS
# ============================================================

ModelOutput = Union[
    torch.Tensor,
    Dict[
        str,
        Union[
            torch.Tensor,
            List[torch.Tensor],
        ],
    ],
]


# ============================================================
# MV-TRANSUNET MODEL
# ============================================================


class MVTransUNet(nn.Module):
    """
    Boundary-aware hybrid CNN-Vision Transformer model.

    The architecture combines:
    - ResNet50 multi-scale CNN features;
    - Vision Transformer global-context features;
    - vessel attention;
    - progressive skip-connected decoding;
    - optional deep supervision.

    When deep supervision is enabled:

    Training mode:
        Returns a dictionary containing the main segmentation output
        and three auxiliary segmentation outputs.

    Evaluation mode:
        Returns only the main segmentation output unless
        return_auxiliary=True is explicitly supplied.
    """

    def __init__(
        self,
        pretrained: bool = True,
        transformer_channels: int = 768,
        vessel_reduction_ratio: int = 16,
        output_channels: int = 1,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()

        if transformer_channels <= 0:
            raise ValueError(
                "transformer_channels must be greater than zero."
            )

        if vessel_reduction_ratio <= 0:
            raise ValueError(
                "vessel_reduction_ratio must be greater than zero."
            )

        if output_channels <= 0:
            raise ValueError(
                "output_channels must be greater than zero."
            )

        self.pretrained = pretrained
        self.transformer_channels = transformer_channels
        self.vessel_reduction_ratio = vessel_reduction_ratio
        self.output_channels = output_channels
        self.deep_supervision = deep_supervision

        # ----------------------------------------------------
        # Hybrid ResNet50 + Vision Transformer encoder
        # ----------------------------------------------------

        self.encoder = TransUNetBackbone(
            pretrained=pretrained,
            hidden_dim=transformer_channels,
        )

        # ----------------------------------------------------
        # Vessel Attention Module
        # ----------------------------------------------------

        self.vessel_attention = VesselAttentionModule(
            channels=transformer_channels,
            reduction_ratio=vessel_reduction_ratio,
        )

        # ----------------------------------------------------
        # Multi-scale progressive decoder
        # ----------------------------------------------------

        self.decoder = MultiScaleDecoder(
            output_channels=output_channels,
            deep_supervision=deep_supervision,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> ModelOutput:
        """
        Run the complete MV-TransUNet forward pass.

        Args:
            x:
                Input retinal image tensor with shape
                [B, 3, H, W].

            return_auxiliary:
                Force auxiliary outputs to be returned when deep
                supervision is enabled. This can be used for model
                analysis during evaluation.

        Returns:
            Training with deep supervision:
                Dictionary with main output, auxiliary outputs,
                and decoder features.

            Evaluation or deep supervision disabled:
                Final segmentation logits tensor.
        """

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape [batch, channels, height, width]."
            )

        if x.shape[1] != 3:
            raise ValueError(
                f"MV-TransUNet expects 3 input channels, "
                f"but received {x.shape[1]}."
            )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        features = self.encoder(x)

        required_features = {
            "transformer",
            "skip1",
            "skip2",
            "skip3",
        }

        missing_features = (
            required_features
            - set(features.keys())
        )

        if missing_features:
            raise KeyError(
                "Encoder output is missing required features: "
                f"{sorted(missing_features)}"
            )

        transformer_feature = features[
            "transformer"
        ]

        skip1 = features[
            "skip1"
        ]

        skip2 = features[
            "skip2"
        ]

        skip3 = features[
            "skip3"
        ]

        # ----------------------------------------------------
        # Vessel attention
        # ----------------------------------------------------

        transformer_feature = self.vessel_attention(
            transformer_feature
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        output = self.decoder(
            transformer_feature=transformer_feature,
            skip1=skip1,
            skip2=skip2,
            skip3=skip3,
            return_auxiliary=return_auxiliary,
        )

        return output


# ============================================================
# MODEL BUILDER
# ============================================================


def build_mv_transunet(
    pretrained: bool = True,
    transformer_channels: int = 768,
    vessel_reduction_ratio: int = 16,
    output_channels: int = 1,
    deep_supervision: bool = True,
) -> MVTransUNet:
    """
    Build an MV-TransUNet model instance.
    """

    return MVTransUNet(
        pretrained=pretrained,
        transformer_channels=transformer_channels,
        vessel_reduction_ratio=vessel_reduction_ratio,
        output_channels=output_channels,
        deep_supervision=deep_supervision,
    )


# ============================================================
# PARAMETER COUNT
# ============================================================


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# MODEL TEST
# ============================================================


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print("=" * 70)
    print("MV-TransUNet Deep Supervision Test")
    print("=" * 70)

    print("Device:", device)

    model = MVTransUNet(
        pretrained=False,
        transformer_channels=768,
        vessel_reduction_ratio=16,
        output_channels=1,
        deep_supervision=True,
    ).to(device)

    input_tensor = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    # --------------------------------------------------------
    # Training-mode test
    # --------------------------------------------------------

    model.train()

    training_output = model(
        input_tensor
    )

    print()
    print("=" * 70)
    print("Training Mode")
    print("=" * 70)

    if not isinstance(
        training_output,
        dict,
    ):
        raise RuntimeError(
            "Deep-supervision training output must be a dictionary."
        )

    main_output = training_output[
        "main_output"
    ]

    auxiliary_outputs = training_output[
        "auxiliary_outputs"
    ]

    print(
        "Input shape:",
        input_tensor.shape,
    )

    print(
        "Main output shape:",
        main_output.shape,
    )

    for index, auxiliary_output in enumerate(
        auxiliary_outputs,
        start=1,
    ):
        print(
            f"Auxiliary output {index} shape:",
            auxiliary_output.shape,
        )

    expected_output_shape = (
        2,
        1,
        256,
        256,
    )

    if tuple(
        main_output.shape
    ) != expected_output_shape:
        raise RuntimeError(
            "Main training output has an incorrect shape."
        )

    if len(
        auxiliary_outputs
    ) != 3:
        raise RuntimeError(
            "Expected exactly three auxiliary outputs."
        )

    for index, auxiliary_output in enumerate(
        auxiliary_outputs,
        start=1,
    ):
        if tuple(
            auxiliary_output.shape
        ) != expected_output_shape:
            raise RuntimeError(
                f"Auxiliary output {index} has an incorrect shape."
            )

    # --------------------------------------------------------
    # Evaluation-mode test
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        evaluation_output = model(
            input_tensor
        )

    print()
    print("=" * 70)
    print("Evaluation Mode")
    print("=" * 70)

    if not torch.is_tensor(
        evaluation_output
    ):
        raise RuntimeError(
            "Evaluation output must be a tensor."
        )

    print(
        "Evaluation output shape:",
        evaluation_output.shape,
    )

    if tuple(
        evaluation_output.shape
    ) != expected_output_shape:
        raise RuntimeError(
            "Evaluation output has an incorrect shape."
        )

    # --------------------------------------------------------
    # Forced auxiliary-output evaluation test
    # --------------------------------------------------------

    with torch.no_grad():
        analysis_output = model(
            input_tensor,
            return_auxiliary=True,
        )

    print()
    print("=" * 70)
    print("Forced Auxiliary Evaluation")
    print("=" * 70)

    if not isinstance(
        analysis_output,
        dict,
    ):
        raise RuntimeError(
            "Forced auxiliary output must be a dictionary."
        )

    print(
        "Main output shape:",
        analysis_output[
            "main_output"
        ].shape,
    )

    for index, auxiliary_output in enumerate(
        analysis_output[
            "auxiliary_outputs"
        ],
        start=1,
    ):
        print(
            f"Auxiliary output {index} shape:",
            auxiliary_output.shape,
        )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    trainable_parameters = (
        count_trainable_parameters(
            model
        )
    )

    print()
    print("=" * 70)
    print("Model Summary")
    print("=" * 70)

    print(
        "Trainable parameters:",
        trainable_parameters,
    )

    print()
    print(
        "All MV-TransUNet deep-supervision tests passed successfully."
    )