"""
MV-TransUNet with parallel topology-aware supervision.

This model preserves the original MV-TransUNet segmentation path and adds
a Topology Feature Fusion Module in parallel.

Architecture
------------
Input image
    -> TransUNet encoder
    -> Vessel Attention Module
    -> Multi-scale decoder
        -> Original segmentation head
        -> Final 64-channel decoder feature
            -> Topology Feature Fusion Module
                -> topology segmentation logits
                -> skeleton logits
                -> endpoint logits
                -> junction logits

The original segmentation path is intentionally retained during this
integration phase so that:

1. Existing training and evaluation code remains compatible.
2. Baseline and topology-aware outputs can be compared directly.
3. The new topology branch can be debugged independently.
4. Existing checkpoints can be partially loaded.
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn

from models.mv_transunet import MVTransUNet
from models.modules.topology_feature_fusion import (
    TopologyAwarePredictionModule,
    TopologyHeadOutput,
)


# ============================================================
# TYPE DEFINITIONS
# ============================================================


TensorList = List[torch.Tensor]

TopologyModelOutput = Dict[
    str,
    Union[
        torch.Tensor,
        TensorList,
    ],
]


# ============================================================
# TOPOLOGY-AWARE MV-TRANSUNET
# ============================================================


class MVTransUNetTopology(MVTransUNet):
    """
    MV-TransUNet with a parallel topology-aware prediction branch.

    The class inherits the complete baseline MV-TransUNet architecture:

    - hybrid CNN-Transformer encoder;
    - Vessel Attention Module;
    - multi-scale progressive decoder;
    - original segmentation head;
    - auxiliary deep-supervision heads.

    A TopologyAwarePredictionModule receives the final decoder feature map
    and predicts:

    - a second vessel-segmentation map;
    - a vessel skeleton map;
    - an endpoint map;
    - a junction map.

    Parameters
    ----------
    pretrained:
        Load pretrained encoder weights when supported by the backbone.

    transformer_channels:
        Number of channels in the transformer representation.

    vessel_reduction_ratio:
        Reduction ratio used by the Vessel Attention Module.

    output_channels:
        Number of channels produced by the original segmentation decoder.
        Retinal vessel segmentation normally uses one output channel.

    deep_supervision:
        Enable the original decoder auxiliary segmentation heads.

    decoder_dropout_rate:
        Spatial dropout probability used inside the decoder.

    topology_channels:
        Number of channels in the shared topology-aware representation.

    topology_branch_channels:
        Number of channels produced by each multi-scale TFFM branch.

    topology_head_hidden_channels:
        Hidden channels in each task-specific topology prediction head.

    topology_refinement_blocks:
        Number of residual topology-refinement blocks.

    topology_fusion_dropout:
        Dropout probability inside topology-refinement blocks.

    topology_head_dropout:
        Dropout probability inside task-specific output heads.
    """

    FINAL_DECODER_CHANNELS = 64

    def __init__(
        self,
        pretrained: bool = True,
        transformer_channels: int = 768,
        vessel_reduction_ratio: int = 16,
        output_channels: int = 1,
        deep_supervision: bool = True,
        decoder_dropout_rate: float = 0.1,
        topology_channels: int = 128,
        topology_branch_channels: int = 32,
        topology_head_hidden_channels: int = 64,
        topology_refinement_blocks: int = 2,
        topology_fusion_dropout: float = 0.1,
        topology_head_dropout: float = 0.0,
        vessel_attention_scale_conditioning: bool = False,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            transformer_channels=transformer_channels,
            vessel_reduction_ratio=vessel_reduction_ratio,
            output_channels=output_channels,
            deep_supervision=deep_supervision,
            decoder_dropout_rate=decoder_dropout_rate,
            vessel_attention_scale_conditioning=(
                vessel_attention_scale_conditioning
            ),
        )

        if output_channels != 1:
            raise ValueError(
                "MVTransUNetTopology currently supports binary vessel "
                "segmentation only, so output_channels must equal 1."
            )

        self.topology_channels = topology_channels
        self.topology_branch_channels = topology_branch_channels
        self.topology_head_hidden_channels = (
            topology_head_hidden_channels
        )
        self.topology_refinement_blocks = topology_refinement_blocks
        self.topology_fusion_dropout = topology_fusion_dropout
        self.topology_head_dropout = topology_head_dropout

        self.topology_prediction = TopologyAwarePredictionModule(
            decoder_channels=self.FINAL_DECODER_CHANNELS,
            topology_channels=topology_channels,
            branch_channels=topology_branch_channels,
            head_hidden_channels=topology_head_hidden_channels,
            number_of_refinement_blocks=topology_refinement_blocks,
            fusion_dropout_probability=topology_fusion_dropout,
            head_dropout_probability=topology_head_dropout,
        )

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:
        """Validate input tensor dimensions and channel count."""

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape [batch, channels, height, width], "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[1] != 3:
            raise ValueError(
                "MVTransUNetTopology expects three input channels, "
                f"but received {x.shape[1]}."
            )

        if x.shape[-2] <= 0 or x.shape[-1] <= 0:
            raise ValueError(
                "Input spatial dimensions must be positive."
            )

    def _extract_encoder_features(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the encoder and validate its required output features.
        """

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

        return features

    @staticmethod
    def _validate_decoder_output(
        decoder_output: object,
    ) -> TopologyModelOutput:
        """
        Validate the rich decoder output required for topology integration.
        """

        if not isinstance(decoder_output, dict):
            raise RuntimeError(
                "Topology integration requires the decoder to return a "
                "dictionary. The decoder must be called with "
                "return_auxiliary=True."
            )

        required_keys = {
            "main_output",
            "auxiliary_outputs",
            "decoder_features",
        }

        missing_keys = (
            required_keys
            - set(decoder_output.keys())
        )

        if missing_keys:
            raise KeyError(
                "Decoder output is missing required entries: "
                f"{sorted(missing_keys)}"
            )

        decoder_features = decoder_output[
            "decoder_features"
        ]

        if not isinstance(decoder_features, list):
            raise TypeError(
                "decoder_features must be returned as a list."
            )

        if len(decoder_features) != 4:
            raise RuntimeError(
                "Expected four decoder feature maps, but received "
                f"{len(decoder_features)}."
            )

        final_decoder_feature = decoder_features[-1]

        if not torch.is_tensor(final_decoder_feature):
            raise TypeError(
                "The final decoder feature must be a tensor."
            )

        if final_decoder_feature.ndim != 4:
            raise RuntimeError(
                "The final decoder feature must have shape [B, C, H, W]."
            )

        if (
            final_decoder_feature.shape[1]
            != MVTransUNetTopology.FINAL_DECODER_CHANNELS
        ):
            raise RuntimeError(
                "The topology module expects the final decoder feature "
                f"to contain {MVTransUNetTopology.FINAL_DECODER_CHANNELS} "
                "channels, but received "
                f"{final_decoder_feature.shape[1]}."
            )

        return decoder_output

    @staticmethod
    def _build_topology_output_dictionary(
        original_decoder_output: TopologyModelOutput,
        topology_output: TopologyHeadOutput,
    ) -> TopologyModelOutput:
        """
        Combine baseline decoder predictions and topology predictions.

        Naming is intentionally explicit:

        original_main_output:
            Segmentation logits from the original MV-TransUNet head.

        topology_segmentation_logits:
            Segmentation logits from the new shared topology representation.
        """

        return {
            "main_output": (
                topology_output.segmentation_logits
            ),
            "original_main_output": (
                original_decoder_output["main_output"]
            ),
            "topology_segmentation_logits": (
                topology_output.segmentation_logits
            ),
            "skeleton_logits": (
                topology_output.skeleton_logits
            ),
            "endpoint_logits": (
                topology_output.endpoint_logits
            ),
            "junction_logits": (
                topology_output.junction_logits
            ),
            "auxiliary_outputs": (
                original_decoder_output["auxiliary_outputs"]
            ),
            "decoder_features": (
                original_decoder_output["decoder_features"]
            ),
            "shared_topology_features": (
                topology_output.shared_topology_features
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        return_topology: bool | None = None,
        return_auxiliary: bool = False,
        scale_ratio: torch.Tensor | float | None = None,
    ) -> Union[
        torch.Tensor,
        TopologyModelOutput,
    ]:
        """
        Run the topology-aware MV-TransUNet forward pass.

        Output behaviour
        ----------------
        During training:
            A dictionary is returned by default because topology losses
            require all task predictions.

        During evaluation:
            The topology-aware segmentation tensor is returned by default
            for compatibility with existing evaluation code.

        Setting return_topology=True:
            Always returns the complete prediction dictionary.

        Setting return_topology=False:
            Always returns only topology-aware segmentation logits.

        Parameters
        ----------
        x:
            Input tensor with shape [B, 3, H, W].

        return_topology:
            Controls whether the complete topology dictionary is returned.
            When None, dictionary output is used during training and tensor
            output is used during evaluation.

        return_auxiliary:
            Preserve the interface of the original MV-TransUNet. When true,
            the complete dictionary is returned so auxiliary predictions are
            accessible.

        scale_ratio:
            Optional per-sample scale-conditioning signal for the Vessel
            Attention Module, with shape [B] (or a Python float, broadcast
            to every sample in the batch). Ignored unless the model was
            constructed with vessel_attention_scale_conditioning=True, in
            which case it defaults to 1.0 (no scale change) when omitted.
        """

        self._validate_input(x)

        input_size = (
            x.shape[-2],
            x.shape[-1],
        )

        features = self._extract_encoder_features(x)

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

        if isinstance(scale_ratio, (int, float)):
            scale_ratio = torch.full(
                (transformer_feature.shape[0],),
                float(scale_ratio),
                device=transformer_feature.device,
            )

        transformer_feature = self.vessel_attention(
            transformer_feature,
            scale_ratio=scale_ratio,
        )

        # Always request the decoder's rich dictionary because the topology
        # branch needs the final 64-channel decoder representation.
        decoder_output = self.decoder(
            transformer_feature=transformer_feature,
            skip1=skip1,
            skip2=skip2,
            skip3=skip3,
            return_auxiliary=True,
        )

        decoder_output = self._validate_decoder_output(
            decoder_output
        )

        final_decoder_feature = decoder_output[
            "decoder_features"
        ][-1]

        topology_output = self.topology_prediction(
            decoder_features=final_decoder_feature,
            output_size=input_size,
        )

        complete_output = (
            self._build_topology_output_dictionary(
                original_decoder_output=decoder_output,
                topology_output=topology_output,
            )
        )

        if return_topology is None:
            should_return_dictionary = (
                self.training
                or return_auxiliary
            )
        else:
            should_return_dictionary = (
                return_topology
                or return_auxiliary
            )

        if should_return_dictionary:
            return complete_output

        return complete_output[
            "topology_segmentation_logits"
        ]


# ============================================================
# MODEL BUILDER
# ============================================================


def build_mv_transunet_topology(
    pretrained: bool = True,
    transformer_channels: int = 768,
    vessel_reduction_ratio: int = 16,
    output_channels: int = 1,
    deep_supervision: bool = True,
    decoder_dropout_rate: float = 0.1,
    topology_channels: int = 128,
    topology_branch_channels: int = 32,
    topology_head_hidden_channels: int = 64,
    topology_refinement_blocks: int = 2,
    topology_fusion_dropout: float = 0.1,
    topology_head_dropout: float = 0.0,
    vessel_attention_scale_conditioning: bool = False,
) -> MVTransUNetTopology:
    """Build the parallel topology-aware MV-TransUNet model."""

    return MVTransUNetTopology(
        pretrained=pretrained,
        transformer_channels=transformer_channels,
        vessel_reduction_ratio=vessel_reduction_ratio,
        output_channels=output_channels,
        deep_supervision=deep_supervision,
        decoder_dropout_rate=decoder_dropout_rate,
        topology_channels=topology_channels,
        topology_branch_channels=topology_branch_channels,
        topology_head_hidden_channels=(
            topology_head_hidden_channels
        ),
        topology_refinement_blocks=(
            topology_refinement_blocks
        ),
        topology_fusion_dropout=topology_fusion_dropout,
        topology_head_dropout=topology_head_dropout,
        vessel_attention_scale_conditioning=(
            vessel_attention_scale_conditioning
        ),
    )


# ============================================================
# PARAMETER UTILITIES
# ============================================================


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Count all trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_topology_parameters(
    model: MVTransUNetTopology,
) -> int:
    """Count parameters introduced by the topology branch."""

    return sum(
        parameter.numel()
        for parameter in (
            model.topology_prediction.parameters()
        )
        if parameter.requires_grad
    )


# ============================================================
# INTEGRATION SMOKE TEST
# ============================================================


def _assert_prediction_shape(
    prediction: torch.Tensor,
    expected_shape: tuple[int, ...],
    prediction_name: str,
) -> None:
    """Check one prediction map during the smoke test."""

    if not torch.is_tensor(prediction):
        raise TypeError(
            f"{prediction_name} must be a tensor."
        )

    if tuple(prediction.shape) != expected_shape:
        raise RuntimeError(
            f"{prediction_name} has shape "
            f"{tuple(prediction.shape)}, expected {expected_shape}."
        )


def run_integration_smoke_test() -> None:
    """
    Test full encoder-to-topology forward and backward propagation.

    Batch size one is used to reduce local VRAM and RAM requirements.
    """

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("MV-TransUNet Topology Integration Test")
    print("=" * 72)
    print("Device:", device)

    model = MVTransUNetTopology(
        pretrained=False,
        transformer_channels=768,
        vessel_reduction_ratio=16,
        output_channels=1,
        deep_supervision=True,
        decoder_dropout_rate=0.1,
        topology_channels=128,
        topology_branch_channels=32,
        topology_head_hidden_channels=64,
        topology_refinement_blocks=2,
        topology_fusion_dropout=0.1,
        topology_head_dropout=0.0,
    ).to(device)

    input_tensor = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
        requires_grad=True,
    )

    expected_prediction_shape = (
        1,
        1,
        256,
        256,
    )

    expected_topology_feature_shape = (
        1,
        128,
        128,
        128,
    )

    # --------------------------------------------------------
    # Training output
    # --------------------------------------------------------

    model.train()

    training_output = model(
        input_tensor
    )

    if not isinstance(training_output, dict):
        raise RuntimeError(
            "Training must return a dictionary."
        )

    required_output_keys = {
        "main_output",
        "original_main_output",
        "topology_segmentation_logits",
        "skeleton_logits",
        "endpoint_logits",
        "junction_logits",
        "auxiliary_outputs",
        "decoder_features",
        "shared_topology_features",
    }

    missing_output_keys = (
        required_output_keys
        - set(training_output.keys())
    )

    if missing_output_keys:
        raise KeyError(
            "Training output is missing keys: "
            f"{sorted(missing_output_keys)}"
        )

    prediction_names = [
        "main_output",
        "original_main_output",
        "topology_segmentation_logits",
        "skeleton_logits",
        "endpoint_logits",
        "junction_logits",
    ]

    for prediction_name in prediction_names:
        _assert_prediction_shape(
            prediction=training_output[
                prediction_name
            ],
            expected_shape=expected_prediction_shape,
            prediction_name=prediction_name,
        )

    auxiliary_outputs = training_output[
        "auxiliary_outputs"
    ]

    if not isinstance(auxiliary_outputs, list):
        raise TypeError(
            "auxiliary_outputs must be a list."
        )

    if len(auxiliary_outputs) != 3:
        raise RuntimeError(
            "Expected exactly three auxiliary outputs."
        )

    for index, auxiliary_output in enumerate(
        auxiliary_outputs,
        start=1,
    ):
        _assert_prediction_shape(
            prediction=auxiliary_output,
            expected_shape=expected_prediction_shape,
            prediction_name=(
                f"auxiliary_output_{index}"
            ),
        )

    shared_features = training_output[
        "shared_topology_features"
    ]

    if tuple(
        shared_features.shape
    ) != expected_topology_feature_shape:
        raise RuntimeError(
            "shared_topology_features has shape "
            f"{tuple(shared_features.shape)}, expected "
            f"{expected_topology_feature_shape}."
        )

    # Use every prediction branch so gradient propagation is tested
    # throughout the complete multi-task architecture.
    test_loss = (
        training_output[
            "topology_segmentation_logits"
        ].mean()
        + training_output[
            "original_main_output"
        ].mean()
        + training_output[
            "skeleton_logits"
        ].mean()
        + training_output[
            "endpoint_logits"
        ].mean()
        + training_output[
            "junction_logits"
        ].mean()
    )

    for auxiliary_output in auxiliary_outputs:
        test_loss = (
            test_loss
            + auxiliary_output.mean()
        )

    test_loss.backward()

    if input_tensor.grad is None:
        raise RuntimeError(
            "Gradient did not propagate to the model input."
        )

    topology_gradient_found = any(
        parameter.grad is not None
        for parameter in (
            model.topology_prediction.parameters()
        )
        if parameter.requires_grad
    )

    if not topology_gradient_found:
        raise RuntimeError(
            "No gradient reached the topology module."
        )

    # --------------------------------------------------------
    # Evaluation output
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        evaluation_output = model(
            input_tensor.detach()
        )

    _assert_prediction_shape(
        prediction=evaluation_output,
        expected_shape=expected_prediction_shape,
        prediction_name="evaluation_output",
    )

    with torch.no_grad():
        complete_evaluation_output = model(
            input_tensor.detach(),
            return_topology=True,
        )

    if not isinstance(
        complete_evaluation_output,
        dict,
    ):
        raise RuntimeError(
            "return_topology=True must return a dictionary."
        )

    total_parameters = count_trainable_parameters(
        model
    )

    topology_parameters = count_topology_parameters(
        model
    )

    base_parameters = (
        total_parameters
        - topology_parameters
    )

    topology_percentage = (
        100.0
        * topology_parameters
        / total_parameters
    )

    print()
    print("=" * 72)
    print("Training Output Shapes")
    print("=" * 72)

    for prediction_name in prediction_names:
        print(
            f"{prediction_name:32s}",
            tuple(
                training_output[
                    prediction_name
                ].shape
            ),
        )

    print(
        f"{'shared_topology_features':32s}",
        tuple(shared_features.shape),
    )

    print()
    print("=" * 72)
    print("Evaluation Behaviour")
    print("=" * 72)
    print(
        "Default evaluation output:",
        tuple(evaluation_output.shape),
    )
    print(
        "Complete evaluation dictionary:",
        "PASSED",
    )

    print()
    print("=" * 72)
    print("Parameter Summary")
    print("=" * 72)
    print(
        "Baseline architecture parameters:",
        f"{base_parameters:,}",
    )
    print(
        "Topology branch parameters:",
        f"{topology_parameters:,}",
    )
    print(
        "Total trainable parameters:",
        f"{total_parameters:,}",
    )
    print(
        "Topology percentage:",
        f"{topology_percentage:.4f}%",
    )

    print()
    print("=" * 72)
    print("Gradient propagation: PASSED")
    print("Full topology integration: PASSED")
    print("=" * 72)


if __name__ == "__main__":
    run_integration_smoke_test()