"""
MV-TransUNet Multi-Scale Progressive Decoder

Features:
- Transformer feature reconstruction
- CNN skip-feature fusion
- Progressive bilinear upsampling
- Boundary-preserving convolution refinement
- Optional deep supervision
- Optional spatial (channel) dropout in decoder ConvBlocks, added in
  response to an observed train/validation Dice gap -- see
  MultiScaleDecoder's dropout_rate parameter
- Backward-compatible inference output

Deep-supervision outputs:
- Auxiliary output 1 from decoder stage 1
- Auxiliary output 2 from decoder stage 2
- Auxiliary output 3 from decoder stage 3
- Main output from the final decoder stage
"""


from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONVOLUTION REFINEMENT BLOCK
# ============================================================


class ConvBlock(nn.Module):
    """
    Two-layer convolution refinement block.

    Structure:
        Conv3x3
        BatchNorm
        ReLU
        Conv3x3
        BatchNorm
        ReLU
        Dropout2d (optional, applied only if dropout_rate > 0)

    dropout_rate uses nn.Dropout2d (spatial/channel dropout), not
    plain elementwise Dropout: neighboring pixels in a conv feature
    map are highly spatially correlated, so zeroing individual pixels
    barely perturbs the effective information content (a dropped
    pixel's neighbors still carry nearly the same signal). Dropout2d
    zeroes entire feature CHANNELS instead, which is the standard,
    effective way to regularize convolutional feature maps (Tompson
    et al., 2015) and is what similar segmentation architectures use.

    Only added to DecoderBlock's internal ConvBlock (decoder1/2/3),
    NOT to SegmentationHead or final_upsample -- regularizing the
    feature-extraction stages, not the final logit-producing layers
    immediately before the loss, is the standard placement; dropout
    directly before a 1x1 logit head tends to just inject prediction
    noise rather than improve generalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be greater than zero."
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels must be greater than zero."
            )

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                "dropout_rate must be in [0, 1)."
            )

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                num_features=out_channels,
            ),

            nn.ReLU(
                inplace=True,
            ),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                num_features=out_channels,
            ),

            nn.ReLU(
                inplace=True,
            ),
        ]

        if dropout_rate > 0.0:
            layers.append(
                nn.Dropout2d(p=dropout_rate)
            )

        self.block = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


# ============================================================
# DECODER UPSAMPLING BLOCK
# ============================================================


class DecoderBlock(nn.Module):
    """
    Upsample the decoder feature, concatenate a CNN skip feature,
    and refine the fused representation.
    """

    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be greater than zero."
            )

        if skip_channels <= 0:
            raise ValueError(
                "skip_channels must be greater than zero."
            )

        if output_channels <= 0:
            raise ValueError(
                "output_channels must be greater than zero."
            )

        self.input_channels = input_channels
        self.skip_channels = skip_channels
        self.output_channels = output_channels

        self.conv = ConvBlock(
            in_channels=(
                input_channels
                + skip_channels
            ),
            out_channels=output_channels,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Decoder input must have shape "
                "[batch, channels, height, width]."
            )

        if skip.ndim != 4:
            raise ValueError(
                "Skip feature must have shape "
                "[batch, channels, height, width]."
            )

        if x.shape[0] != skip.shape[0]:
            raise ValueError(
                "Decoder input and skip feature must have "
                "the same batch size."
            )

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        return self.conv(x)


# ============================================================
# SEGMENTATION HEAD
# ============================================================


class SegmentationHead(nn.Module):
    """
    Convert a decoder feature map into segmentation logits.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = None,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be greater than zero."
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels must be greater than zero."
            )

        if hidden_channels is None:
            hidden_channels = max(
                in_channels // 2,
                out_channels,
            )

        if hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be greater than zero."
            )

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                num_features=hidden_channels,
            ),

            nn.ReLU(
                inplace=True,
            ),

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(x)


# ============================================================
# MV-TRANSUNET MULTI-SCALE DECODER
# ============================================================


class MultiScaleDecoder(nn.Module):
    """
    Progressive MV-TransUNet decoder.

    Feature hierarchy:

        Transformer feature: 768 channels, 8x8
                    |
                    v
        Decoder stage 1 + skip3: 512 channels, 16x16
                    |
                    v
        Decoder stage 2 + skip2: 256 channels, 32x32
                    |
                    v
        Decoder stage 3 + skip1: 128 channels, 64x64
                    |
                    v
        Final upsampling: 64 channels, 128x128
                    |
                    v
        Main segmentation logits: 1 channel, 256x256

    When deep supervision is enabled, auxiliary segmentation heads
    are attached to decoder stages 1, 2, and 3.

    Auxiliary logits are resized to the final output resolution before
    being returned.
    """

    def __init__(
        self,
        output_channels: int = 1,
        deep_supervision: bool = False,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                "dropout_rate must be in [0, 1)."
            )

        self.output_channels = output_channels
        self.deep_supervision = deep_supervision
        self.dropout_rate = dropout_rate

        # ----------------------------------------------------
        # Progressive decoder stages
        #
        # dropout_rate=0.1 by default: motivated by an observed
        # train/validation Dice gap (~0.85 train vs. ~0.78 val) on a
        # 16-image training split -- a small, targeted regularization
        # response to a measured overfitting signal, not a
        # speculative addition. Set to 0.0 to exactly reproduce prior
        # (undropped) runs for a clean ablation comparison.
        # ----------------------------------------------------

        self.decoder1 = DecoderBlock(
            input_channels=768,
            skip_channels=1024,
            output_channels=512,
            dropout_rate=dropout_rate,
        )

        self.decoder2 = DecoderBlock(
            input_channels=512,
            skip_channels=512,
            output_channels=256,
            dropout_rate=dropout_rate,
        )

        self.decoder3 = DecoderBlock(
            input_channels=256,
            skip_channels=256,
            output_channels=128,
            dropout_rate=dropout_rate,
        )

        # ----------------------------------------------------
        # Final decoder upsampling
        # ----------------------------------------------------

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False,
            ),

            nn.BatchNorm2d(
                num_features=64,
            ),

            nn.ReLU(
                inplace=True,
            ),
        )

        # ----------------------------------------------------
        # Main segmentation head
        # ----------------------------------------------------

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=output_channels,
            hidden_channels=32,
        )

        # ----------------------------------------------------
        # Auxiliary deep-supervision heads
        # ----------------------------------------------------

        if self.deep_supervision:
            self.auxiliary_head1 = SegmentationHead(
                in_channels=512,
                out_channels=output_channels,
                hidden_channels=128,
            )

            self.auxiliary_head2 = SegmentationHead(
                in_channels=256,
                out_channels=output_channels,
                hidden_channels=64,
            )

            self.auxiliary_head3 = SegmentationHead(
                in_channels=128,
                out_channels=output_channels,
                hidden_channels=32,
            )

        else:
            self.auxiliary_head1 = None
            self.auxiliary_head2 = None
            self.auxiliary_head3 = None

    def _resize_logits(
        self,
        logits: torch.Tensor,
        output_size: tuple,
    ) -> torch.Tensor:
        """
        Resize segmentation logits to the requested spatial size.
        """

        if logits.shape[-2:] == output_size:
            return logits

        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        transformer_feature: torch.Tensor,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
        skip3: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> Union[
        torch.Tensor,
        Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    ]:
        """
        Run progressive decoding.

        Args:
            transformer_feature:
                Deep transformer feature map with shape
                [B, 768, H/32, W/32].

            skip1:
                Shallow CNN skip feature with 256 channels.

            skip2:
                Intermediate CNN skip feature with 512 channels.

            skip3:
                Deep CNN skip feature with 1024 channels.

            return_auxiliary:
                Force auxiliary outputs to be returned. This is useful
                for debugging or validation studies.

        Returns:
            If deep supervision is disabled:
                Tensor containing the final segmentation logits.

            If deep supervision is enabled and either the module is
            training or return_auxiliary=True:
                Dictionary containing:
                    main_output
                    auxiliary_outputs

            Otherwise:
                Tensor containing the final segmentation logits.
        """

        # Decoder stage 1
        decoder_feature1 = self.decoder1(
            transformer_feature,
            skip3,
        )

        # Decoder stage 2
        decoder_feature2 = self.decoder2(
            decoder_feature1,
            skip2,
        )

        # Decoder stage 3
        decoder_feature3 = self.decoder3(
            decoder_feature2,
            skip1,
        )

        # Final 2x upsampling
        final_feature = self.final_upsample(
            decoder_feature3
        )

        # Main segmentation logits at 128x128
        main_output = self.segmentation_head(
            final_feature
        )

        # Final 2x interpolation to 256x256
        final_output_size = (
            main_output.shape[-2] * 2,
            main_output.shape[-1] * 2,
        )

        main_output = self._resize_logits(
            main_output,
            output_size=final_output_size,
        )

        should_return_auxiliary = (
            self.deep_supervision
            and (
                self.training
                or return_auxiliary
            )
        )

        if not should_return_auxiliary:
            return main_output

        auxiliary_output1 = self.auxiliary_head1(
            decoder_feature1
        )

        auxiliary_output2 = self.auxiliary_head2(
            decoder_feature2
        )

        auxiliary_output3 = self.auxiliary_head3(
            decoder_feature3
        )

        auxiliary_output1 = self._resize_logits(
            auxiliary_output1,
            output_size=final_output_size,
        )

        auxiliary_output2 = self._resize_logits(
            auxiliary_output2,
            output_size=final_output_size,
        )

        auxiliary_output3 = self._resize_logits(
            auxiliary_output3,
            output_size=final_output_size,
        )

        return {
            "main_output": main_output,

            "auxiliary_outputs": [
                auxiliary_output1,
                auxiliary_output2,
                auxiliary_output3,
            ],

            "decoder_features": [
                decoder_feature1,
                decoder_feature2,
                decoder_feature3,
                final_feature,
            ],
        }


# ============================================================
# TEST
# ============================================================


if __name__ == "__main__":
    torch.manual_seed(42)

    transformer = torch.randn(
        2,
        768,
        8,
        8,
    )

    skip3 = torch.randn(
        2,
        1024,
        16,
        16,
    )

    skip2 = torch.randn(
        2,
        512,
        32,
        32,
    )

    skip1 = torch.randn(
        2,
        256,
        64,
        64,
    )

    # --------------------------------------------------------
    # Test normal backward-compatible mode
    # --------------------------------------------------------

    standard_decoder = MultiScaleDecoder(
        output_channels=1,
        deep_supervision=False,
    )

    standard_decoder.eval()

    standard_output = standard_decoder(
        transformer_feature=transformer,
        skip1=skip1,
        skip2=skip2,
        skip3=skip3,
    )

    print("=" * 60)
    print("Standard Decoder Test")
    print("=" * 60)
    print(
        "Output shape:",
        standard_output.shape,
    )

    expected_shape = (
        2,
        1,
        256,
        256,
    )

    if tuple(standard_output.shape) != expected_shape:
        raise RuntimeError(
            "Standard decoder output shape is incorrect."
        )

    # --------------------------------------------------------
    # Test deep-supervision mode
    # --------------------------------------------------------

    supervised_decoder = MultiScaleDecoder(
        output_channels=1,
        deep_supervision=True,
    )

    supervised_decoder.train()

    supervised_outputs = supervised_decoder(
        transformer_feature=transformer,
        skip1=skip1,
        skip2=skip2,
        skip3=skip3,
    )

    print()
    print("=" * 60)
    print("Deep Supervision Decoder Test")
    print("=" * 60)

    print(
        "Main output shape:",
        supervised_outputs["main_output"].shape,
    )

    for index, auxiliary_output in enumerate(
        supervised_outputs["auxiliary_outputs"],
        start=1,
    ):
        print(
            f"Auxiliary output {index} shape:",
            auxiliary_output.shape,
        )

        if tuple(auxiliary_output.shape) != expected_shape:
            raise RuntimeError(
                f"Auxiliary output {index} shape is incorrect."
            )

    if (
        tuple(
            supervised_outputs[
                "main_output"
            ].shape
        )
        != expected_shape
    ):
        raise RuntimeError(
            "Main deep-supervision output shape is incorrect."
        )

    # --------------------------------------------------------
    # Test deep-supervision inference behavior
    # --------------------------------------------------------

    supervised_decoder.eval()

    inference_output = supervised_decoder(
        transformer_feature=transformer,
        skip1=skip1,
        skip2=skip2,
        skip3=skip3,
    )

    print()
    print("=" * 60)
    print("Deep Supervision Inference Test")
    print("=" * 60)
    print(
        "Inference output shape:",
        inference_output.shape,
    )

    if tuple(inference_output.shape) != expected_shape:
        raise RuntimeError(
            "Inference output shape is incorrect."
        )

    print()
    print("All decoder tests passed successfully.")