"""
MV-TransUNet Vessel Attention Module

Novel contribution:
Dual Spatial-Channel Attention for retinal vessel enhancement.

Input:
    Feature map B x C x H x W

Output:
    Refined feature map B x C x H x W

Scale conditioning (SC-VAM extension)
--------------------------------------
When ``scale_conditioning=True``, ChannelAttention additionally accepts a
per-sample scalar ``scale_ratio`` (the ratio of a sample's effective vessel
scale to the DRIVE source scale) and uses it to modulate the channel
attention gates via a small FiLM-style pathway. The pathway's final linear
layer is zero-initialized, so at construction time (before any training)
the module is numerically identical to the unconditioned VesselAttentionModule
regardless of the scale_ratio value supplied -- this is a deliberate, safe
warm start, not a claim that the pathway is well-trained at init.

The conditioning input is log(scale_ratio) rather than the raw ratio, since
scale ratios are multiplicative quantities and log(1.0) == 0.0 aligns with
"no scale change" naturally, without requiring the network to learn an
arbitrary offset.
"""


import torch

import torch.nn as nn



# ============================================================
# CHANNEL ATTENTION MODULE
# ============================================================


class ChannelAttention(nn.Module):


    def __init__(

        self,

        channels,

        reduction_ratio=16,

        scale_conditioning=False,

        scale_hidden_channels=16

    ):


        super().__init__()



        reduced_channels = channels // reduction_ratio



        self.global_pool = nn.AdaptiveAvgPool2d(

            1

        )



        self.mlp = nn.Sequential(

            nn.Linear(

                channels,

                reduced_channels

            ),


            nn.ReLU(

                inplace=True

            ),


            nn.Linear(

                reduced_channels,

                channels

            )

        )



        self.activation = nn.Sigmoid()



        self.scale_conditioning = bool(scale_conditioning)

        if self.scale_conditioning:

            self.scale_mlp = nn.Sequential(

                nn.Linear(
                    1,
                    scale_hidden_channels
                ),

                nn.ReLU(
                    inplace=True
                ),

                nn.Linear(
                    scale_hidden_channels,
                    channels
                ),

            )

            # Zero-initialize the final layer so this pathway contributes
            # exactly zero at construction time, for any scale_ratio value.
            # This makes scale_conditioning=True a safe warm start: at
            # init, the model is identical to scale_conditioning=False.
            nn.init.zeros_(self.scale_mlp[-1].weight)
            nn.init.zeros_(self.scale_mlp[-1].bias)



    def forward(self, x, scale_ratio=None):


        B,C,H,W = x.shape



        pooled = self.global_pool(

            x

        )



        pooled = pooled.view(

            B,

            C

        )



        logits = self.mlp(

            pooled

        )



        if self.scale_conditioning:

            if scale_ratio is None:
                scale_ratio = torch.ones(
                    B,
                    device=x.device,
                    dtype=pooled.dtype,
                )

            scale_ratio = scale_ratio.to(
                device=x.device,
                dtype=pooled.dtype,
            ).view(B)

            # Clamp before taking the log to avoid -inf / NaN if a caller
            # ever passes a zero or negative ratio (should not happen with
            # the augmentation- or VSDI-derived ratios used in this
            # project, but the clamp keeps the module well-defined).
            log_scale_ratio = torch.log(
                torch.clamp(scale_ratio, min=1e-3)
            ).view(B, 1)

            logits = logits + self.scale_mlp(log_scale_ratio)



        weights = self.activation(

            logits

        )



        weights = weights.view(

            B,

            C,

            1,

            1

        )



        return x * weights





# ============================================================
# SPATIAL ATTENTION MODULE
# ============================================================



class SpatialAttention(nn.Module):


    def __init__(self):


        super().__init__()



        self.conv = nn.Conv2d(

            2,

            1,

            kernel_size=7,

            padding=3,

            bias=False

        )


        self.activation = nn.Sigmoid()



    def forward(self,x):


        average_map = torch.mean(

            x,

            dim=1,

            keepdim=True

        )



        max_map = torch.max(

            x,

            dim=1,

            keepdim=True

        )[0]



        combined = torch.cat(

            [

                average_map,

                max_map

            ],

            dim=1

        )



        attention_map = self.conv(

            combined

        )



        attention_map = self.activation(

            attention_map

        )



        return x * attention_map





# ============================================================
# VESSEL ATTENTION MODULE
# ============================================================



class VesselAttentionModule(nn.Module):


    """
    Dual attention mechanism:

    Channel Attention
          +
    Spatial Attention
          +
    Residual Enhancement


    Designed for retinal vessel features.

    When scale_conditioning=True, the channel-attention gate additionally
    receives a per-sample scale_ratio (see forward()), letting the module
    modulate its channel reweighting according to how far the current
    sample's vessel scale is from the DRIVE source scale it was trained on.
    """



    def __init__(

        self,

        channels,

        reduction_ratio=16,

        scale_conditioning=False

    ):


        super().__init__()



        self.channel_attention = ChannelAttention(

            channels,

            reduction_ratio,

            scale_conditioning=scale_conditioning,

        )



        self.spatial_attention = SpatialAttention()



        self.refinement = nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                kernel_size=3,

                padding=1,

                bias=False

            ),


            nn.BatchNorm2d(

                channels

            ),


            nn.ReLU(

                inplace=True

            )

        )



    def forward(self, x, scale_ratio=None):


        identity = x



        x = self.channel_attention(

            x,

            scale_ratio=scale_ratio,

        )



        x = self.spatial_attention(

            x

        )



        x = self.refinement(

            x

        )



        output = x + identity



        return output





# ============================================================
# TEST FUNCTION
# ============================================================


if __name__ == "__main__":


    feature = torch.randn(

        2,

        768,

        8,

        8

    )



    vam = VesselAttentionModule(

        channels=768

    )



    output = vam(

        feature

    )



    print(

        "Input shape:",

        feature.shape

    )


    print(

        "Output shape:",

        output.shape

    )

    print()
    print("Scale-conditioning smoke test")
    print("-" * 40)

    sc_vam = VesselAttentionModule(
        channels=768,
        scale_conditioning=True,
    )

    unit_scale = torch.ones(2)
    varied_scale = torch.tensor([0.75, 2.0])

    output_unit = sc_vam(feature, scale_ratio=unit_scale)
    output_varied = sc_vam(feature, scale_ratio=varied_scale)
    output_none = sc_vam(feature, scale_ratio=None)

    identical_at_init = torch.allclose(output_unit, output_varied)
    identical_to_default = torch.allclose(output_unit, output_none)

    print("Output shape (scale-conditioned):", output_varied.shape)
    print(
        "Identical output for scale_ratio=1.0 vs scale_ratio=[0.75, 2.0] "
        "at init (expected True, zero-init warm start):",
        identical_at_init,
    )
    print(
        "scale_ratio=None matches scale_ratio=1.0 (expected True):",
        identical_to_default,
    )

    if not identical_at_init:
        raise RuntimeError(
            "Zero-init warm start failed: scale conditioning is not inert "
            "at initialization."
        )

    if not identical_to_default:
        raise RuntimeError(
            "scale_ratio=None did not default to scale_ratio=1.0 behaviour."
        )

    # Confirm the pathway actually receives gradient once the zero-init
    # weights are perturbed (i.e. it is wired into the graph correctly).
    with torch.no_grad():
        for parameter in sc_vam.channel_attention.scale_mlp.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))

    output_unit_trained = sc_vam(feature, scale_ratio=unit_scale)
    output_varied_trained = sc_vam(feature, scale_ratio=varied_scale)

    now_different = not torch.allclose(
        output_unit_trained, output_varied_trained
    )

    print(
        "After perturbing scale_mlp weights, outputs now differ by "
        "scale_ratio (expected True):",
        now_different,
    )

    if not now_different:
        raise RuntimeError(
            "Scale conditioning pathway appears disconnected: varying "
            "scale_ratio had no effect after perturbing scale_mlp."
        )

    test_input = torch.randn(2, 768, 8, 8, requires_grad=True)
    test_output = sc_vam(test_input, scale_ratio=varied_scale)
    test_output.sum().backward()

    gradient_reached_scale_mlp = any(
        parameter.grad is not None
        for parameter in sc_vam.channel_attention.scale_mlp.parameters()
    )

    print(
        "Gradient reached scale_mlp (expected True):",
        gradient_reached_scale_mlp,
    )

    if not gradient_reached_scale_mlp:
        raise RuntimeError(
            "Gradient did not reach the scale-conditioning pathway."
        )

    print()
    print("All VesselAttentionModule scale-conditioning checks passed.")
