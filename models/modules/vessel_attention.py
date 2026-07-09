"""
MV-TransUNet Vessel Attention Module

Novel contribution:
Dual Spatial-Channel Attention for retinal vessel enhancement.

Input:
    Feature map B x C x H x W

Output:
    Refined feature map B x C x H x W
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

        reduction_ratio=16

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



    def forward(self,x):


        B,C,H,W = x.shape



        pooled = self.global_pool(

            x

        )



        pooled = pooled.view(

            B,

            C

        )



        weights = self.mlp(

            pooled

        )



        weights = self.activation(

            weights

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
    """



    def __init__(

        self,

        channels,

        reduction_ratio=16

    ):


        super().__init__()



        self.channel_attention = ChannelAttention(

            channels,

            reduction_ratio

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



    def forward(self,x):


        identity = x



        x = self.channel_attention(

            x

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