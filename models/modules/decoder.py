"""
MV-TransUNet Multi-Scale Progressive Decoder

Features:
- Transformer feature reconstruction
- CNN skip fusion
- Progressive upsampling
- Boundary-preserving convolution blocks
"""


import torch

import torch.nn as nn

import torch.nn.functional as F



# ============================================================
# CONVOLUTION REFINEMENT BLOCK
# ============================================================


class ConvBlock(nn.Module):


    def __init__(

        self,

        in_channels,

        out_channels

    ):


        super().__init__()



        self.block = nn.Sequential(

            nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size=3,

                padding=1,

                bias=False

            ),


            nn.BatchNorm2d(

                out_channels

            ),


            nn.ReLU(

                inplace=True

            ),


            nn.Conv2d(

                out_channels,

                out_channels,

                kernel_size=3,

                padding=1,

                bias=False

            ),


            nn.BatchNorm2d(

                out_channels

            ),


            nn.ReLU(

                inplace=True

            )

        )



    def forward(self,x):

        return self.block(x)





# ============================================================
# DECODER UPSAMPLING BLOCK
# ============================================================


class DecoderBlock(nn.Module):


    def __init__(

        self,

        input_channels,

        skip_channels,

        output_channels

    ):


        super().__init__()



        self.conv = ConvBlock(

            input_channels + skip_channels,

            output_channels

        )



    def forward(

        self,

        x,

        skip

    ):



        x = F.interpolate(

            x,

            size=skip.shape[-2:],

            mode="bilinear",

            align_corners=False

        )



        x = torch.cat(

            [

                x,

                skip

            ],

            dim=1

        )



        x = self.conv(

            x

        )


        return x





# ============================================================
# FINAL SEGMENTATION HEAD
# ============================================================



class SegmentationHead(nn.Module):


    def __init__(

        self,

        in_channels,

        out_channels=1

    ):


        super().__init__()



        self.head = nn.Sequential(

            nn.Conv2d(

                in_channels,

                in_channels // 2,

                kernel_size=3,

                padding=1

            ),


            nn.BatchNorm2d(

                in_channels // 2

            ),


            nn.ReLU(

                inplace=True

            ),


            nn.Conv2d(

                in_channels // 2,

                out_channels,

                kernel_size=1

            )

        )



    def forward(self,x):

        return self.head(x)





# ============================================================
# MV-TRANSUNET DECODER
# ============================================================



class MultiScaleDecoder(nn.Module):


    """
    Progressive decoder:

    Transformer
        |
        ↓
    Skip3
        |
        ↓
    Skip2
        |
        ↓
    Skip1
        |
        ↓
    Segmentation


    """



    def __init__(self):


        super().__init__()



        self.decoder1 = DecoderBlock(

            input_channels=768,

            skip_channels=1024,

            output_channels=512

        )



        self.decoder2 = DecoderBlock(

            input_channels=512,

            skip_channels=512,

            output_channels=256

        )



        self.decoder3 = DecoderBlock(

            input_channels=256,

            skip_channels=256,

            output_channels=128

        )



        self.final_upsample = nn.Sequential(

            nn.ConvTranspose2d(

                128,

                64,

                kernel_size=2,

                stride=2

            ),


            nn.BatchNorm2d(

                64

            ),


            nn.ReLU(

                inplace=True

            )

        )



        self.segmentation_head = SegmentationHead(

            64,

            1

        )



    def forward(

        self,

        transformer_feature,

        skip1,

        skip2,

        skip3

    ):



        x = self.decoder1(

            transformer_feature,

            skip3

        )



        x = self.decoder2(

            x,

            skip2

        )



        x = self.decoder3(

            x,

            skip1

        )



        x = self.final_upsample(

            x

        )



        output = self.segmentation_head(

            x

        )



        output = F.interpolate(

            output,

            scale_factor=2,

            mode="bilinear",

            align_corners=False

        )



        return output





# ============================================================
# TEST
# ============================================================



if __name__ == "__main__":


    transformer = torch.randn(

        2,

        768,

        8,

        8

    )


    skip3 = torch.randn(

        2,

        1024,

        16,

        16

    )


    skip2 = torch.randn(

        2,

        512,

        32,

        32

    )


    skip1 = torch.randn(

        2,

        256,

        64,

        64

    )


    decoder = MultiScaleDecoder()



    output = decoder(

        transformer,

        skip1,

        skip2,

        skip3

    )



    print(

        "Output shape:",

        output.shape

    )