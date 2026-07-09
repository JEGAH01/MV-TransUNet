"""
MV-TransUNet Full Architecture

Boundary-Aware Hybrid CNN-Vision Transformer
for Retinal Vessel Segmentation


Components:

1. ResNet50 + ViT Encoder
2. Vessel Attention Module
3. Multi-scale Decoder


Input:
    B x 3 x H x W


Output:
    B x 1 x H x W

"""


import torch

import torch.nn as nn


from models.backbones.transunet_base import TransUNetBackbone

from models.modules.vessel_attention import VesselAttentionModule

from models.modules.decoder import MultiScaleDecoder





# ============================================================
# MV-TRANSUNET MODEL
# ============================================================



class MVTransUNet(nn.Module):


    def __init__(

        self,

        pretrained=True,

        transformer_channels=768,

        vessel_reduction_ratio=16

    ):


        super().__init__()



        # ----------------------------------------------------
        # Hybrid CNN Transformer Encoder
        # ----------------------------------------------------


        self.encoder = TransUNetBackbone(

            pretrained=pretrained,

            hidden_dim=transformer_channels

        )



        # ----------------------------------------------------
        # Vessel Attention Module
        # ----------------------------------------------------


        self.vessel_attention = VesselAttentionModule(

            channels=transformer_channels,

            reduction_ratio=vessel_reduction_ratio

        )



        # ----------------------------------------------------
        # Multi-scale Decoder
        # ----------------------------------------------------


        self.decoder = MultiScaleDecoder()



    def forward(self,x):


        features = self.encoder(

            x

        )



        transformer_feature = features["transformer"]


        skip1 = features["skip1"]


        skip2 = features["skip2"]


        skip3 = features["skip3"]



        transformer_feature = self.vessel_attention(

            transformer_feature

        )



        output = self.decoder(

            transformer_feature,

            skip1,

            skip2,

            skip3

        )



        return output





# ============================================================
# MODEL BUILDER FUNCTION
# ============================================================



def build_mv_transunet(

    pretrained=True

):


    model = MVTransUNet(

        pretrained=pretrained

    )


    return model





# ============================================================
# MODEL TEST
# ============================================================



if __name__ == "__main__":


    device = (

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )



    model = MVTransUNet(

        pretrained=False

    ).to(device)



    input_tensor = torch.randn(

        2,

        3,

        256,

        256

    ).to(device)



    output = model(

        input_tensor

    )



    print(

        "Input:",

        input_tensor.shape

    )


    print(

        "Output:",

        output.shape

    )


    parameters = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


    print(

        "Trainable parameters:",

        parameters

    )