"""
MV-TransUNet Hybrid CNN-Vision Transformer Backbone

Components:
- ResNet50 CNN encoder
- ViT-B/16 Transformer encoder
- Patch embedding
- Multi-head self attention
- Transformer blocks
- Skip feature extraction
"""


import torch

import torch.nn as nn

import torchvision.models as models



# ============================================================
# MULTI HEAD SELF ATTENTION
# ============================================================


class MultiHeadSelfAttention(nn.Module):


    def __init__(
        self,
        dim,
        heads=12,
        dropout=0.1
    ):

        super().__init__()


        self.heads = heads


        self.scale = (

            dim // heads

        ) ** -0.5



        self.qkv = nn.Linear(

            dim,

            dim * 3

        )


        self.projection = nn.Linear(

            dim,

            dim

        )


        self.dropout = nn.Dropout(

            dropout

        )



    def forward(self,x):


        B,N,C = x.shape



        qkv = self.qkv(x)



        qkv = qkv.reshape(

            B,

            N,

            3,

            self.heads,

            C // self.heads

        )



        qkv = qkv.permute(

            2,

            0,

            3,

            1,

            4

        )



        q,k,v = qkv[0],qkv[1],qkv[2]



        attention = (

            q @ k.transpose(-2,-1)

        ) * self.scale



        attention = attention.softmax(

            dim=-1

        )


        attention = self.dropout(

            attention

        )


        output = attention @ v



        output = output.transpose(

            1,

            2

        )


        output = output.reshape(

            B,

            N,

            C

        )


        output = self.projection(

            output

        )


        return output





# ============================================================
# TRANSFORMER BLOCK
# ============================================================



class TransformerBlock(nn.Module):


    def __init__(

        self,

        dim=768,

        heads=12,

        mlp_dim=3072,

        dropout=0.1

    ):


        super().__init__()



        self.norm1 = nn.LayerNorm(

            dim

        )


        self.attention = MultiHeadSelfAttention(

            dim,

            heads,

            dropout

        )


        self.norm2 = nn.LayerNorm(

            dim

        )



        self.mlp = nn.Sequential(

            nn.Linear(

                dim,

                mlp_dim

            ),

            nn.GELU(),

            nn.Dropout(

                dropout

            ),

            nn.Linear(

                mlp_dim,

                dim

            ),

            nn.Dropout(

                dropout

            )

        )



    def forward(self,x):


        x = x + self.attention(

            self.norm1(x)

        )


        x = x + self.mlp(

            self.norm2(x)

        )


        return x





# ============================================================
# VISION TRANSFORMER ENCODER
# ============================================================



class VisionTransformerEncoder(nn.Module):


    def __init__(

        self,

        dim=768,

        depth=12,

        heads=12,

        mlp_dim=3072,

        dropout=0.1

    ):


        super().__init__()



        self.layers = nn.ModuleList(

            [

                TransformerBlock(

                    dim,

                    heads,

                    mlp_dim,

                    dropout

                )

                for _ in range(depth)

            ]

        )



        self.norm = nn.LayerNorm(

            dim

        )



    def forward(self,x):


        for block in self.layers:


            x = block(x)



        return self.norm(x)





# ============================================================
# HYBRID RESNET50 + VIT BACKBONE
# ============================================================



class TransUNetBackbone(nn.Module):


    def __init__(

        self,

        pretrained=True,

        hidden_dim=768

    ):


        super().__init__()



        resnet = models.resnet50(

            weights=(

                models.ResNet50_Weights.DEFAULT

                if pretrained

                else None

            )

        )



        self.layer0 = nn.Sequential(

            resnet.conv1,

            resnet.bn1,

            resnet.relu,

            resnet.maxpool

        )



        self.layer1 = resnet.layer1


        self.layer2 = resnet.layer2


        self.layer3 = resnet.layer3


        self.layer4 = resnet.layer4



        self.patch_projection = nn.Conv2d(

            2048,

            hidden_dim,

            kernel_size=1

        )



        self.transformer = VisionTransformerEncoder(

            dim=hidden_dim

        )



        self.position_embedding = nn.Parameter(

            torch.zeros(

                1,

                1024,

                hidden_dim

            )

        )



    def forward(self,x):


        x = self.layer0(x)



        skip1 = self.layer1(x)



        skip2 = self.layer2(skip1)



        skip3 = self.layer3(skip2)



        deep_feature = self.layer4(skip3)



        tokens = self.patch_projection(

            deep_feature

        )



        B,C,H,W = tokens.shape



        tokens = tokens.flatten(

            2

        ).transpose(

            1,

            2

        )



        tokens = tokens + self.position_embedding[:, :tokens.size(1)]



        transformer_output = self.transformer(

            tokens

        )



        transformer_output = transformer_output.transpose(

            1,

            2

        )



        transformer_output = transformer_output.reshape(

            B,

            C,

            H,

            W

        )



        return {

            "transformer": transformer_output,

            "skip1": skip1,

            "skip2": skip2,

            "skip3": skip3

        }