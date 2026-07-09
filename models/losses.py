"""
MV-TransUNet Boundary-Aware Loss Functions

Components:

1. Binary Cross Entropy Loss
2. Soft Dice Loss
3. Laplacian Boundary Loss
4. Combined Joint Loss


Formula:

L = alpha*BCE + beta*Dice + gamma*Boundary


CUDA / AMP Compatible
"""


import torch
import torch.nn as nn
import torch.nn.functional as F





# ============================================================
# SOFT DICE LOSS
# ============================================================


class SoftDiceLoss(nn.Module):


    def __init__(self, smooth=1e-6):

        super().__init__()

        self.smooth = smooth



    def forward(self, prediction, target):


        prediction = torch.sigmoid(prediction)


        prediction = prediction.contiguous().view(
            prediction.size(0),
            -1
        )


        target = target.contiguous().view(
            target.size(0),
            -1
        )



        intersection = torch.sum(
            prediction * target,
            dim=1
        )


        dice = (

            2.0 * intersection
            +
            self.smooth

        ) / (

            torch.sum(prediction, dim=1)

            +

            torch.sum(target, dim=1)

            +

            self.smooth

        )


        return (1 - dice).mean()





# ============================================================
# BOUNDARY EXTRACTION
# ============================================================


class BoundaryExtractor(nn.Module):


    def __init__(self):

        super().__init__()



        # Laplacian edge detector

        kernel = torch.tensor(

            [
                [
                    [
                        0, 1, 0
                    ],

                    [
                        1, -4, 1
                    ],

                    [
                        0, 1, 0
                    ]

                ]
            ],

            dtype=torch.float32

        )


        # Convert:
        # [1,3,3]
        #
        # to:
        # [1,1,3,3]

        kernel = kernel.unsqueeze(0)



        self.register_buffer(

            "kernel",

            kernel

        )





    def forward(self, x):


        # IMPORTANT:
        # Move kernel to same device and dtype
        # as input tensor

        kernel = self.kernel.to(

            device=x.device,

            dtype=x.dtype

        )



        boundary = F.conv2d(

            x,

            kernel,

            padding=1

        )



        boundary = torch.abs(

            boundary

        )



        return boundary





# ============================================================
# BOUNDARY LOSS
# ============================================================


class BoundaryLoss(nn.Module):


    def __init__(self):

        super().__init__()


        self.extractor = BoundaryExtractor()





    def forward(self, prediction, target):


        prediction = torch.sigmoid(

            prediction

        )



        pred_boundary = self.extractor(

            prediction

        )


        target_boundary = self.extractor(

            target

        )



        loss = F.mse_loss(

            pred_boundary,

            target_boundary

        )



        return loss





# ============================================================
# COMPLETE MV-TRANSUNET LOSS
# ============================================================


class MVTransUNetLoss(nn.Module):


    def __init__(

        self,

        alpha=0.4,

        beta=0.4,

        gamma=0.2

    ):


        super().__init__()



        self.alpha = alpha

        self.beta = beta

        self.gamma = gamma



        self.bce = nn.BCEWithLogitsLoss()


        self.dice = SoftDiceLoss()


        self.boundary = BoundaryLoss()





    def forward(

        self,

        prediction,

        target

    ):



        bce_loss = self.bce(

            prediction,

            target

        )



        dice_loss = self.dice(

            prediction,

            target

        )



        boundary_loss = self.boundary(

            prediction,

            target

        )



        total_loss = (

            self.alpha * bce_loss

            +

            self.beta * dice_loss

            +

            self.gamma * boundary_loss

        )



        return {


            "total_loss":

                total_loss,


            "bce_loss":

                bce_loss,


            "dice_loss":

                dice_loss,


            "boundary_loss":

                boundary_loss

        }





# ============================================================
# LOCAL TEST
# ============================================================


if __name__ == "__main__":


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )



    print("Device:", device)



    prediction = torch.randn(

        2,

        1,

        256,

        256,

        device=device,

        requires_grad=True

    )



    target = torch.randint(

        0,

        2,

        (

            2,

            1,

            256,

            256

        ),

        device=device

    ).float()



    criterion = MVTransUNetLoss().to(device)



    losses = criterion(

        prediction,

        target

    )



    print("\nLoss Results")

    print("----------------")



    for name,value in losses.items():

        print(

            name,

            ":",

            value.item()

        )



    losses["total_loss"].backward()



    print("\nBackward pass successful")