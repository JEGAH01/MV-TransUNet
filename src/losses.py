"""
MV-TransUNet Loss Functions

Boundary-Aware Joint Loss

Components:

1. BCE Loss
2. Dice Loss
3. Boundary Loss

Total:

Loss =
0.4 BCE
+
0.4 Dice
+
0.2 Boundary


Author:
MV-TransUNet Research Project
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
# DICE LOSS
# ============================================================


class DiceLoss(nn.Module):


    def __init__(self, smooth=1e-6):

        super().__init__()

        self.smooth = smooth



    def forward(self, predictions, targets):


        predictions = torch.sigmoid(predictions)



        predictions = predictions.contiguous()

        targets = targets.contiguous()



        intersection = (

            predictions * targets

        ).sum(dim=(2,3))



        union = (

            predictions.sum(dim=(2,3))

            +

            targets.sum(dim=(2,3))

        )



        dice = (

            (2. * intersection + self.smooth)

            /

            (union + self.smooth)

        )



        loss = 1 - dice.mean()



        return loss







# ============================================================
# BOUNDARY EXTRACTION
# ============================================================


class BoundaryExtractor(nn.Module):


    """
    Extracts vessel boundaries using
    Laplacian edge operator.
    """



    def __init__(self):

        super().__init__()



        kernel = torch.tensor(

            [

                [-1,-1,-1],

                [-1, 8,-1],

                [-1,-1,-1]

            ],

            dtype=torch.float32

        )



        self.register_buffer(

            "kernel",

            kernel.view(

                1,

                1,

                3,

                3

            )

        )





    def forward(self,x):


        boundary = F.conv2d(

            x,

            self.kernel,

            padding=1

        )


        return torch.abs(boundary)







# ============================================================
# BOUNDARY LOSS
# ============================================================


class BoundaryLoss(nn.Module):


    def __init__(self):

        super().__init__()



        self.extractor = BoundaryExtractor()





    def forward(self,predictions,targets):


        predictions=torch.sigmoid(

            predictions

        )



        pred_boundary = self.extractor(

            predictions

        )



        target_boundary = self.extractor(

            targets

        )



        loss = F.l1_loss(

            pred_boundary,

            target_boundary

        )



        return loss







# ============================================================
# MV-TRANSUNET JOINT LOSS
# ============================================================


class BoundaryAwareLoss(nn.Module):


    def __init__(

        self,

        bce_weight=0.4,

        dice_weight=0.4,

        boundary_weight=0.2

    ):


        super().__init__()



        self.bce_weight = bce_weight

        self.dice_weight = dice_weight

        self.boundary_weight = boundary_weight




        self.bce = nn.BCEWithLogitsLoss()



        self.dice = DiceLoss()



        self.boundary = BoundaryLoss()





    def forward(

        self,

        predictions,

        targets

    ):



        bce_loss = self.bce(

            predictions,

            targets

        )



        dice_loss = self.dice(

            predictions,

            targets

        )



        boundary_loss = self.boundary(

            predictions,

            targets

        )



        total_loss = (

            self.bce_weight*bce_loss

            +

            self.dice_weight*dice_loss

            +

            self.boundary_weight*boundary_loss

        )



        return {


            "loss": total_loss,


            "bce": bce_loss,


            "dice": dice_loss,


            "boundary": boundary_loss

        }








# ============================================================
# TEST
# ============================================================


if __name__ == "__main__":


    print(

        "Testing Boundary-Aware Loss"

    )



    prediction=torch.randn(

        2,

        1,

        256,

        256

    )



    target=torch.randint(

        0,

        2,

        (

            2,

            1,

            256,

            256

        )

    ).float()



    criterion=BoundaryAwareLoss()



    output=criterion(

        prediction,

        target

    )



    print()

    for key,value in output.items():

        print(

            key,

            ":",

            value.item()

        )