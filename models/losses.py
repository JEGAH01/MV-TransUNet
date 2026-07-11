"""
MV-TransUNet Experiment B Loss Functions

Experiment B:
Focal Tversky Loss
+
Boundary Loss
+
clDice Topology Loss
+
Deep Supervision


Designed for:
Retinal Vessel Segmentation


Purpose:
Improve:
- thin vessel recall
- vessel connectivity
- topology preservation
- class imbalance handling


Formula:

Main Loss:

L =
0.5 * FocalTversky
+
0.3 * Boundary
+
0.2 * clDice


Deep Supervision:

Total =
Main
+
0.1 Aux1
+
0.2 Aux2
+
0.3 Aux3


AMP / CUDA Compatible
"""


import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Dict, Tuple





# ============================================================
# FOCAL TVERSKY LOSS
# ============================================================


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss

    Designed for highly imbalanced segmentation.

    Penalizes false negatives more heavily.

    Useful for:
    - thin vessels
    - micro vessels
    - small structures


    Formula:

    Tversky =
        TP /
        (TP + alpha*FP + beta*FN)


    Focal:

    (1 - Tversky)^gamma
    """


    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        gamma=0.75,
        smooth=1e-6,
    ):

        super().__init__()

        self.alpha = alpha

        self.beta = beta

        self.gamma = gamma

        self.smooth = smooth



    def forward(
        self,
        prediction,
        target,
    ):


        prediction = torch.sigmoid(
            prediction
        )


        prediction = prediction.contiguous().view(
            prediction.size(0),
            -1
        )


        target = target.contiguous().view(
            target.size(0),
            -1
        )



        true_positive = torch.sum(
            prediction * target,
            dim=1,
        )


        false_positive = torch.sum(
            prediction * (1 - target),
            dim=1,
        )


        false_negative = torch.sum(
            (1 - prediction) * target,
            dim=1,
        )



        tversky = (

            true_positive
            +
            self.smooth

        ) / (

            true_positive

            +

            self.alpha * false_positive

            +

            self.beta * false_negative

            +

            self.smooth

        )



        focal_tversky = torch.pow(
            1 - tversky,
            self.gamma
        )


        return focal_tversky.mean()







# ============================================================
# SOFT DICE LOSS
# ============================================================


class SoftDiceLoss(nn.Module):
    """
    Kept for logging compatibility.

    Used as a metric component.
    """


    def __init__(
        self,
        smooth=1e-6,
    ):

        super().__init__()

        self.smooth = smooth



    def forward(
        self,
        prediction,
        target,
    ):


        prediction = torch.sigmoid(
            prediction
        )


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
            dim=1,
        )



        dice = (

            2.0 * intersection

            +

            self.smooth

        ) / (

            torch.sum(
                prediction,
                dim=1
            )

            +

            torch.sum(
                target,
                dim=1
            )

            +

            self.smooth

        )



        return (
            1 - dice
        ).mean()







# ============================================================
# DIFFERENTIABLE SKELETONIZATION
# ============================================================


def soft_erode(
    img,
):

    p1 = -F.max_pool2d(
        -img,
        kernel_size=(3,1),
        stride=(1,1),
        padding=(1,0),
    )


    p2 = -F.max_pool2d(
        -img,
        kernel_size=(1,3),
        stride=(1,1),
        padding=(0,1),
    )


    return torch.min(
        p1,
        p2,
    )




def soft_dilate(
    img,
):

    return F.max_pool2d(
        img,
        kernel_size=3,
        stride=1,
        padding=1,
    )




def soft_open(
    img,
):

    return soft_dilate(
        soft_erode(img)
    )




def soft_skeletonize(
    img,
        iterations=10,
):

    skeleton = F.relu(
        img - soft_open(img)
    )


    for _ in range(iterations):

        img = soft_erode(
            img
        )

        opened = soft_open(
            img
        )

        delta = F.relu(
            img - opened
        )

        skeleton = skeleton + (
            (1 - skeleton)
            *
            delta
        )


    return skeleton







# ============================================================
# SOFT CLDICE LOSS
# ============================================================


class SoftCLDiceLoss(nn.Module):
    """
    Topology-aware centerline Dice loss.

    Improves:
    - vessel continuity
    - thin vessel preservation
    - connectivity
    """


    def __init__(
        self,
        iterations=10,
        smooth=1e-6,
    ):

        super().__init__()

        self.iterations = iterations

        self.smooth = smooth



    def forward(
        self,
        prediction,
        target,
    ):


        prediction = torch.sigmoid(
            prediction
        )


        pred_skeleton = soft_skeletonize(
            prediction,
            self.iterations,
        )


        target_skeleton = soft_skeletonize(
            target,
            self.iterations,
        )



        tprec = (

            torch.sum(
                pred_skeleton * target
            )

            +

            self.smooth

        ) / (

            torch.sum(
                pred_skeleton
            )

            +

            self.smooth

        )


        tsens = (

            torch.sum(
                target_skeleton * prediction
            )

            +

            self.smooth

        ) / (

            torch.sum(
                target_skeleton
            )

            +

            self.smooth

        )



        cldice = (

            2
            *
            tprec
            *
            tsens

        ) / (

            tprec
            +
            tsens
            +
            self.smooth

        )


        return 1 - cldice
    # ============================================================
# BOUNDARY EXTRACTION
# ============================================================


class BoundaryExtractor(nn.Module):
    """
    Laplacian boundary extractor.
    """


    def __init__(self):

        super().__init__()


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
            dtype=torch.float32,
        )


        kernel = kernel.unsqueeze(0)


        self.register_buffer(
            "kernel",
            kernel,
        )



    def forward(
        self,
        x,
    ):


        kernel = self.kernel.to(
            device=x.device,
            dtype=x.dtype,
        )


        boundary = F.conv2d(
            x,
            kernel,
            padding=1,
        )


        return torch.abs(
            boundary
        )






# ============================================================
# BOUNDARY LOSS
# ============================================================


class BoundaryLoss(nn.Module):


    def __init__(self):

        super().__init__()

        self.extractor = BoundaryExtractor()



    def forward(
        self,
        prediction,
        target,
    ):


        prediction = torch.sigmoid(
            prediction
        )


        pred_boundary = self.extractor(
            prediction
        )


        target_boundary = self.extractor(
            target
        )


        return F.mse_loss(
            pred_boundary,
            target_boundary,
        )








# ============================================================
# EXPERIMENT B COMPLETE LOSS
# ============================================================


class MVTransUNetLoss(nn.Module):
    """
    Experiment B:

    Main:

    0.5 Focal Tversky
    0.3 Boundary
    0.2 clDice


    Supports:

    Tensor output:

        prediction


    Deep supervision output:

        {
            "main_output": tensor,
            "auxiliary_outputs": [
                tensor,
                tensor,
                tensor
            ]
        }

    """


    def __init__(
        self,

        focal_tversky_weight=0.5,

        boundary_weight=0.3,

        cldice_weight=0.2,

        auxiliary_weights=(
            0.10,
            0.20,
            0.30,
        ),

    ):


        super().__init__()



        self.focal_tversky_weight = (
            focal_tversky_weight
        )


        self.boundary_weight = (
            boundary_weight
        )


        self.cldice_weight = (
            cldice_weight
        )


        self.auxiliary_weights = (
            auxiliary_weights
        )



        self.focal_tversky = (
            FocalTverskyLoss()
        )


        self.boundary = (
            BoundaryLoss()
        )


        self.cldice = (
            SoftCLDiceLoss()
        )


        self.dice = (
            SoftDiceLoss()
        )



    # --------------------------------------------------------
    # MAIN LOSS
    # --------------------------------------------------------


    def compute_main_loss(
        self,
        prediction,
        target,
    ):


        focal_tversky_loss = (
            self.focal_tversky(
                prediction,
                target,
            )
        )


        boundary_loss = (
            self.boundary(
                prediction,
                target,
            )
        )


        cldice_loss = (
            self.cldice(
                prediction,
                target,
            )
        )



        total = (

            self.focal_tversky_weight
            *
            focal_tversky_loss


            +

            self.boundary_weight
            *
            boundary_loss


            +

            self.cldice_weight
            *
            cldice_loss

        )



        return {

            "loss": total,

            "focal_tversky_loss":
                focal_tversky_loss,

            "boundary_loss":
                boundary_loss,

            "cldice_loss":
                cldice_loss,

        }



    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------


    def forward(
        self,
        prediction,
        target,
    ):



        # ============================================
        # STANDARD OUTPUT
        # ============================================


        if torch.is_tensor(
            prediction
        ):


            main_losses = (
                self.compute_main_loss(
                    prediction,
                    target,
                )
            )


            total_loss = (
                main_losses["loss"]
            )


            return {

                "total_loss":
                    total_loss,


                "main_loss":
                    total_loss,


                "auxiliary_loss":
                    torch.zeros_like(
                        total_loss
                    ),


                # compatibility

                "dice_loss":
                    self.dice(
                        prediction,
                        target,
                    ),


                "boundary_loss":
                    main_losses[
                        "boundary_loss"
                    ],


                "focal_tversky_loss":
                    main_losses[
                        "focal_tversky_loss"
                    ],


                "cldice_loss":
                    main_losses[
                        "cldice_loss"
                    ],
            }



        # ============================================
        # DEEP SUPERVISION OUTPUT
        # ============================================


        if isinstance(
            prediction,
            dict,
        ):


            main_output = prediction[
                "main_output"
            ]


            main_losses = (
                self.compute_main_loss(
                    main_output,
                    target,
                )
            )


            total_loss = (
                main_losses["loss"]
            )


            auxiliary_loss = torch.zeros_like(
                total_loss
            )



            auxiliary_outputs = prediction.get(
                "auxiliary_outputs",
                [],
            )



            for idx, aux_output in enumerate(
                auxiliary_outputs
            ):


                if idx >= len(
                    self.auxiliary_weights
                ):
                    break



                aux_loss = (
                    self.compute_main_loss(
                        aux_output,
                        target,
                    )["loss"]
                )


                auxiliary_loss += (
                    self.auxiliary_weights[idx]
                    *
                    aux_loss
                )



            total = (
                total_loss
                +
                auxiliary_loss
            )



            return {


                "total_loss":
                    total,


                "main_loss":
                    main_losses["loss"],


                "auxiliary_loss":
                    auxiliary_loss,


                # logging compatibility

                "dice_loss":
                    self.dice(
                        main_output,
                        target,
                    ),


                "boundary_loss":
                    main_losses[
                        "boundary_loss"
                    ],


                "focal_tversky_loss":
                    main_losses[
                        "focal_tversky_loss"
                    ],


                "cldice_loss":
                    main_losses[
                        "cldice_loss"
                    ],


            }


        raise TypeError(
            "Prediction must be tensor or dictionary"
        )






# ============================================================
# LOCAL VERIFICATION TEST
# ============================================================


if __name__ == "__main__":


    print("=" * 70)

    print(
        "MV-TransUNet Experiment B Loss Verification"
    )

    print("=" * 70)



    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )


    print(
        "Device:",
        device,
    )



    prediction = torch.randn(
        2,
        1,
        256,
        256,
        device=device,
        requires_grad=True,
    )


    target = torch.randint(
        0,
        2,
        (
            2,
            1,
            256,
            256,
        ),
        device=device,
    ).float()



    criterion = MVTransUNetLoss().to(
        device
    )



    print("\nStandard output test")

    result = criterion(
        prediction,
        target,
    )



    for key,value in result.items():

        if torch.is_tensor(value):

            print(
                key,
                ":",
                value.item(),
            )



    result["total_loss"].backward()


    print(
        "Standard backward pass successful"
    )



    print("\nDeep supervision test")



    deep_prediction = {

        "main_output":
            torch.randn(
                2,
                1,
                256,
                256,
                device=device,
                requires_grad=True,
            ),


        "auxiliary_outputs":
        [

            torch.randn(
                2,
                1,
                256,
                256,
                device=device,
                requires_grad=True,
            ),


            torch.randn(
                2,
                1,
                256,
                256,
                device=device,
                requires_grad=True,
            ),


            torch.randn(
                2,
                1,
                256,
                256,
                device=device,
                requires_grad=True,
            ),

        ]

    }



    deep_result = criterion(
        deep_prediction,
        target,
    )



    print(
        "Deep supervision total loss:",
        deep_result[
            "total_loss"
        ].item(),
    )



    deep_result[
        "total_loss"
    ].backward()



    print(
        "Deep supervision backward pass successful"
    )


    print(
        "\nAll Experiment B loss tests passed."
    )