"""
MV-TransUNet Boundary-Aware Deep-Supervision Losses

Components:
1. Binary Cross-Entropy Loss
2. Soft Dice Loss
3. Laplacian Boundary Loss
4. Boundary-Aware Joint Loss
5. Multi-Scale Deep-Supervision Loss

Single-output formulation:

    L_joint =
        alpha * L_BCE
        + beta * L_Dice
        + gamma * L_Boundary

Deep-supervision formulation:

    L_total =
        L_main
        + w1 * L_aux1
        + w2 * L_aux2
        + w3 * L_aux3

Default auxiliary weights:

    w1 = 0.10
    w2 = 0.20
    w3 = 0.30

Features:
- CUDA compatible
- Automatic mixed-precision compatible
- Supports normal tensor predictions
- Supports deep-supervision dictionaries
- Validates output and target shapes
- Reports main and auxiliary loss components
"""


from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TYPE DEFINITIONS
# ============================================================

PredictionType = Union[
    torch.Tensor,
    Dict[
        str,
        Union[
            torch.Tensor,
            List[torch.Tensor],
        ],
    ],
]


# ============================================================
# SOFT DICE LOSS
# ============================================================

class SoftDiceLoss(nn.Module):
    """
    Differentiable Dice loss operating on segmentation logits.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError(
                "smooth must be greater than zero."
            )

        self.smooth = float(smooth)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate mean soft Dice loss across the batch.

        Args:
            prediction:
                Raw segmentation logits with shape [B, C, H, W].

            target:
                Binary ground-truth mask with shape [B, C, H, W].
        """

        if prediction.shape != target.shape:
            raise ValueError(
                "SoftDiceLoss shape mismatch: "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}."
            )

        probability = torch.sigmoid(
            prediction
        )

        probability = probability.contiguous().reshape(
            probability.size(0),
            -1,
        )

        target = target.contiguous().reshape(
            target.size(0),
            -1,
        )

        intersection = torch.sum(
            probability * target,
            dim=1,
        )

        probability_sum = torch.sum(
            probability,
            dim=1,
        )

        target_sum = torch.sum(
            target,
            dim=1,
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probability_sum
            + target_sum
            + self.smooth
        )

        return (
            1.0 - dice
        ).mean()


# ============================================================
# LAPLACIAN BOUNDARY EXTRACTION
# ============================================================

class BoundaryExtractor(nn.Module):
    """
    Extract soft object boundaries using a Laplacian kernel.
    """

    def __init__(self) -> None:
        super().__init__()

        kernel = torch.tensor(
            [
                [
                    [
                        0.0,
                        1.0,
                        0.0,
                    ],
                    [
                        1.0,
                        -4.0,
                        1.0,
                    ],
                    [
                        0.0,
                        1.0,
                        0.0,
                    ],
                ]
            ],
            dtype=torch.float32,
        )

        # Kernel shape:
        # [out_channels=1, in_channels=1, height=3, width=3]
        kernel = kernel.unsqueeze(0)

        self.register_buffer(
            "kernel",
            kernel,
            persistent=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract absolute Laplacian boundary responses.
        """

        if x.ndim != 4:
            raise ValueError(
                "BoundaryExtractor expects a four-dimensional "
                "tensor with shape [B, C, H, W]."
            )

        if x.shape[1] != 1:
            raise ValueError(
                "BoundaryExtractor currently supports one-channel "
                f"masks, but received {x.shape[1]} channels."
            )

        kernel = self.kernel.to(
            device=x.device,
            dtype=x.dtype,
        )

        boundary = F.conv2d(
            input=x,
            weight=kernel,
            bias=None,
            stride=1,
            padding=1,
        )

        return torch.abs(
            boundary
        )


# ============================================================
# BOUNDARY LOSS
# ============================================================

class BoundaryLoss(nn.Module):
    """
    Penalize differences between predicted and target boundaries.
    """

    def __init__(self) -> None:
        super().__init__()

        self.extractor = BoundaryExtractor()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate mean-squared boundary discrepancy.
        """

        if prediction.shape != target.shape:
            raise ValueError(
                "BoundaryLoss shape mismatch: "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}."
            )

        probability = torch.sigmoid(
            prediction
        )

        prediction_boundary = self.extractor(
            probability
        )

        target_boundary = self.extractor(
            target
        )

        return F.mse_loss(
            prediction_boundary,
            target_boundary,
        )


# ============================================================
# MV-TRANSUNET JOINT LOSS
# ============================================================

class MVTransUNetLoss(nn.Module):
    """
    Boundary-aware joint loss with optional deep supervision.

    The module accepts either:

    1. A single segmentation-logit tensor.

    2. A dictionary containing:
        - main_output
        - auxiliary_outputs

    The main output is always optimized with full weight 1.0.
    Auxiliary outputs are weighted using auxiliary_weights.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        auxiliary_weights: Sequence[float] = (
            0.10,
            0.20,
            0.30,
        ),
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()

        if alpha < 0:
            raise ValueError(
                "alpha cannot be negative."
            )

        if beta < 0:
            raise ValueError(
                "beta cannot be negative."
            )

        if gamma < 0:
            raise ValueError(
                "gamma cannot be negative."
            )

        if alpha + beta + gamma <= 0:
            raise ValueError(
                "At least one primary loss weight must be positive."
            )

        if len(auxiliary_weights) == 0:
            raise ValueError(
                "auxiliary_weights cannot be empty."
            )

        if any(
            weight < 0
            for weight in auxiliary_weights
        ):
            raise ValueError(
                "Auxiliary loss weights cannot be negative."
            )

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.auxiliary_weights = tuple(
            float(weight)
            for weight in auxiliary_weights
        )

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = SoftDiceLoss(
            smooth=smooth
        )

        self.boundary = BoundaryLoss()

    def _validate_tensor_pair(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        output_name: str,
    ) -> None:
        """
        Validate one prediction-target pair.
        """

        if not torch.is_tensor(prediction):
            raise TypeError(
                f"{output_name} must be a torch.Tensor."
            )

        if not torch.is_tensor(target):
            raise TypeError(
                "target must be a torch.Tensor."
            )

        if prediction.ndim != 4:
            raise ValueError(
                f"{output_name} must have shape [B, C, H, W], "
                f"but received {tuple(prediction.shape)}."
            )

        if target.ndim != 4:
            raise ValueError(
                "target must have shape [B, C, H, W], "
                f"but received {tuple(target.shape)}."
            )

        if prediction.shape != target.shape:
            raise ValueError(
                f"{output_name} and target shapes do not match: "
                f"{tuple(prediction.shape)} versus "
                f"{tuple(target.shape)}."
            )

    def _compute_single_output_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        output_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BCE, Dice, boundary, and joint losses for one output.
        """

        self._validate_tensor_pair(
            prediction=prediction,
            target=target,
            output_name=output_name,
        )

        bce_loss = self.bce(
            prediction,
            target,
        )

        dice_loss = self.dice(
            prediction,
            target,
        )

        boundary_loss = self.boundary(
            prediction,
            target,
        )

        joint_loss = (
            self.alpha * bce_loss
            + self.beta * dice_loss
            + self.gamma * boundary_loss
        )

        return {
            "joint_loss": joint_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "boundary_loss": boundary_loss,
        }

    def _extract_deep_supervision_outputs(
        self,
        prediction: Dict,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Validate and extract main and auxiliary predictions.
        """

        if "main_output" not in prediction:
            raise KeyError(
                "Deep-supervision prediction dictionary is missing "
                "'main_output'."
            )

        if "auxiliary_outputs" not in prediction:
            raise KeyError(
                "Deep-supervision prediction dictionary is missing "
                "'auxiliary_outputs'."
            )

        main_output = prediction[
            "main_output"
        ]

        auxiliary_outputs = prediction[
            "auxiliary_outputs"
        ]

        if not isinstance(
            auxiliary_outputs,
            (list, tuple),
        ):
            raise TypeError(
                "'auxiliary_outputs' must be a list or tuple."
            )

        auxiliary_outputs = list(
            auxiliary_outputs
        )

        if len(auxiliary_outputs) != len(
            self.auxiliary_weights
        ):
            raise ValueError(
                "The number of auxiliary outputs must match the "
                "number of auxiliary weights. "
                f"Received {len(auxiliary_outputs)} outputs and "
                f"{len(self.auxiliary_weights)} weights."
            )

        return (
            main_output,
            auxiliary_outputs,
        )

    def forward(
        self,
        prediction: PredictionType,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute baseline or deep-supervision loss.

        Returned keys always include:
            total_loss
            bce_loss
            dice_loss
            boundary_loss
            main_loss
            auxiliary_loss

        The bce_loss, dice_loss, and boundary_loss fields correspond
        to the main segmentation output, making them compatible with
        the current training logger.
        """

        target = target.float()

        # ----------------------------------------------------
        # Standard single-output mode
        # ----------------------------------------------------

        if torch.is_tensor(
            prediction
        ):
            main_losses = (
                self._compute_single_output_loss(
                    prediction=prediction,
                    target=target,
                    output_name="prediction",
                )
            )

            zero_loss = (
                main_losses["joint_loss"]
                * 0.0
            )

            return {
                "total_loss": main_losses[
                    "joint_loss"
                ],
                "main_loss": main_losses[
                    "joint_loss"
                ],
                "auxiliary_loss": zero_loss,
                "bce_loss": main_losses[
                    "bce_loss"
                ],
                "dice_loss": main_losses[
                    "dice_loss"
                ],
                "boundary_loss": main_losses[
                    "boundary_loss"
                ],
            }

        # ----------------------------------------------------
        # Deep-supervision mode
        # ----------------------------------------------------

        if not isinstance(
            prediction,
            dict,
        ):
            raise TypeError(
                "prediction must be either a tensor or a dictionary."
            )

        (
            main_output,
            auxiliary_outputs,
        ) = self._extract_deep_supervision_outputs(
            prediction
        )

        main_losses = (
            self._compute_single_output_loss(
                prediction=main_output,
                target=target,
                output_name="main_output",
            )
        )

        weighted_auxiliary_loss = (
            main_losses["joint_loss"]
            * 0.0
        )

        unweighted_auxiliary_loss = (
            main_losses["joint_loss"]
            * 0.0
        )

        auxiliary_bce_loss = (
            main_losses["bce_loss"]
            * 0.0
        )

        auxiliary_dice_loss = (
            main_losses["dice_loss"]
            * 0.0
        )

        auxiliary_boundary_loss = (
            main_losses["boundary_loss"]
            * 0.0
        )

        output_dictionary: Dict[
            str,
            torch.Tensor,
        ] = {}

        for index, (
            auxiliary_output,
            auxiliary_weight,
        ) in enumerate(
            zip(
                auxiliary_outputs,
                self.auxiliary_weights,
            ),
            start=1,
        ):
            auxiliary_losses = (
                self._compute_single_output_loss(
                    prediction=auxiliary_output,
                    target=target,
                    output_name=(
                        f"auxiliary_output_{index}"
                    ),
                )
            )

            weighted_loss = (
                auxiliary_weight
                * auxiliary_losses[
                    "joint_loss"
                ]
            )

            weighted_auxiliary_loss = (
                weighted_auxiliary_loss
                + weighted_loss
            )

            unweighted_auxiliary_loss = (
                unweighted_auxiliary_loss
                + auxiliary_losses[
                    "joint_loss"
                ]
            )

            auxiliary_bce_loss = (
                auxiliary_bce_loss
                + auxiliary_losses[
                    "bce_loss"
                ]
            )

            auxiliary_dice_loss = (
                auxiliary_dice_loss
                + auxiliary_losses[
                    "dice_loss"
                ]
            )

            auxiliary_boundary_loss = (
                auxiliary_boundary_loss
                + auxiliary_losses[
                    "boundary_loss"
                ]
            )

            output_dictionary[
                f"auxiliary_{index}_loss"
            ] = auxiliary_losses[
                "joint_loss"
            ]

            output_dictionary[
                f"auxiliary_{index}_weighted_loss"
            ] = weighted_loss

            output_dictionary[
                f"auxiliary_{index}_bce_loss"
            ] = auxiliary_losses[
                "bce_loss"
            ]

            output_dictionary[
                f"auxiliary_{index}_dice_loss"
            ] = auxiliary_losses[
                "dice_loss"
            ]

            output_dictionary[
                f"auxiliary_{index}_boundary_loss"
            ] = auxiliary_losses[
                "boundary_loss"
            ]

        number_of_auxiliary_outputs = float(
            len(auxiliary_outputs)
        )

        mean_auxiliary_loss = (
            unweighted_auxiliary_loss
            / number_of_auxiliary_outputs
        )

        mean_auxiliary_bce_loss = (
            auxiliary_bce_loss
            / number_of_auxiliary_outputs
        )

        mean_auxiliary_dice_loss = (
            auxiliary_dice_loss
            / number_of_auxiliary_outputs
        )

        mean_auxiliary_boundary_loss = (
            auxiliary_boundary_loss
            / number_of_auxiliary_outputs
        )

        total_loss = (
            main_losses["joint_loss"]
            + weighted_auxiliary_loss
        )

        output_dictionary.update(
            {
                "total_loss": total_loss,

                "main_loss": main_losses[
                    "joint_loss"
                ],

                "auxiliary_loss": (
                    weighted_auxiliary_loss
                ),

                "mean_auxiliary_loss": (
                    mean_auxiliary_loss
                ),

                "bce_loss": main_losses[
                    "bce_loss"
                ],

                "dice_loss": main_losses[
                    "dice_loss"
                ],

                "boundary_loss": main_losses[
                    "boundary_loss"
                ],

                "mean_auxiliary_bce_loss": (
                    mean_auxiliary_bce_loss
                ),

                "mean_auxiliary_dice_loss": (
                    mean_auxiliary_dice_loss
                ),

                "mean_auxiliary_boundary_loss": (
                    mean_auxiliary_boundary_loss
                ),
            }
        )

        return output_dictionary


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print("=" * 70)
    print("MV-TransUNet Deep-Supervision Loss Test")
    print("=" * 70)
    print("Device:", device)

    target = torch.randint(
        low=0,
        high=2,
        size=(
            2,
            1,
            256,
            256,
        ),
        device=device,
    ).float()

    criterion = MVTransUNetLoss(
        alpha=0.4,
        beta=0.4,
        gamma=0.2,
        auxiliary_weights=(
            0.10,
            0.20,
            0.30,
        ),
    ).to(device)

    # --------------------------------------------------------
    # Standard single-output test
    # --------------------------------------------------------

    standard_prediction = torch.randn(
        2,
        1,
        256,
        256,
        device=device,
        requires_grad=True,
    )

    standard_losses = criterion(
        standard_prediction,
        target,
    )

    print()
    print("=" * 70)
    print("Standard Output Test")
    print("=" * 70)

    for loss_name in [
        "total_loss",
        "main_loss",
        "auxiliary_loss",
        "bce_loss",
        "dice_loss",
        "boundary_loss",
    ]:
        print(
            f"{loss_name}:",
            standard_losses[
                loss_name
            ].item(),
        )

    standard_losses[
        "total_loss"
    ].backward()

    if standard_prediction.grad is None:
        raise RuntimeError(
            "Standard-output backward pass failed."
        )

    print(
        "Standard-output backward pass successful."
    )

    # --------------------------------------------------------
    # Deep-supervision output test
    # --------------------------------------------------------

    main_prediction = torch.randn(
        2,
        1,
        256,
        256,
        device=device,
        requires_grad=True,
    )

    auxiliary_prediction1 = torch.randn(
        2,
        1,
        256,
        256,
        device=device,
        requires_grad=True,
    )

    auxiliary_prediction2 = torch.randn(
        2,
        1,
        256,
        256,
        device=device,
        requires_grad=True,
    )

    auxiliary_prediction3 = torch.randn(
        2,
        1,
        256,
        256,
        device=device,
        requires_grad=True,
    )

    deep_supervision_prediction = {
        "main_output": main_prediction,

        "auxiliary_outputs": [
            auxiliary_prediction1,
            auxiliary_prediction2,
            auxiliary_prediction3,
        ],
    }

    deep_supervision_losses = criterion(
        deep_supervision_prediction,
        target,
    )

    print()
    print("=" * 70)
    print("Deep Supervision Test")
    print("=" * 70)

    for loss_name in [
        "total_loss",
        "main_loss",
        "auxiliary_loss",
        "bce_loss",
        "dice_loss",
        "boundary_loss",
        "auxiliary_1_loss",
        "auxiliary_2_loss",
        "auxiliary_3_loss",
    ]:
        print(
            f"{loss_name}:",
            deep_supervision_losses[
                loss_name
            ].item(),
        )

    deep_supervision_losses[
        "total_loss"
    ].backward()

    prediction_tensors = [
        main_prediction,
        auxiliary_prediction1,
        auxiliary_prediction2,
        auxiliary_prediction3,
    ]

    for index, prediction_tensor in enumerate(
        prediction_tensors,
        start=1,
    ):
        if prediction_tensor.grad is None:
            raise RuntimeError(
                f"Gradient was not computed for prediction {index}."
            )

    print(
        "Deep-supervision backward pass successful."
    )

    print()
    print(
        "All loss tests passed successfully."
    )