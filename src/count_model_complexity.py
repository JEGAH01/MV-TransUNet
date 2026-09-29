"""
Trainable parameter count and FLOP (multiply-accumulate) comparison between
the baseline MV-TransUNet (SCVAM) and the full topology-aware model
(A8, SCVAM) at a single 256x256 inference-time forward pass.

Both models are built with the exact same build_model() functions used by
the real evaluators (imported directly from evaluate_baseline_v2.py /
evaluate_topology_v2.py, not reimplemented), so the architectures profiled
here are guaranteed identical to the ones actually trained and evaluated.

The trainable-parameter counts printed here are a cross-check against the
"Trainable parameters" line already printed at the start of training by
src/train_baseline_matched.py and run_ablation.py (recorded in the actual
training logs as 131,454,838 for the SCVAM baseline family and
131,947,628 for the topology A8 SCVAM family). If this script's count
does not match those logged numbers, something about the config passed
here does not match the config actually used for training, and that
should be investigated before trusting the FLOP number.

FLOP counting depends on thop, which only has hooks for standard PyTorch
modules (Conv2d, Linear, BatchNorm, etc.). Any custom operation that is
not built from those (e.g. hand-written attention/matmul code that isn't
wrapped in nn.Linear) will not be traced, which silently UNDER-counts
MACs for that layer. This script cross-checks thop's own traced parameter
count against model.parameters() and prints a warning if they diverge by
more than 1%, so an undercount is visible rather than silently reported
as exact.

Requires: pip install thop

Example:
    python -m src.count_model_complexity ^
        --baseline-config config_baseline_matched_topology_scvam_seed42.yaml ^
        --topology-config config_topology_ablation_a8_scvam.yaml ^
        --input-size 256
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch

from src.evaluate_baseline_v2 import build_model as build_baseline_model
from src.evaluate_baseline_v2 import load_config as load_baseline_config
from src.evaluate_topology_v2 import build_model as build_topology_model
from src.evaluate_topology_v2 import load_config as load_topology_config


# ============================================================
# ARGUMENTS
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameter and FLOP comparison: baseline vs topology (A8)."
    )

    parser.add_argument(
        "--baseline-config",
        type=str,
        default="config_baseline_matched_topology_scvam_seed42.yaml",
    )

    parser.add_argument(
        "--topology-config",
        type=str,
        default="config_topology_ablation_a8_scvam.yaml",
    )

    parser.add_argument("--input-size", type=int, default=256)

    return parser.parse_args()


# ============================================================
# MODEL PROFILING
# ============================================================


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


class TopologyForwardWrapper(torch.nn.Module):
    """
    Fixes return_topology=True on the wrapped forward call so thop's single
    positional-input tracing can execute the full multi-head forward pass
    (segmentation + skeleton + endpoint + junction), not just the default
    call signature.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model(x, return_topology=True)


def profile_model(
    model: torch.nn.Module,
    input_size: int,
) -> Tuple[float, int]:
    try:
        from thop import profile
    except ImportError as error:
        raise ImportError(
            "thop is required for FLOP counting. "
            "Install with: pip install thop"
        ) from error

    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        macs, params = profile(
            model,
            inputs=(dummy_input,),
            verbose=False,
        )

    return float(macs), int(params)


def report_model(
    label: str,
    trainable_parameters: int,
    thop_parameters: int,
    macs: float,
) -> None:
    print()
    print("-" * 78)
    print(label)
    print("-" * 78)
    print(f"Trainable parameters (model.parameters()): {trainable_parameters:,}")
    print(f"Trainable parameters (thop traced):         {thop_parameters:,}")
    print(f"MACs (multiply-accumulate ops):              {macs:,.0f}")
    print(f"GMACs:                                       {macs / 1e9:.4f}")
    print(f"Approx. GFLOPs (2 x MACs):                    {2 * macs / 1e9:.4f}")

    relative_difference = (
        abs(thop_parameters - trainable_parameters) / max(trainable_parameters, 1)
    )
    if relative_difference > 0.01:
        print(
            "WARNING: thop's traced parameter count differs from "
            "model.parameters() by more than 1% "
            f"({thop_parameters:,} vs {trainable_parameters:,}). "
            "Some custom layers were likely not traced by thop -- treat "
            "the MAC/FLOP number above as a LOWER BOUND, not an exact count."
        )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    arguments = parse_arguments()

    print("=" * 78)
    print("Model complexity comparison: baseline (SCVAM) vs topology A8 (SCVAM)")
    print("=" * 78)
    print("Input size:", arguments.input_size, "x", arguments.input_size)
    print("Baseline config:", arguments.baseline_config)
    print("Topology config:", arguments.topology_config)

    baseline_config = load_baseline_config(arguments.baseline_config)
    baseline_model = build_baseline_model(baseline_config)
    baseline_trainable = count_trainable_parameters(baseline_model)
    baseline_macs, baseline_thop_params = profile_model(
        baseline_model, arguments.input_size
    )

    topology_config = load_topology_config(arguments.topology_config)
    topology_model = build_topology_model(topology_config)
    topology_trainable = count_trainable_parameters(topology_model)
    wrapped_topology_model = TopologyForwardWrapper(topology_model)
    topology_macs, topology_thop_params = profile_model(
        wrapped_topology_model, arguments.input_size
    )

    report_model(
        "Baseline (SCVAM)",
        baseline_trainable,
        baseline_thop_params,
        baseline_macs,
    )

    report_model(
        "Topology A8 (SCVAM)",
        topology_trainable,
        topology_thop_params,
        topology_macs,
    )

    print()
    print("-" * 78)
    print("Delta (topology - baseline)")
    print("-" * 78)

    parameter_delta = topology_trainable - baseline_trainable
    macs_delta = topology_macs - baseline_macs

    print(
        f"Extra trainable parameters: {parameter_delta:,} "
        f"({100.0 * parameter_delta / baseline_trainable:.2f}% of baseline)"
    )
    print(
        f"Extra MACs: {macs_delta:,.0f} "
        f"({100.0 * macs_delta / baseline_macs:.2f}% of baseline)"
    )


if __name__ == "__main__":
    main()
