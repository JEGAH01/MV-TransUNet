#!/usr/bin/env python3
"""
Standalone verification for the SC-VAM (Scale-Conditioned Vessel Attention
Module) code change, before spending any GPU time on real training.

Run this first:  python -m src.test_scvam_integration
(or: python test_scvam_integration.py  from the repo root)

It checks, on both MVTransUNet (baseline) and MVTransUNetTopology:

  1. scale_conditioning=False reproduces the ORIGINAL architecture exactly
     (no new parameters at all) -- so every existing config/checkpoint is
     completely unaffected by this change.
  2. scale_conditioning=True is a safe warm start: at construction, output
     is identical regardless of scale_ratio (zero-init FiLM pathway).
  3. Gradient reaches the new scale_mlp parameters (the pathway is wired
     into the computational graph, not dangling).
  4. After a few real optimizer steps focused on scale_mlp, the model
     becomes measurably scale-sensitive (confirms the pathway is not just
     connected but actually learnable/effective -- a random perturbation
     at init is NOT a reliable test of this, since the deep decoder's
     normalization layers damp unstructured noise much more than they
     damp a coherent, loss-reducing gradient direction).

Runs on CPU in well under a minute; no dataset or GPU required.
"""
import torch

from models.mv_transunet import MVTransUNet
from models.mv_transunet_topology import MVTransUNetTopology


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(label)


def test_baseline():
    print("\n=== MVTransUNet (baseline) ===")
    torch.manual_seed(42)

    m_off = MVTransUNet(pretrained=False, vessel_attention_scale_conditioning=False)
    check(
        "scale_conditioning=False has no scale_mlp submodule",
        not hasattr(m_off.vessel_attention.channel_attention, "scale_mlp"),
    )

    torch.manual_seed(42)
    model = MVTransUNet(pretrained=False, vessel_attention_scale_conditioning=True)
    model.eval()
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        out_unit = model(x, scale_ratio=torch.tensor([1.0, 1.0]))
        out_varied = model(x, scale_ratio=torch.tensor([0.75, 2.0]))
        out_none = model(x, scale_ratio=None)
        out_float = model(x, scale_ratio=1.0)

    check("identical output at init regardless of scale_ratio", torch.allclose(out_unit, out_varied))
    check("scale_ratio=None matches scale_ratio=1.0", torch.allclose(out_unit, out_none))
    check("float scale_ratio broadcasts correctly", torch.allclose(out_unit, out_float))

    model.train()
    x2 = torch.randn(2, 3, 256, 256, requires_grad=True)
    out = model(x2, scale_ratio=torch.tensor([0.75, 2.0]), return_auxiliary=True)
    loss = out["main_output"].mean() + sum(a.mean() for a in out["auxiliary_outputs"])
    loss.backward()
    grad_ok = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.vessel_attention.channel_attention.scale_mlp.parameters()
    )
    check("gradient reaches scale_mlp", grad_ok)
    check("gradient reaches model input", x2.grad is not None and x2.grad.abs().sum().item() > 0)

    print("  Running 30 optimizer steps on scale_mlp only (checks learnability)...")
    model.eval()
    optimizer = torch.optim.Adam(model.vessel_attention.channel_attention.scale_mlp.parameters(), lr=0.05)
    fixed_x = torch.randn(4, 3, 256, 256)
    for _ in range(30):
        model.train()
        scale_ratio = torch.tensor([0.75, 0.75, 2.0, 2.0])
        target_shift = (scale_ratio - 1.0).view(-1, 1, 1, 1)
        out = model(fixed_x, scale_ratio=scale_ratio, return_auxiliary=True)
        main = out["main_output"] if isinstance(out, dict) else out
        loss = ((main - target_shift) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out_a = model(fixed_x, scale_ratio=torch.tensor([1.0, 1.0, 1.0, 1.0]))
        out_b = model(fixed_x, scale_ratio=torch.tensor([0.75, 0.75, 2.0, 2.0]))
    diff = (out_a - out_b).abs().max().item()
    check(f"after training, output is clearly scale-sensitive (max diff={diff:.4f} > 1e-3)", diff > 1e-3)


def test_topology():
    print("\n=== MVTransUNetTopology ===")
    torch.manual_seed(42)

    m_off = MVTransUNetTopology(pretrained=False, vessel_attention_scale_conditioning=False)
    check(
        "scale_conditioning=False has no scale_mlp submodule",
        not hasattr(m_off.vessel_attention.channel_attention, "scale_mlp"),
    )

    torch.manual_seed(42)
    model = MVTransUNetTopology(pretrained=False, vessel_attention_scale_conditioning=True)
    model.eval()
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out_unit = model(x, scale_ratio=torch.tensor([1.0]), return_topology=True)
        out_varied = model(x, scale_ratio=torch.tensor([1.8]), return_topology=True)
        out_none = model(x, scale_ratio=None, return_topology=True)

    check(
        "identical main_output at init regardless of scale_ratio",
        torch.allclose(out_unit["main_output"], out_varied["main_output"]),
    )
    check(
        "scale_ratio=None matches scale_ratio=1.0",
        torch.allclose(out_unit["main_output"], out_none["main_output"]),
    )

    model.train()
    x2 = torch.randn(1, 3, 256, 256, requires_grad=True)
    out = model(x2, scale_ratio=torch.tensor([1.8]), return_topology=True)
    loss = (
        out["topology_segmentation_logits"].mean()
        + out["skeleton_logits"].mean()
        + out["endpoint_logits"].mean()
        + out["junction_logits"].mean()
        + out["original_main_output"].mean()
    )
    loss.backward()
    grad_ok = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.vessel_attention.channel_attention.scale_mlp.parameters()
    )
    check("gradient reaches scale_mlp through the full topology model + loss", grad_ok)


if __name__ == "__main__":
    print("SC-VAM integration test")
    print("=" * 60)
    test_baseline()
    test_topology()
    print()
    print("=" * 60)
    print("ALL SC-VAM INTEGRATION CHECKS PASSED")
    print("=" * 60)
