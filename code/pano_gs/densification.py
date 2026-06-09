# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""Adaptive density control: gradient tracking, clone/split/prune, opacity reset."""

from __future__ import annotations

import torch

from gaussians import GaussianModel
from utils import inverse_sigmoid, build_rotation_from_quaternion


class DensificationState:
    """Tracks gradient statistics between densification rounds."""

    def __init__(self, num_gaussians: int, device: str = "cuda"):
        self.grad_accum = torch.zeros(num_gaussians, 1, device=device)
        self.denom = torch.zeros(num_gaussians, 1, device=device)

    def reset(self, num_gaussians: int, device: str = "cuda"):
        self.grad_accum = torch.zeros(num_gaussians, 1, device=device)
        self.denom = torch.zeros(num_gaussians, 1, device=device)

    def accumulate(self, means2d: torch.Tensor, visibility_filter: torch.Tensor):
        """Accumulate 2D position gradients for visible Gaussians.

        Args:
            means2d: (1, N, 2) from gsplat meta, must have .grad populated.
            visibility_filter: (N,) bool.
        """
        if means2d.grad is None:
            return
        grad_2d = means2d.grad.squeeze(0)  # (N, 2)
        grad_norm = grad_2d.norm(dim=-1, keepdim=True)  # (N, 1)
        self.grad_accum[visibility_filter] += grad_norm[visibility_filter]
        self.denom[visibility_filter] += 1


def densify_and_prune(
    gaussians: GaussianModel,
    state: DensificationState,
    grad_threshold: float,
    min_opacity: float,
    extent: float,
    max_screen_size: int | None,
    percent_dense: float,
):
    """Run one round of clone + split + prune."""
    grads = state.grad_accum / state.denom.clamp(min=1)
    grads[grads.isnan()] = 0.0
    grads_flat = grads.squeeze(-1)

    N = gaussians.num_gaussians
    scaling = gaussians.scaling  # (N, 3)
    max_scale = scaling.max(dim=1).values

    high_grad = grads_flat >= grad_threshold
    small_enough = max_scale <= percent_dense * extent
    too_large = max_scale > percent_dense * extent

    clone_mask = high_grad & small_enough
    split_mask = high_grad & too_large

    # ---- Clone ----
    if clone_mask.any():
        _clone(gaussians, clone_mask)

    # ---- Split ----
    if split_mask.any():
        _split(gaussians, split_mask)

    # ---- Prune ----
    opacity = torch.sigmoid(gaussians._opacity).squeeze(-1)
    prune_mask = opacity < min_opacity
    if max_screen_size is not None:
        big_screen = gaussians.max_radii2D > max_screen_size
        big_world = max_scale > 0.1 * extent
        prune_mask = prune_mask | big_screen | big_world
    if prune_mask.any():
        _prune(gaussians, prune_mask)

    state.reset(gaussians.num_gaussians, device=gaussians._xyz.device)
    gaussians.max_radii2D = torch.zeros(gaussians.num_gaussians, device=gaussians._xyz.device)


def reset_opacity(gaussians: GaussianModel):
    """Reset all opacities to ~0.01."""
    new_opacity = inverse_sigmoid(
        torch.clamp(torch.sigmoid(gaussians._opacity), max=0.01)
    )
    param = gaussians._replace_param_in_optimizer(new_opacity.data, "opacity")
    gaussians._opacity = param


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clone(g: GaussianModel, mask: torch.Tensor):
    ext = {
        "xyz": g._xyz.data[mask],
        "sh_dc": g._sh_dc.data[mask],
        "sh_rest": g._sh_rest.data[mask],
        "opacity": g._opacity.data[mask],
        "scaling": g._scaling.data[mask],
        "rotation": g._rotation.data[mask],
    }
    out = g._cat_tensors_to_optimizer(ext)
    g._xyz = out["xyz"]
    g._sh_dc = out["sh_dc"]
    g._sh_rest = out["sh_rest"]
    g._opacity = out["opacity"]
    g._scaling = out["scaling"]
    g._rotation = out["rotation"]
    g.max_radii2D = torch.zeros(g.num_gaussians, device=g._xyz.device)


def _split(g: GaussianModel, mask: torch.Tensor, N: int = 2):
    """Split large Gaussians into N smaller ones."""
    stds = g.scaling[mask].repeat(N, 1)
    means = torch.zeros(stds.shape[0], 3, device=stds.device)
    samples = torch.normal(means, stds)
    rots = build_rotation_from_quaternion(g._rotation.data[mask]).repeat(N, 1, 1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + g._xyz.data[mask].repeat(N, 1)
    new_scaling = torch.log(g.scaling[mask].repeat(N, 1) / (0.8 * N))
    new_rotation = g._rotation.data[mask].repeat(N, 1)
    new_sh_dc = g._sh_dc.data[mask].repeat(N, 1, 1)
    new_sh_rest = g._sh_rest.data[mask].repeat(N, 1, 1)
    new_opacity = g._opacity.data[mask].repeat(N, 1)

    ext = {
        "xyz": new_xyz,
        "sh_dc": new_sh_dc,
        "sh_rest": new_sh_rest,
        "opacity": new_opacity,
        "scaling": new_scaling,
        "rotation": new_rotation,
    }
    out = g._cat_tensors_to_optimizer(ext)
    g._xyz = out["xyz"]
    g._sh_dc = out["sh_dc"]
    g._sh_rest = out["sh_rest"]
    g._opacity = out["opacity"]
    g._scaling = out["scaling"]
    g._rotation = out["rotation"]

    # Remove original points that were split
    prune_filter = torch.cat([
        mask,
        torch.zeros(N * mask.sum(), device=mask.device, dtype=torch.bool),
    ])
    _prune(g, prune_filter)


def _prune(g: GaussianModel, mask: torch.Tensor):
    """Remove Gaussians where mask is True."""
    keep = ~mask
    out = g._prune_optimizer(keep)
    g._xyz = out["xyz"]
    g._sh_dc = out["sh_dc"]
    g._sh_rest = out["sh_rest"]
    g._opacity = out["opacity"]
    g._scaling = out["scaling"]
    g._rotation = out["rotation"]
    g.max_radii2D = g.max_radii2D[keep]
