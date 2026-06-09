# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""gsplat-based Gaussian rasterization wrapper."""

import torch
import gsplat

from camera import Camera
from gaussians import GaussianModel


def _extract_radii(meta: dict, N: int, device: torch.device) -> torch.Tensor:
    """Extract per-Gaussian screen-space radii from gsplat meta dict."""
    radii = torch.zeros(N, device=device, dtype=torch.int32)
    if "radii" not in meta:
        return radii
    gs_radii = meta["radii"]
    if gs_radii.dim() == 3:
        radii = gs_radii[0].max(dim=-1).values.int()
    elif gs_radii.dim() == 2:
        if gs_radii.shape[-1] == 2:
            gids = meta.get("gaussian_ids")
            if gids is not None:
                radii[gids.long()] = gs_radii.max(dim=-1).values.int()
            else:
                radii = gs_radii.max(dim=-1).values.int()
        else:
            radii = gs_radii[0].int()
    elif gs_radii.dim() == 1:
        gids = meta.get("gaussian_ids")
        if gids is not None:
            radii[gids.long()] = gs_radii.int()
        else:
            radii = gs_radii.int()
    return radii


def render(
    camera: Camera,
    gaussians: GaussianModel,
    bg_color: torch.Tensor,
    render_depth: bool = True,
    antialiased: bool = False,
) -> dict:
    """Render the scene for a single camera.

    Returns a dict with keys:
        render  – (3, H, W) RGB image
        depth   – (1, H, W) expected depth (only if render_depth=True)
        alpha   – (1, H, W) accumulated opacity
        means2d – (1, N, 2) 2D projection (gradient carrier)
        radii   – (N,) int32 screen-space radii
        visibility_filter – (N,) bool mask of visible Gaussians
    """
    render_mode = "RGB+ED" if render_depth else "RGB"
    rasterize_mode = "antialiased" if antialiased else "classic"

    render_colors, render_alphas, meta = gsplat.rasterization(
        means=gaussians.xyz,
        quats=gaussians.rotation,
        scales=gaussians.scaling,
        opacities=gaussians.opacity,
        colors=gaussians.sh_coeffs,
        viewmats=camera.viewmat,
        Ks=camera.K,
        width=camera.image_width,
        height=camera.image_height,
        near_plane=0.01,
        far_plane=1e10,
        eps2d=0.3,
        sh_degree=gaussians.active_sh_degree,
        backgrounds=bg_color.unsqueeze(0),
        packed=False,
        render_mode=render_mode,
        rasterize_mode=rasterize_mode,
    )

    out = {}
    if render_depth:
        out["render"] = render_colors[0, ..., :3].permute(2, 0, 1)   # (3,H,W)
        out["depth"] = render_colors[0, ..., 3:4].permute(2, 0, 1)   # (1,H,W)
    else:
        out["render"] = render_colors[0].permute(2, 0, 1)            # (3,H,W)
        out["depth"] = None

    out["alpha"] = render_alphas[0].permute(2, 0, 1)                 # (1,H,W)

    N = gaussians.num_gaussians
    radii = _extract_radii(meta, N, gaussians.xyz.device)
    out["radii"] = radii
    out["visibility_filter"] = radii > 0

    means2d = meta["means2d"]  # (1, N, 2)
    if means2d.requires_grad:
        means2d.retain_grad()
    out["means2d"] = means2d

    return out
