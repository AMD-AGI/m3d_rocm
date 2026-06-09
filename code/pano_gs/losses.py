# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""Loss functions: L1, SSIM (Phase 1), depth-normal + depth TV (Phase 2, off by default)."""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Photometric losses (active in Phase 1)
# ---------------------------------------------------------------------------

def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - gt).mean()


def _gaussian_window(size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g /= g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """Structural Similarity Index (returns the SSIM *value*, not the loss)."""
    C = img1.shape[-3]
    window = _gaussian_window(window_size, 1.5, C).to(img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean() if size_average else ssim_map.mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Geometric losses (Phase 2 — off by default, enabled via lambda > 0)
# ---------------------------------------------------------------------------

def depth_normal_consistency_loss(depth: torch.Tensor) -> torch.Tensor:
    """Penalise inconsistency between normals derived from the rendered depth.

    Args:
        depth: (1, H, W) rendered expected depth.

    Returns:
        Scalar loss.
    """
    d = depth.squeeze(0)  # (H, W)
    dz_dx = d[:, 1:] - d[:, :-1]  # horizontal gradient
    dz_dy = d[1:, :] - d[:-1, :]  # vertical gradient

    # Construct normals from finite differences (unnormalised)
    # n = (-dz/dx, -dz/dy, 1), then normalise
    H, W = d.shape
    dz_dx_c = dz_dx[:H - 1, :W - 1]
    dz_dy_c = dz_dy[:H - 1, :W - 1]
    normal = torch.stack([-dz_dx_c, -dz_dy_c, torch.ones_like(dz_dx_c)], dim=-1)
    normal = F.normalize(normal, dim=-1)

    # Smoothness: dot product between neighbours should be close to 1
    dot_h = (normal[:, :-1] * normal[:, 1:]).sum(dim=-1)
    dot_v = (normal[:-1, :] * normal[1:, :]).sum(dim=-1)
    loss = (1.0 - dot_h).mean() + (1.0 - dot_v).mean()
    return loss


def depth_tv_loss(
    depth: torch.Tensor,
    rgb: torch.Tensor,
) -> torch.Tensor:
    """Edge-aware total variation on depth, weighted by RGB edge magnitude.

    Args:
        depth: (1, H, W) rendered depth.
        rgb:   (3, H, W) rendered or GT RGB image.

    Returns:
        Scalar loss.
    """
    d = depth.squeeze(0)
    grad_d_x = torch.abs(d[:, 1:] - d[:, :-1])
    grad_d_y = torch.abs(d[1:, :] - d[:-1, :])

    grad_rgb_x = torch.abs(rgb[:, :, 1:] - rgb[:, :, :-1]).mean(dim=0)
    grad_rgb_y = torch.abs(rgb[:, 1:, :] - rgb[:, :-1, :]).mean(dim=0)

    weight_x = torch.exp(-grad_rgb_x)
    weight_y = torch.exp(-grad_rgb_y)

    H, W = d.shape
    loss = (weight_x[:H, :W - 1] * grad_d_x).mean() + (weight_y[:H - 1, :W] * grad_d_y).mean()
    return loss
