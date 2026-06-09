# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT
"""Post-training Gaussian compaction: significance pruning + fine-tuning.

Enabled via --compact flag. Off by default for Phase 1 parity.
"""

from __future__ import annotations

import random
import torch
from tqdm import tqdm

from camera import Camera
from gaussians import GaussianModel
from renderer import render
from losses import l1_loss, ssim


@torch.no_grad()
def _compute_significance(
    gaussians: GaussianModel,
    cameras: list[Camera],
    bg_color: torch.Tensor,
    antialiased: bool = False,
) -> torch.Tensor:
    """Compute per-Gaussian significance = accumulated alpha contribution across views."""
    significance = torch.zeros(gaussians.num_gaussians, device="cuda")
    for cam in tqdm(cameras, desc="Computing significance"):
        out = render(cam, gaussians, bg_color, render_depth=False, antialiased=antialiased)
        vis = out["visibility_filter"]
        significance[vis] += 1.0
    return significance


def compact_and_finetune(
    gaussians: GaussianModel,
    train_cameras: list[Camera],
    bg_color: torch.Tensor,
    opt,
    compaction_ratio: float = 0.5,
    finetune_iters: int = 2000,
    antialiased: bool = False,
    test_cameras: list[Camera] | None = None,
):
    """Prune least-significant Gaussians and fine-tune to recover quality.

    Args:
        compaction_ratio: fraction of Gaussians to *keep* (0.5 = remove 50%).
        finetune_iters: number of optimisation steps after pruning.
    """
    before = gaussians.num_gaussians
    sig = _compute_significance(gaussians, train_cameras, bg_color, antialiased)

    # Also factor in opacity
    opacity = torch.sigmoid(gaussians._opacity).squeeze(-1)
    score = sig * opacity

    keep_count = max(int(before * compaction_ratio), 1)
    if keep_count < before:
        _, keep_idx = score.topk(keep_count)
        keep_mask = torch.zeros(before, dtype=torch.bool, device="cuda")
        keep_mask[keep_idx] = True
        out = gaussians._prune_optimizer(keep_mask)
        gaussians._xyz = out["xyz"]
        gaussians._sh_dc = out["sh_dc"]
        gaussians._sh_rest = out["sh_rest"]
        gaussians._opacity = out["opacity"]
        gaussians._scaling = out["scaling"]
        gaussians._rotation = out["rotation"]
        gaussians.max_radii2D = torch.zeros(gaussians.num_gaussians, device="cuda")

    after = gaussians.num_gaussians
    print(f"Compaction: {before} -> {after} Gaussians ({100 * after / before:.1f}%)")

    if finetune_iters <= 0:
        return

    # Fine-tune with photometric loss only, no densification
    print(f"Fine-tuning for {finetune_iters} iterations...")
    cam_list = list(train_cameras)
    for step in tqdm(range(finetune_iters), desc="Fine-tuning"):
        gaussians.update_learning_rate(step)
        cam = random.choice(cam_list)
        out = render(cam, gaussians, bg_color, render_depth=False, antialiased=antialiased)
        image = out["render"]
        gt = cam.original_image
        loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt) + opt.lambda_dssim * (1.0 - ssim(image, gt))
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

    # Evaluate after fine-tuning
    torch.cuda.empty_cache()
    with torch.no_grad():
        def _eval_set(cameras, label):
            total_l1, total_psnr, n = 0.0, 0.0, 0
            for cam in cameras:
                out = render(cam, gaussians, bg_color, render_depth=False, antialiased=antialiased)
                img = torch.clamp(out["render"], 0, 1)
                gt = torch.clamp(cam.original_image, 0, 1)
                mse = ((img - gt) ** 2).mean()
                total_l1 += l1_loss(img, gt).item()
                total_psnr += (-10.0 * torch.log10(mse)).item()
                n += 1
            avg_l1 = total_l1 / max(n, 1)
            avg_psnr = total_psnr / max(n, 1)
            print(f"[Compact] {label}: L1 {avg_l1:.6f}  PSNR {avg_psnr:.4f}")

        subset = [cam_list[i % len(cam_list)] for i in range(5, 30, 5)]
        _eval_set(subset, "Train")
        if test_cameras:
            _eval_set(test_cameras, "Test")
