# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""Evaluation metrics: PSNR, SSIM, LPIPS. Computes per-view and averages."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image


def _load_images(directory: str) -> list[torch.Tensor]:
    """Load all PNG images from a directory, sorted, as (3,H,W) float tensors."""
    paths = sorted(Path(directory).glob("*.png"))
    imgs = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(img).permute(2, 0, 1))
    return imgs


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = (pred - gt).pow(2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> float:
    """Compute SSIM between two (3,H,W) tensors."""
    from losses import ssim
    return ssim(pred.unsqueeze(0), gt.unsqueeze(0), window_size=window_size).item()


def compute_lpips(pred: torch.Tensor, gt: torch.Tensor, net: str = "vgg") -> float:
    """Compute LPIPS (lazy import to avoid hard dependency)."""
    try:
        import lpips
    except ImportError:
        return float("nan")
    if not hasattr(compute_lpips, "_fn") or compute_lpips._net != net:
        compute_lpips._fn = lpips.LPIPS(net=net).cuda()
        compute_lpips._net = net
    fn = compute_lpips._fn
    with torch.no_grad():
        val = fn(pred.unsqueeze(0).cuda(), gt.unsqueeze(0).cuda())
    return val.item()


def evaluate_directories(pred_dir: str, gt_dir: str, output_json: str | None = None) -> dict:
    """Compute metrics between two directories of images.

    Returns a dict with per-image and average PSNR / SSIM / LPIPS.
    """
    preds = _load_images(pred_dir)
    gts = _load_images(gt_dir)
    assert len(preds) == len(gts), f"Mismatch: {len(preds)} preds vs {len(gts)} gts"

    records = []
    for i, (p, g) in enumerate(zip(preds, gts)):
        rec = {
            "index": i,
            "psnr": compute_psnr(p, g),
            "ssim": compute_ssim(p, g),
            "lpips": compute_lpips(p, g),
        }
        records.append(rec)

    avg = {
        "psnr": np.mean([r["psnr"] for r in records]),
        "ssim": np.mean([r["ssim"] for r in records]),
        "lpips": np.nanmean([r["lpips"] for r in records]),
    }
    result = {"average": avg, "per_image": records}
    print(f"Avg PSNR: {avg['psnr']:.4f}  SSIM: {avg['ssim']:.4f}  LPIPS: {avg['lpips']:.4f}")

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved metrics to {output_json}")
    return result


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Compute image quality metrics")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    evaluate_directories(args.pred_dir, args.gt_dir, args.output)
