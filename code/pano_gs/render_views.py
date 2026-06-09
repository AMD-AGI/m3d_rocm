# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""Render all views from a trained model and save RGB / depth / alpha images."""

from __future__ import annotations

import os
from argparse import ArgumentParser

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from camera import Camera
from data_loader import load_blender_npz
from gaussians import GaussianModel
from renderer import render


def _find_latest_iteration(model_path: str) -> int:
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_dir):
        raise FileNotFoundError(f"No point_cloud directory in {model_path}")
    iters = []
    for name in os.listdir(pc_dir):
        if name.startswith("iteration_"):
            try:
                iters.append(int(name.split("_")[1]))
            except ValueError:
                pass
    if not iters:
        raise FileNotFoundError("No iteration_* folders found")
    return max(iters)


@torch.no_grad()
def render_set(
    model_path: str,
    split_name: str,
    iteration: int,
    views: list[Camera],
    gaussians: GaussianModel,
    bg: torch.Tensor,
    antialiased: bool = False,
):
    base = os.path.join(model_path, split_name, f"ours_{iteration}")
    pred_dir = os.path.join(base, "test_preds")
    gt_dir = os.path.join(base, "gt")
    depth_dir = os.path.join(base, "depth")
    alpha_dir = os.path.join(base, "alpha")

    for d in [pred_dir, gt_dir, depth_dir, alpha_dir]:
        os.makedirs(d, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {split_name}")):
        out = render(view, gaussians, bg, render_depth=True, antialiased=antialiased)
        image = torch.clamp(out["render"], 0, 1)
        gt = torch.clamp(view.original_image, 0, 1)

        torchvision.utils.save_image(image, os.path.join(pred_dir, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gt_dir, f"{idx:05d}.png"))

        # Depth (normalised to 0-255 grayscale)
        if out["depth"] is not None:
            depth = out["depth"].squeeze(0)  # (H, W)
            dmin, dmax = depth.min(), depth.max()
            if dmax - dmin > 1e-6:
                depth_norm = ((depth - dmin) / (dmax - dmin) * 255).byte()
            else:
                depth_norm = torch.zeros_like(depth, dtype=torch.uint8)
            Image.fromarray(depth_norm.cpu().numpy(), mode="L").save(
                os.path.join(depth_dir, f"{idx:05d}.png")
            )

        # Alpha
        if out["alpha"] is not None:
            alpha = torch.clamp(out["alpha"], 0, 1)
            torchvision.utils.save_image(alpha, os.path.join(alpha_dir, f"{idx:05d}.png"))


def main():
    parser = ArgumentParser(description="Render views from trained 3DGS model")
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--source_path", "-s", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", "-w", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--img_sample_interval", type=int, default=9)
    parser.add_argument("--num_views_per_view", type=int, default=20)
    parser.add_argument("--num_of_point_cloud", type=int, default=5_000_000)
    parser.add_argument("--antialiased", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.cuda.set_device(torch.device(args.device))
    args.source_path = os.path.abspath(args.source_path)

    iteration = args.iteration
    if iteration < 0:
        iteration = _find_latest_iteration(args.model_path)
    print(f"Rendering iteration {iteration} from {args.model_path}")

    # Load data (for cameras / GT images)
    train_cameras, test_cameras, _, _ = load_blender_npz(
        args.source_path,
        eval_mode=args.eval,
        interval=args.img_sample_interval,
        num_views_per_view=args.num_views_per_view,
        max_points=args.num_of_point_cloud,
    )

    # Load model
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    ply_path = os.path.join(
        args.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"
    )
    gaussians.load_ply(ply_path)
    print(f"Loaded {gaussians.num_gaussians} Gaussians")

    bg = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    if not args.skip_train:
        render_set(args.model_path, "train", iteration, train_cameras, gaussians, bg, args.antialiased)
    if not args.skip_test:
        render_set(args.model_path, "test", iteration, test_cameras, gaussians, bg, args.antialiased)


if __name__ == "__main__":
    main()
