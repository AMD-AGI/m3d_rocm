# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""3DGS training script using gsplat (Apache 2.0). Clean-room implementation."""

from __future__ import annotations

import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from camera import Camera
from data_loader import load_blender_npz
from gaussians import GaussianModel
from renderer import render
from losses import l1_loss, ssim, depth_normal_consistency_loss, depth_tv_loss
from densification import DensificationState, densify_and_prune, reset_opacity

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = (pred - gt).pow(2).mean()
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))


@torch.no_grad()
def evaluate(
    cameras: list[Camera],
    gaussians: GaussianModel,
    bg: torch.Tensor,
    antialiased: bool,
) -> tuple[float, float]:
    """Average L1 and PSNR over a set of cameras."""
    total_l1, total_psnr, n = 0.0, 0.0, 0
    for cam in cameras:
        out = render(cam, gaussians, bg, render_depth=False, antialiased=antialiased)
        img = torch.clamp(out["render"], 0, 1)
        gt = torch.clamp(cam.original_image, 0, 1)
        total_l1 += l1_loss(img, gt).item()
        total_psnr += _psnr(img, gt).item()
        n += 1
    return total_l1 / max(n, 1), total_psnr / max(n, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def training(args):
    # -- Setup --
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(vars(args)))

    tb_writer = None
    if TB_AVAILABLE:
        tb_writer = SummaryWriter(args.model_path)

    # -- Data --
    train_cameras, test_cameras, pcd, extent = load_blender_npz(
        source_path=args.source_path,
        eval_mode=args.eval,
        interval=args.img_sample_interval,
        num_views_per_view=args.num_views_per_view,
        max_points=args.num_of_point_cloud,
    )
    for idx, cam in enumerate(train_cameras + test_cameras):
        cam.idx = idx

    # -- Gaussian model --
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.create_from_point_cloud(pcd, spatial_lr_scale=extent)
    gaussians.setup_optimizer(args)

    if args.start_checkpoint:
        ckpt = torch.load(args.start_checkpoint)
        gaussians.restore(ckpt["model"], args)
        first_iter = ckpt["iteration"]
    else:
        first_iter = 0

    bg_color = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    # -- Densification state --
    den_state = DensificationState(gaussians.num_gaussians)

    # -- Training --
    viewpoint_stack = []
    ema_loss = 0.0
    progress = tqdm(range(first_iter, args.iterations), desc="Training")

    for iteration in range(first_iter + 1, args.iterations + 1):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.increase_sh_degree()

        # Random camera
        if not viewpoint_stack:
            viewpoint_stack = list(train_cameras)
            random.shuffle(viewpoint_stack)
        cam = viewpoint_stack.pop()

        # Render (always render depth so it's available for logging / Phase 2 losses)
        out = render(cam, gaussians, bg_color, render_depth=True, antialiased=args.antialiased)

        image = out["render"]
        gt_image = cam.original_image

        # Photometric loss
        Ll1 = l1_loss(image, gt_image)
        loss_ssim = 1.0 - ssim(image, gt_image)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * loss_ssim

        # Phase-2 geometric losses (off by default: lambdas are 0)
        if args.lambda_depth_normal > 0 and iteration > args.depth_normal_from_iter and out["depth"] is not None:
            loss = loss + args.lambda_depth_normal * depth_normal_consistency_loss(out["depth"])
        if args.lambda_depth_tv > 0 and iteration > args.depth_normal_from_iter and out["depth"] is not None:
            loss = loss + args.lambda_depth_tv * depth_tv_loss(out["depth"], image)

        loss.backward()

        with torch.no_grad():
            # EMA loss for progress bar
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress.set_postfix({"Loss": f"{ema_loss:.7f}", "N": gaussians.num_gaussians})
                progress.update(10)

            # TensorBoard
            if tb_writer:
                tb_writer.add_scalar("train/l1", Ll1.item(), iteration)
                tb_writer.add_scalar("train/total_loss", loss.item(), iteration)
                tb_writer.add_scalar("train/num_gaussians", gaussians.num_gaussians, iteration)

            # Periodic evaluation
            if iteration in args.test_iterations:
                torch.cuda.empty_cache()
                if test_cameras:
                    test_l1, test_psnr = evaluate(test_cameras, gaussians, bg_color, args.antialiased)
                    print(f"\n[ITER {iteration}] Test: L1 {test_l1:.6f}  PSNR {test_psnr:.4f}")
                    if tb_writer:
                        tb_writer.add_scalar("test/l1", test_l1, iteration)
                        tb_writer.add_scalar("test/psnr", test_psnr, iteration)

                subset = [train_cameras[i % len(train_cameras)] for i in range(5, 30, 5)]
                train_l1, train_psnr = evaluate(subset, gaussians, bg_color, args.antialiased)
                print(f"[ITER {iteration}] Train: L1 {train_l1:.6f}  PSNR {train_psnr:.4f}")
                if tb_writer:
                    tb_writer.add_scalar("train_eval/l1", train_l1, iteration)
                    tb_writer.add_scalar("train_eval/psnr", train_psnr, iteration)
                torch.cuda.empty_cache()

            # Save PLY
            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians ({gaussians.num_gaussians} points)")
                ply_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{iteration}")
                os.makedirs(ply_dir, exist_ok=True)
                gaussians.save_ply(os.path.join(ply_dir, "point_cloud.ply"))

            # Densification
            if iteration < args.densify_until_iter:
                gaussians.max_radii2D[out["visibility_filter"]] = torch.max(
                    gaussians.max_radii2D[out["visibility_filter"]],
                    out["radii"][out["visibility_filter"]].float(),
                )
                den_state.accumulate(out["means2d"], out["visibility_filter"])

                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_thresh = 20 if iteration > args.opacity_reset_interval else None
                    densify_and_prune(
                        gaussians, den_state,
                        grad_threshold=args.densify_grad_threshold,
                        min_opacity=0.05,
                        extent=extent,
                        max_screen_size=size_thresh,
                        percent_dense=args.percent_dense,
                    )

                if iteration % args.opacity_reset_interval == 0 or \
                   (args.white_background and iteration == args.densify_from_iter):
                    reset_opacity(gaussians)

            # Optimizer step
            if iteration < args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Checkpoint
            if iteration in args.checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    {"model": gaussians.capture(), "iteration": iteration},
                    os.path.join(args.model_path, f"chkpnt{iteration}.pth"),
                )

            # Periodic log image
            if iteration % args.densification_interval == 1:
                os.makedirs(os.path.join(args.model_path, "log_images"), exist_ok=True)
                eval_cam = random.choice(train_cameras + test_cameras)
                with torch.no_grad():
                    log_img = render(eval_cam, gaussians, bg_color, render_depth=False, antialiased=args.antialiased)["render"]
                    log_img = torch.clamp(log_img, 0, 1)
                    gt = torch.clamp(eval_cam.original_image, 0, 1)
                    row = torch.cat([gt, log_img], dim=2)
                    torchvision.utils.save_image(row, os.path.join(args.model_path, "log_images", f"{iteration}.jpg"))

    progress.close()

    # Final save
    ply_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iterations}")
    os.makedirs(ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(ply_dir, "point_cloud.ply"))

    # Post-training compaction (Phase 2, off by default)
    if args.compact:
        from compaction import compact_and_finetune
        compact_and_finetune(
            gaussians, train_cameras, bg_color, args,
            compaction_ratio=args.compaction_ratio,
            finetune_iters=args.compaction_finetune_iters,
            antialiased=args.antialiased,
            test_cameras=test_cameras,
        )
        compact_dir = os.path.join(args.model_path, "point_cloud", "compact")
        os.makedirs(compact_dir, exist_ok=True)
        gaussians.save_ply(os.path.join(compact_dir, "point_cloud.ply"))
        print(f"Compact model saved: {gaussians.num_gaussians} Gaussians")

    if tb_writer:
        tb_writer.close()
    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="3DGS training (gsplat)")

    # Data
    parser.add_argument("--source_path", "-s", type=str, required=True)
    parser.add_argument("--model_path", "-m", type=str, default="./output")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", "-w", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--img_sample_interval", type=int, default=9)
    parser.add_argument("--num_views_per_view", type=int, default=20)
    parser.add_argument("--num_of_point_cloud", type=int, default=5_000_000)

    # Optimisation
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--position_lr_max_steps", type=int, default=30_000)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.01)
    parser.add_argument("--scaling_lr", type=float, default=0.0005)
    parser.add_argument("--rotation_lr", type=float, default=0.0002)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--percent_dense", type=float, default=0.01)

    # Densification
    parser.add_argument("--densification_interval", type=int, default=500)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=15_000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)

    # Phase 2 losses (off by default)
    parser.add_argument("--lambda_depth_normal", type=float, default=0.0)
    parser.add_argument("--lambda_depth_tv", type=float, default=0.0)
    parser.add_argument("--depth_normal_from_iter", type=int, default=1000)

    # Rendering
    parser.add_argument("--antialiased", action="store_true")

    # Compaction (Phase 2, off by default)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--compaction_ratio", type=float, default=0.5)
    parser.add_argument("--compaction_finetune_iters", type=int, default=2000)

    # Checkpoints / logging
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.source_path = os.path.abspath(args.source_path)
    args.save_iterations = list(set(args.save_iterations + [args.iterations]))

    print(f"Optimizing {args.model_path}")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device(args.device))

    training(args)
