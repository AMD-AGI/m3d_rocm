# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""BlenderNPZ dataset loader: cameras, train/test split, point cloud from depth."""

from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot, Slerp

from camera import Camera
from utils import PointCloud, fetch_ply_points, store_ply_points


# ---------------------------------------------------------------------------
# Depth edge detection helpers (for point-cloud init from depth maps)
# ---------------------------------------------------------------------------

def _max_pool_2d(x: np.ndarray, k: int) -> np.ndarray:
    """Simple max-pool with same-size padding."""
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    out = x.copy()
    for di in range(k):
        for dj in range(k):
            out = np.maximum(out, xp[di : di + x.shape[0], dj : dj + x.shape[1]])
    return out


def _depth_edge_mask(depth: np.ndarray, rtol: float = 0.03, k: int = 3) -> np.ndarray:
    diff = _max_pool_2d(depth, k) + _max_pool_2d(-depth, k)
    return ~(diff / np.clip(np.abs(depth), 1e-8, None) > rtol)


def _unproject_depth(depth: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Unproject a depth map to world-space points. Returns (H*W, 3)."""
    H, W = depth.shape[:2]
    c2w = np.linalg.inv(w2c).astype(np.float32)
    u, v = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5,
                       np.arange(H, dtype=np.float32) + 0.5)
    ones = np.ones_like(u)
    pixels = np.stack([u, v, ones], axis=-1)  # (H, W, 3)
    K_inv = np.linalg.inv(K).astype(np.float32)
    rays_cam = pixels @ K_inv.T  # (H, W, 3)
    rays_world = rays_cam @ c2w[:3, :3].T
    origin = c2w[:3, 3]
    points = depth[..., None] * rays_world + origin[None, None, :]
    return points.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Camera reading
# ---------------------------------------------------------------------------

def _focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def _read_cameras(path: str, extension: str = ".png", prefix: str = "mv_rgb"):
    """Read cameras from world_matrix.npz + para.json."""
    cam_dict = np.load(os.path.join(path, "world_matrix.npz"))
    c2w_all = cam_dict["arr_0"].reshape(-1, 4, 4).copy()

    with open(os.path.join(path, "para.json")) as f:
        focal = json.load(f)["focal_length_in_pixel"]

    cameras = []
    for idx in range(c2w_all.shape[0]):
        c2w = c2w_all[idx].copy()
        c2w[:3, 1:3] *= -1  # OpenGL -> COLMAP convention

        w2c = np.linalg.inv(c2w).astype(np.float32)
        R = w2c[:3, :3].T  # stored transposed for glm convention
        T = w2c[:3, 3]

        img_path = os.path.join(path, prefix, f"{idx:04d}{extension}")
        if not os.path.exists(img_path):
            continue
        pil_img = Image.open(img_path).convert("RGB")
        W, H = pil_img.size
        img_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).permute(2, 0, 1)

        fovx = _focal_to_fov(focal, W)
        fovy = _focal_to_fov(focal, H)

        cameras.append(Camera(
            uid=idx, R=R, T=T,
            FoVx=fovx, FoVy=fovy,
            image=img_tensor,
            image_name=f"{idx:04d}",
            width=W, height=H,
        ))
    return cameras, focal


# ---------------------------------------------------------------------------
# Interpolated test cameras (360-degree path)
# ---------------------------------------------------------------------------

def _interpolate_cameras(cameras: list[Camera], num_per_pair: int = 3) -> list[Camera]:
    """Generate interpolated views between consecutive train cameras."""
    interp = []
    for i in range(len(cameras) - 1):
        c0, c1 = cameras[i], cameras[i + 1]
        rot0 = c0.R.T  # back to w2c rotation
        rot1 = c1.R.T
        q0 = Rot.from_matrix(rot0)
        q1 = Rot.from_matrix(rot1)
        slerp = Slerp([0, 1], Rot.from_quat(np.stack([q0.as_quat(), q1.as_quat()])))
        for j, t in enumerate(np.linspace(0, 1, num_per_pair)):
            rot_interp = slerp(t).as_matrix()
            T_interp = (1 - t) * c0.T + t * c1.T
            R_interp = rot_interp.T  # transpose for glm storage
            uid = i * num_per_pair + j
            interp.append(Camera(
                uid=uid, R=R_interp.astype(np.float32), T=T_interp.astype(np.float32),
                FoVx=c0.FoVx, FoVy=c0.FoVy,
                image=c0.original_image.cpu(),
                image_name=f"{c0.image_name}_interp{t:.2f}",
                width=c0.image_width, height=c0.image_height,
            ))
    return interp


# ---------------------------------------------------------------------------
# Point cloud from depth maps
# ---------------------------------------------------------------------------

def _build_point_cloud_from_depth(
    path: str,
    all_cameras: list[Camera],
    focal: float,
    max_points: int = 5_000_000,
) -> PointCloud:
    depth_dir = os.path.join(path, "mv_depth")
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    all_xyz, all_rgb = [], []
    for cam in all_cameras:
        depth_path = os.path.join(depth_dir, f"{cam.image_name}.exr")
        if not os.path.exists(depth_path):
            continue
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        depth = depth.astype(np.float32)
        depth[depth >= 500.0] = 500.0

        edge_mask = _depth_edge_mask(depth, rtol=0.03, k=3)
        valid = edge_mask.ravel()

        H, W = depth.shape
        K = np.array(
            [[focal, 0.0, W / 2.0],
             [0.0, focal, H / 2.0],
             [0.0, 0.0, 1.0]], dtype=np.float32,
        )
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = cam.R.T
        w2c[:3, 3] = cam.T

        pts = _unproject_depth(depth, K, w2c)
        rgb = np.array(Image.open(os.path.join(path, "mv_rgb", f"{cam.image_name}.png")).convert("RGB"), dtype=np.float32)
        rgb = rgb.reshape(-1, 3)

        all_xyz.append(pts[valid])
        all_rgb.append(rgb[valid])

    if not all_xyz:
        return None
    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    if xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    return PointCloud(xyz=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))


# ---------------------------------------------------------------------------
# Camera extent (normalization radius)
# ---------------------------------------------------------------------------

def _compute_camera_extent(cameras: list[Camera]) -> float:
    centres = []
    for cam in cameras:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = cam.R.T
        w2c[:3, 3] = cam.T
        c2w = np.linalg.inv(w2c)
        centres.append(c2w[:3, 3])
    centres = np.stack(centres, axis=0)
    avg = centres.mean(axis=0)
    radius = np.linalg.norm(centres - avg, axis=1).max() * 1.1
    return float(radius)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_blender_npz(
    source_path: str,
    eval_mode: bool = False,
    interval: int = 9,
    num_views_per_view: int = 20,
    max_points: int = 5_000_000,
):
    """Load a BlenderNPZ dataset.

    Returns
    -------
    train_cameras, test_cameras, point_cloud, camera_extent
    """
    all_cameras, focal = _read_cameras(source_path)
    print(f"Loaded {len(all_cameras)} cameras from {source_path}")

    if eval_mode:
        train_cameras = [
            c for c in all_cameras
            if int(c.image_name) % interval == 0
        ]
        test_cameras = _interpolate_cameras(all_cameras, num_per_pair=num_views_per_view)
    else:
        train_cameras = all_cameras
        test_cameras = []

    print(f"Train cameras: {len(train_cameras)}, Test cameras: {len(test_cameras)}")

    ply_path = os.path.join(source_path, "points3d.ply")
    depth_dir = os.path.join(source_path, "mv_depth")

    if os.path.exists(ply_path):
        pcd = fetch_ply_points(ply_path)
    elif os.path.exists(depth_dir):
        pcd = _build_point_cloud_from_depth(source_path, all_cameras, focal, max_points)
        if pcd is not None:
            store_ply_points(ply_path, pcd.xyz, (pcd.colors * 255).astype(np.uint8))
    else:
        num_pts = 100_000
        print(f"No depth or PLY found, generating {num_pts} random points")
        xyz = np.random.uniform(-1.3, 1.3, (num_pts, 3)).astype(np.float32)
        colors = np.random.uniform(0, 1, (num_pts, 3)).astype(np.float32)
        pcd = PointCloud(xyz=xyz, colors=colors, normals=np.zeros_like(xyz))
        store_ply_points(ply_path, xyz, (colors * 255).astype(np.uint8))

    extent = _compute_camera_extent(train_cameras)
    return train_cameras, test_cameras, pcd, extent
