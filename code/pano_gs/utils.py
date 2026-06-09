# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""Shared helpers: PointCloud, inverse_sigmoid, LR schedule, PLY I/O."""

from __future__ import annotations

from typing import NamedTuple
import numpy as np
import torch
from plyfile import PlyData, PlyElement


class PointCloud(NamedTuple):
    xyz: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB to 0-th order SH coefficient."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def sh0_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def build_rotation_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (N,4) -> rotation matrix (N,3,3). Convention: [w,x,y,z]."""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def make_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1_000_000,
):
    """Return a callable step -> lr implementing log-linear (exponential) decay."""
    def _lr(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        t = min(step / max_steps, 1.0)
        log_lerp = np.exp(np.log(lr_init) * (1.0 - t) + np.log(lr_final) * t)
        delay = lr_delay_mult + (1.0 - lr_delay_mult) * np.sin(
            0.5 * np.pi * min(step / max_steps, 1.0)
        )
        return delay * log_lerp

    return _lr


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------


def save_ply(path: str, xyz: np.ndarray, attrs: dict[str, np.ndarray], attr_names: list[str]):
    """Write a PLY with named float32 attributes.

    *attrs* maps short group name -> (N, K) array.
    *attr_names* is the flat ordered list of per-scalar attribute names
    (must match total column count across all attrs arrays).
    """
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    normals = np.zeros_like(xyz)
    all_data = [xyz, normals] + [v for v in attrs.values()]
    full_names = ["x", "y", "z", "nx", "ny", "nz"] + attr_names
    all_data = np.concatenate(all_data, axis=1)
    dtype = [(n, "f4") for n in full_names]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, all_data))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def load_ply(path: str):
    """Load a PLY file and return the PlyData element for flexible reading."""
    return PlyData.read(path)


def fetch_ply_points(path: str) -> PointCloud:
    """Load positions + colors from a simple colored PLY."""
    plydata = PlyData.read(path)
    v = plydata["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    normals = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)
    return PointCloud(xyz, colors, normals)


def store_ply_points(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """Save a simple colored PLY (positions + colors, rgb in 0-255 uint8)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    N = xyz.shape[0]
    normals = np.zeros_like(xyz)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255 if rgb.max() <= 1.1 else rgb, 0, 255).astype(np.uint8)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(N, dtype=dtype)
    for i, name in enumerate(["x", "y", "z"]):
        elements[name] = xyz[:, i]
    for i, name in enumerate(["nx", "ny", "nz"]):
        elements[name] = normals[:, i]
    for i, name in enumerate(["red", "green", "blue"]):
        elements[name] = rgb[:, i]
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
