# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""GaussianModel: learnable 3D Gaussians with optimizer, PLY I/O, checkpoint support."""

from __future__ import annotations

import os
import numpy as np
import torch
from torch import nn
from plyfile import PlyData, PlyElement

from utils import (
    PointCloud,
    inverse_sigmoid,
    rgb_to_sh0,
    make_expon_lr_func,
)


class GaussianModel:
    """Manages 3D Gaussian parameters and their optimizer."""

    def __init__(self, sh_degree: int = 3):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0

        self._xyz = torch.empty(0)
        self._sh_dc = torch.empty(0)
        self._sh_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0.0

    # ------------------------------------------------------------------
    # Properties (activated parameters)
    # ------------------------------------------------------------------

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity).squeeze(-1)

    @property
    def sh_coeffs(self) -> torch.Tensor:
        """(N, K, 3) SH coefficients where K = (max_sh_degree+1)^2."""
        return torch.cat([self._sh_dc, self._sh_rest], dim=1)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def create_from_point_cloud(self, pcd: PointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        xyz = torch.tensor(pcd.xyz, dtype=torch.float32, device="cuda")
        colors = torch.tensor(pcd.colors, dtype=torch.float32, device="cuda")
        sh0 = rgb_to_sh0(colors)  # (N, 3)

        n_sh = (self.max_sh_degree + 1) ** 2
        sh_dc = sh0.unsqueeze(1)  # (N, 1, 3)
        sh_rest = torch.zeros(xyz.shape[0], n_sh - 1, 3, device="cuda")

        print(f"Initialising {xyz.shape[0]} Gaussians")

        dists = torch.clamp_min(
            self._knn_dist_sq(xyz), 1e-7
        )
        scales = torch.log(torch.sqrt(dists)).unsqueeze(-1).repeat(1, 3)
        rots = torch.zeros(xyz.shape[0], 4, device="cuda")
        rots[:, 0] = 1.0
        opacities = inverse_sigmoid(0.1 * torch.ones(xyz.shape[0], 1, device="cuda"))

        self._xyz = nn.Parameter(xyz)
        self._sh_dc = nn.Parameter(sh_dc)
        self._sh_rest = nn.Parameter(sh_rest)
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities)
        self.max_radii2D = torch.zeros(xyz.shape[0], device="cuda")

    @staticmethod
    def _knn_dist_sq(pts: torch.Tensor) -> torch.Tensor:
        """Average squared distance to k nearest neighbours via simple_knn."""
        from simple_knn._C import distCUDA2
        return torch.clamp_min(distCUDA2(pts.float().contiguous()), 1e-7)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def setup_optimizer(self, opt):
        self.percent_dense = opt.percent_dense
        lr_xyz = opt.position_lr_init * self.spatial_lr_scale
        param_groups = [
            {"params": [self._xyz], "lr": lr_xyz, "name": "xyz"},
            {"params": [self._sh_dc], "lr": opt.feature_lr, "name": "sh_dc"},
            {"params": [self._sh_rest], "lr": opt.feature_lr / 20.0, "name": "sh_rest"},
            {"params": [self._opacity], "lr": opt.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": opt.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": opt.rotation_lr, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self._xyz_lr_func = make_expon_lr_func(
            lr_init=lr_xyz,
            lr_final=opt.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=opt.position_lr_delay_mult,
            max_steps=opt.position_lr_max_steps,
        )

    def update_learning_rate(self, step: int):
        for pg in self.optimizer.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = self._xyz_lr_func(step)

    def increase_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ------------------------------------------------------------------
    # Optimizer state manipulation (for densification)
    # ------------------------------------------------------------------

    def _replace_param_in_optimizer(self, new_tensor: torch.Tensor, name: str):
        for pg in self.optimizer.param_groups:
            if pg["name"] == name:
                old = pg["params"][0]
                state = self.optimizer.state.get(old, None)
                if state is not None:
                    state["exp_avg"] = torch.zeros_like(new_tensor)
                    state["exp_avg_sq"] = torch.zeros_like(new_tensor)
                    del self.optimizer.state[old]
                pg["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                if state is not None:
                    self.optimizer.state[pg["params"][0]] = state
                return pg["params"][0]

    def _prune_optimizer(self, mask: torch.Tensor) -> dict:
        """Keep only entries where *mask* is True."""
        out = {}
        for pg in self.optimizer.param_groups:
            old = pg["params"][0]
            state = self.optimizer.state.get(old, None)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][mask]
                state["exp_avg_sq"] = state["exp_avg_sq"][mask]
                del self.optimizer.state[old]
            pg["params"][0] = nn.Parameter(old.data[mask].requires_grad_(True))
            if state is not None:
                self.optimizer.state[pg["params"][0]] = state
            out[pg["name"]] = pg["params"][0]
        return out

    def _cat_tensors_to_optimizer(self, ext: dict[str, torch.Tensor]) -> dict:
        out = {}
        for pg in self.optimizer.param_groups:
            old = pg["params"][0]
            extension = ext[pg["name"]]
            state = self.optimizer.state.get(old, None)
            if state is not None:
                state["exp_avg"] = torch.cat([state["exp_avg"], torch.zeros_like(extension)], dim=0)
                state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], torch.zeros_like(extension)], dim=0)
                del self.optimizer.state[old]
            pg["params"][0] = nn.Parameter(torch.cat([old.data, extension], dim=0).requires_grad_(True))
            if state is not None:
                self.optimizer.state[pg["params"][0]] = state
            out[pg["name"]] = pg["params"][0]
        return out

    # ------------------------------------------------------------------
    # PLY I/O
    # ------------------------------------------------------------------

    def _attr_names(self) -> list[str]:
        names = []
        for i in range(self._sh_dc.shape[1] * self._sh_dc.shape[2]):
            names.append(f"f_dc_{i}")
        for i in range(self._sh_rest.shape[1] * self._sh_rest.shape[2]):
            names.append(f"f_rest_{i}")
        names.append("opacity")
        for i in range(self._scaling.shape[1]):
            names.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            names.append(f"rot_{i}")
        return names

    def save_ply(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._sh_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._sh_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()

        attr_names = ["x", "y", "z", "nx", "ny", "nz"] + self._attr_names()
        dtype = [(n, "f4") for n in attr_names]
        data = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scales, rots], axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, data))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path: str):
        ply = PlyData.read(path)
        v = ply.elements[0]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
        opacities = np.asarray(v["opacity"])[..., np.newaxis]

        dc = np.zeros((xyz.shape[0], 3, 1))
        dc[:, 0, 0] = v["f_dc_0"]
        dc[:, 1, 0] = v["f_dc_1"]
        dc[:, 2, 0] = v["f_dc_2"]

        rest_names = sorted(
            [p.name for p in v.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        n_rest = len(rest_names)
        rest = np.zeros((xyz.shape[0], n_rest))
        for i, name in enumerate(rest_names):
            rest[:, i] = np.asarray(v[name])
        rest = rest.reshape(xyz.shape[0], 3, n_rest // 3)

        scale_names = sorted(
            [p.name for p in v.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.stack([v[n] for n in scale_names], axis=1)

        rot_names = sorted(
            [p.name for p in v.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.stack([v[n] for n in rot_names], axis=1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda"))
        self._sh_dc = nn.Parameter(torch.tensor(dc, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous())
        self._sh_rest = nn.Parameter(torch.tensor(rest, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous())
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device="cuda"))
        self.max_radii2D = torch.zeros(xyz.shape[0], device="cuda")
        self.active_sh_degree = self.max_sh_degree

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def capture(self):
        return {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            "sh_dc": self._sh_dc,
            "sh_rest": self._sh_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "max_radii2D": self.max_radii2D,
            "spatial_lr_scale": self.spatial_lr_scale,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
        }

    def restore(self, ckpt: dict, opt):
        self.active_sh_degree = ckpt["active_sh_degree"]
        self._xyz = ckpt["xyz"]
        self._sh_dc = ckpt["sh_dc"]
        self._sh_rest = ckpt["sh_rest"]
        self._scaling = ckpt["scaling"]
        self._rotation = ckpt["rotation"]
        self._opacity = ckpt["opacity"]
        self.max_radii2D = ckpt["max_radii2D"]
        self.spatial_lr_scale = ckpt["spatial_lr_scale"]
        self.setup_optimizer(opt)
        if ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
