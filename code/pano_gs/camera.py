# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT
"""Camera dataclass with projection math and gsplat-compatible properties."""

import math
import numpy as np
import torch


class Camera:
    """Holds per-view camera parameters and the ground-truth image."""

    def __init__(
        self,
        uid: int,
        R: np.ndarray,
        T: np.ndarray,
        FoVx: float,
        FoVy: float,
        image: torch.Tensor,
        image_name: str,
        width: int,
        height: int,
    ):
        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        self.original_image = image.clamp(0.0, 1.0).cuda()

        tan_fovx = math.tan(FoVx * 0.5)
        tan_fovy = math.tan(FoVy * 0.5)
        self.focal_x = width / (2.0 * tan_fovx)
        self.focal_y = height / (2.0 * tan_fovy)

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = T
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32).T.cuda()

        c2w = np.linalg.inv(w2c)
        self.camera_center = torch.tensor(c2w[:3, 3], dtype=torch.float32).cuda()

        self._viewmat = torch.tensor(w2c, dtype=torch.float32).unsqueeze(0).cuda()
        self._K = torch.tensor(
            [
                [self.focal_x, 0.0, width / 2.0],
                [0.0, self.focal_y, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0).cuda()

    @property
    def viewmat(self) -> torch.Tensor:
        """(1, 4, 4) world-to-camera for gsplat."""
        return self._viewmat

    @property
    def K(self) -> torch.Tensor:
        """(1, 3, 3) intrinsic matrix for gsplat."""
        return self._K
