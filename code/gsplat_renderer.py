# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, repeat

try:
    from gsplat import rasterization
except ImportError:
    raise ImportError("Please install gsplat: pip install gsplat")



@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    depth: Float[Tensor, "batch view height width"] | None


class GsplatRenderer(nn.Module):
    """
    A simple gsplat-based renderer compatible with the DecoderSplattingCUDA interface.
    """
    
    def __init__(
        self,
        make_scale_invariant: bool = True,
        background_color: list[float] = [1., 1., 1.]
    ):
        super().__init__()
        self.make_scale_invariant = make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
    
    def _covariance_to_scale_rotation(self, covariance: Float[Tensor, "N 3 3"]):
        """
        Convert covariance matrices to scale and quaternion rotation.
        Covariance = R @ diag(scale^2) @ R^T
        
        Args:
            covariance: [N, 3, 3] covariance matrices
            
        Returns:
            scales: [N, 3] scale values
            quats: [N, 4] quaternions (w, x, y, z)
        """
        # Eigendecomposition: covariance = V @ diag(eigenvalues) @ V^T
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        
        # Scales are sqrt of eigenvalues (since cov = R @ diag(s^2) @ R^T)
        scales = torch.sqrt(torch.clamp(eigenvalues, min=1e-10))  # [N, 3]
        
        # Convert rotation matrix to quaternion
        # eigenvectors is [N, 3, 3] rotation matrix
        quats = self._rotation_matrix_to_quaternion(eigenvectors)  # [N, 4]
        
        return scales, quats
    
    def _rotation_matrix_to_quaternion(self, R: Float[Tensor, "N 3 3"]):
        """
        Convert rotation matrices to quaternions (w, x, y, z).
        
        Args:
            R: [N, 3, 3] rotation matrices
            
        Returns:
            quats: [N, 4] quaternions (w, x, y, z)
        """
        batch_size = R.shape[0]
        
        # Allocate quaternion tensor
        q = torch.zeros(batch_size, 4, dtype=R.dtype, device=R.device)
        
        # Compute trace
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        
        # Case 1: trace > 0
        mask1 = trace > 0
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s
        
        # Case 2: R[0,0] is the largest diagonal element
        mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
        
        # Case 3: R[1,1] is the largest diagonal element
        mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
        
        # Case 4: R[2,2] is the largest diagonal element
        mask4 = (~mask1) & (~mask2) & (~mask3)
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s
        
        # Normalize quaternions
        q = q / torch.norm(q, dim=1, keepdim=True)
        
        return q
    
    def _convert_extrinsics_to_viewmat(self, extrinsics: Float[Tensor, "N 4 4"]):
        """
        Convert extrinsics (c2w) to view matrix (w2c).
        
        Args:
            extrinsics: [N, 4, 4] camera-to-world matrices
            
        Returns:
            viewmat: [N, 4, 4] world-to-camera matrices
        """
        # gsplat expects world-to-camera (w2c), which is the inverse of c2w
        return torch.inverse(extrinsics)
    
    def _sh_to_rgb(self, harmonics: Float[Tensor, "N 3 d_sh"]):
        """
        Convert spherical harmonics to RGB.
        For degree 0 (DC component only), the conversion is simple.
        
        Args:
            harmonics: [N, 3, d_sh] spherical harmonics coefficients
            
        Returns:
            colors: [N, 3] RGB colors
        """
        # For SH degree 0, we just need the DC component
        # The DC component has coefficient C_0 = 0.28209479177387814
        C0 = 0.28209479177387814
        colors = harmonics[..., 0] * C0 + 0.5
        return colors
    
    def forward_direct(
        self,
        means: Float[Tensor, "batch gaussian 3"],
        scales: Float[Tensor, "batch gaussian 3"],
        quats: Float[Tensor, "batch gaussian 4"],
        colors: Float[Tensor, "batch gaussian 3"],
        opacities: Float[Tensor, "batch gaussian"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: str | None = None,
    ) -> DecoderOutput:
        """
        Render Gaussians using gsplat with direct scale/quat parameters (faster).
        
        Args:
            means: [batch, N, 3] Gaussian centers
            scales: [batch, N, 3] Gaussian scales
            quats: [batch, N, 4] Gaussian quaternions (w, x, y, z)
            colors: [batch, N, 3] RGB colors
            opacities: [batch, N] Gaussian opacities
            extrinsics: [batch, view, 4, 4] camera extrinsics (c2w)
            intrinsics: [batch, view, 3, 3] camera intrinsics
            near: [batch, view] near plane
            far: [batch, view] far plane
            image_shape: (height, width) output image size
            depth_mode: optional depth rendering mode (not used in this simple version)
            
        Returns:
            DecoderOutput with color and depth
        """
        b, v, _, _ = extrinsics.shape
        height, width = image_shape
        
        # Flatten batch and view dimensions
        extrinsics_flat = rearrange(extrinsics, "b v i j -> (b v) i j")
        intrinsics_flat = rearrange(intrinsics, "b v i j -> (b v) i j")
        
        # Convert extrinsics to view matrices (w2c)
        viewmats = self._convert_extrinsics_to_viewmat(extrinsics_flat)  # [BV, 4, 4]
        
        # Extract focal lengths and principal points from intrinsics
        fx = intrinsics_flat[:, 0, 0] * width
        fy = intrinsics_flat[:, 1, 1] * height
        cx = intrinsics_flat[:, 0, 2] * width
        cy = intrinsics_flat[:, 1, 2] * height
        
        # Repeat for all views
        means_rep = repeat(means, "b g xyz -> (b v) g xyz", v=v)
        scales_rep = repeat(scales, "b g xyz -> (b v) g xyz", v=v)
        quats_rep = repeat(quats, "b g q -> (b v) g q", v=v)
        colors_rep = repeat(colors, "b g c -> (b v) g c", v=v)
        opacities_rep = repeat(opacities, "b g -> (b v) g", v=v)
        
        bv = b * v
        
        # Render each view
        output_colors = []
        output_depths = []
        
        for i in range(bv):
            # Prepare camera parameters for this view
            K = torch.tensor([
                [fx[i], 0, cx[i]],
                [0, fy[i], cy[i]],
                [0, 0, 1]
            ], dtype=torch.float32, device=means.device)
            
            # Render with gsplat
            render_colors, render_alphas, meta = rasterization(
                means=means_rep[i],
                quats=quats_rep[i],
                scales=scales_rep[i],
                opacities=opacities_rep[i],
                colors=colors_rep[i],
                viewmats=viewmats[i:i+1],
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                packed=False,
                render_mode="RGB",
            )
            
            # render_colors is [1, H, W, 3], we need [3, H, W]
            color = render_colors[0].permute(2, 0, 1)
            
            # Apply background color
            alpha = render_alphas[0, ..., 0]
            bg_color = self.background_color.view(3, 1, 1)
            color = color * alpha.unsqueeze(0) + bg_color * (1 - alpha.unsqueeze(0))
            
            output_colors.append(color)
            output_depths.append(torch.zeros(height, width, device=means.device))
        
        # Stack outputs
        colors = torch.stack(output_colors, dim=0)
        depths = torch.stack(output_depths, dim=0)
        
        # Reshape to [B, V, 3, H, W] and [B, V, H, W]
        colors = rearrange(colors, "(b v) c h w -> b v c h w", b=b, v=v)
        depths = rearrange(depths, "(b v) h w -> b v h w", b=b, v=v)
        
        return DecoderOutput(color=colors, depth=depths)
    
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: str | None = None,
        cam_rot_delta=None,
        cam_trans_delta=None,
    ) -> DecoderOutput:
        """
        Render Gaussians using gsplat.
        
        Args:
            gaussians: Gaussians dataclass with means, covariances, harmonics, opacities
            extrinsics: [batch, view, 4, 4] camera extrinsics (c2w)
            intrinsics: [batch, view, 3, 3] camera intrinsics
            near: [batch, view] near plane
            far: [batch, view] far plane
            image_shape: (height, width) output image size
            depth_mode: optional depth rendering mode (not used in this simple version)
            
        Returns:
            DecoderOutput with color and depth
        """
        b, v, _, _ = extrinsics.shape
        height, width = image_shape
        
        # Flatten batch and view dimensions
        extrinsics_flat = rearrange(extrinsics, "b v i j -> (b v) i j")
        intrinsics_flat = rearrange(intrinsics, "b v i j -> (b v) i j")
        
        # Convert extrinsics to view matrices (w2c)
        viewmats = self._convert_extrinsics_to_viewmat(extrinsics_flat)  # [BV, 4, 4]
        
        # Extract focal lengths and principal points from intrinsics
        # Intrinsics format: [[fx*w, 0, cx*w], [0, fy*h, cy*h], [0, 0, 1]]
        fx = intrinsics_flat[:, 0, 0] * width  # [BV]
        fy = intrinsics_flat[:, 1, 1] * height  # [BV]
        cx = intrinsics_flat[:, 0, 2] * width  # [BV]
        cy = intrinsics_flat[:, 1, 2] * height  # [BV]
        
        # Prepare Gaussian parameters
        # Repeat Gaussians for all views
        means = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v)  # [BV, N, 3]
        covariances = repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v)  # [BV, N, 3, 3]
        harmonics = repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v)  # [BV, N, 3, d_sh]
        opacities = repeat(gaussians.opacities, "b g -> (b v) g", v=v)  # [BV, N]
        
        # Convert covariances to scales and quaternions for gsplat
        bv, num_gaussians, _, _ = covariances.shape
        covariances_flat = rearrange(covariances, "bv n i j -> (bv n) i j")
        scales_flat, quats_flat = self._covariance_to_scale_rotation(covariances_flat)
        scales = rearrange(scales_flat, "(bv n) xyz -> bv n xyz", bv=bv, n=num_gaussians)
        quats = rearrange(quats_flat, "(bv n) q -> bv n q", bv=bv, n=num_gaussians)
        
        # Convert SH to RGB colors
        harmonics_flat = rearrange(harmonics, "bv n c d_sh -> (bv n) c d_sh")
        colors_flat = self._sh_to_rgb(harmonics_flat)
        colors = rearrange(colors_flat, "(bv n) c -> bv n c", bv=bv, n=num_gaussians)
        
        # Render each view
        output_colors = []
        output_depths = []
        
        for i in range(bv):
            # Prepare camera parameters for this view
            K = torch.tensor([
                [fx[i], 0, cx[i]],
                [0, fy[i], cy[i]],
                [0, 0, 1]
            ], dtype=torch.float32, device=means.device)
            
            # Render with gsplat
            render_colors, render_alphas, meta = rasterization(
                means=means[i],  # [N, 3]
                quats=quats[i],  # [N, 4]
                scales=scales[i],  # [N, 3]
                opacities=opacities[i],  # [N]
                colors=colors[i],  # [N, 3]
                viewmats=viewmats[i:i+1],  # [1, 4, 4]
                Ks=K.unsqueeze(0),  # [1, 3, 3]
                width=width,
                height=height,
                packed=False,
                render_mode="RGB",
            )
            
            # render_colors is [1, H, W, 3], we need [1, 3, H, W]
            color = render_colors[0].permute(2, 0, 1)  # [3, H, W]
            
            # Apply background color
            alpha = render_alphas[0, ..., 0]  # [H, W]
            bg_color = self.background_color.view(3, 1, 1)  # [3, 1, 1]
            color = color * alpha.unsqueeze(0) + bg_color * (1 - alpha.unsqueeze(0))
            
            output_colors.append(color)
            
            # For depth, we can use the meta information if needed
            # For now, return None or zeros
            output_depths.append(torch.zeros(height, width, device=means.device))
        
        # Stack outputs
        colors = torch.stack(output_colors, dim=0)  # [BV, 3, H, W]
        depths = torch.stack(output_depths, dim=0)  # [BV, H, W]
        
        # Reshape to [B, V, 3, H, W] and [B, V, H, W]
        colors = rearrange(colors, "(b v) c h w -> b v c h w", b=b, v=v)
        depths = rearrange(depths, "(b v) h w -> b v h w", b=b, v=v)
        
        return DecoderOutput(color=colors, depth=depths)
