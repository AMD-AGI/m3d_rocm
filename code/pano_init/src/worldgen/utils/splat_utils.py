
import numpy as np
import torch
from plyfile import PlyData, PlyElement


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions (w, x, y, z)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q = torch.zeros(batch_dim + (4,), dtype=matrix.dtype, device=matrix.device)
    trace = m00 + m11 + m22
    s = torch.zeros_like(trace)
    cond0 = trace > 0
    s[cond0] = 0.5 / torch.sqrt(trace[cond0] + 1.0)
    q[cond0, 0] = 0.25 / s[cond0]
    q[cond0, 1] = (m21[cond0] - m12[cond0]) * s[cond0]
    q[cond0, 2] = (m02[cond0] - m20[cond0]) * s[cond0]
    q[cond0, 3] = (m10[cond0] - m01[cond0]) * s[cond0]
    cond1 = (~cond0) & (m00 > m11) & (m00 > m22)
    s[cond1] = 2.0 * torch.sqrt(1.0 + m00[cond1] - m11[cond1] - m22[cond1])
    q[cond1, 0] = (m21[cond1] - m12[cond1]) / s[cond1]
    q[cond1, 1] = 0.25 * s[cond1]
    q[cond1, 2] = (m01[cond1] + m10[cond1]) / s[cond1]
    q[cond1, 3] = (m02[cond1] + m20[cond1]) / s[cond1]
    cond2 = (~cond0) & (~cond1) & (m11 > m22)
    s[cond2] = 2.0 * torch.sqrt(1.0 + m11[cond2] - m00[cond2] - m22[cond2])
    q[cond2, 0] = (m02[cond2] - m20[cond2]) / s[cond2]
    q[cond2, 1] = (m01[cond2] + m10[cond2]) / s[cond2]
    q[cond2, 2] = 0.25 * s[cond2]
    q[cond2, 3] = (m12[cond2] + m21[cond2]) / s[cond2]
    cond3 = (~cond0) & (~cond1) & (~cond2)
    s[cond3] = 2.0 * torch.sqrt(1.0 + m22[cond3] - m00[cond3] - m11[cond3])
    q[cond3, 0] = (m10[cond3] - m01[cond3]) / s[cond3]
    q[cond3, 1] = (m02[cond3] + m20[cond3]) / s[cond3]
    q[cond3, 2] = (m12[cond3] + m21[cond3]) / s[cond3]
    q[cond3, 3] = 0.25 * s[cond3]
    return q

class SplatFile:
    def __init__(
        self,
        centers: np.ndarray,
        rgbs: np.ndarray,
        opacities: np.ndarray,
        covariances: np.ndarray,
        rotations: np.ndarray,
        scales: np.ndarray,
    ):
        self.centers = centers # (N, 3)
        self.rgbs = rgbs # (N, 3)
        self.opacities = opacities # (N, 1)
        self.covariances = covariances # (N, 3, 3)
        self.rotations = rotations # (N, 4) # quaternion wxyz
        self.scales = scales # (N, 3)

    def save(self, path: str):
        xyz = self.centers
        normals = np.zeros_like(xyz)
        f_dc = (self.rgbs - 0.5) / 0.28209479177387814 # convert to SH coefficients
        opacities = self.opacities
        scale = np.log(self.scales)
        rotation = self.rotations

        attribute_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(f_dc.shape[1]):
            attribute_names.append('f_dc_{}'.format(i))
        attribute_names.append('opacity')
        for i in range(scale.shape[1]):
            attribute_names.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            attribute_names.append('rot_{}'.format(i))

        dtype_full = [(name, 'f4') for name in attribute_names]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, opacities, scale, rotation),
            axis=1
        )
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


def convert_rgbd_to_gs(rgb, distance, rays, dis_threshold=0., epsilon=1e-3) -> SplatFile:
    """
    Given an equirectangular RGB-D image, back-project each pixel to a 3D point
    and compute the corresponding 3D Gaussian covariance so that the projection covers 1 pixel.

    Parameters:
        rgb (H x W x 3): RGB image as torch.Tensor, uint8
        distance (H x W): Distance map (in meters) as torch.Tensor, float32
        rays (H x W x 3): Ray directions as torch.Tensor, float32
        epsilon (float): Small Z-scale for the splat in ray direction

    Returns:
        centers (N x 3): 3D positions of splats
        covariances (N x 3 x 3): 3D Gaussian covariances of splats
        colors (N x 3): RGB values of splats
        opacities (N x 1): Opacities of splats
        scales (N x 3): Scales of splats
        rotations (N x 4): Rotations of splats
    """
    H, W = rgb.shape[:2]
    device = rgb.device

    valid_mask = distance > dis_threshold
    rays_flat = rays.view(-1, 3)
    distance_flat = distance.view(-1)
    valid_rays = rays_flat[valid_mask.view(-1)]
    valid_distance = distance_flat[valid_mask.view(-1)]
    centers = valid_rays * valid_distance[:, None]

    delta_phi = 2 * torch.pi / W
    delta_theta = torch.pi / H
    sigma_x = valid_distance * delta_phi 
    sigma_y = valid_distance * delta_theta 
    sigma_z = torch.ones_like(valid_distance) * epsilon

    S = torch.stack([sigma_x, sigma_y, sigma_z], dim=1)
    covariances = torch.einsum('ni,nj->nij', S, S)  # Sigma = S @ S.T        # (N, 3, 3)

    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    x_axis = torch.nn.functional.normalize(torch.cross(up, valid_rays), dim=1)
    fallback_up = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    degenerate_mask = torch.isnan(x_axis).any(dim=1)
    x_axis[degenerate_mask] = torch.nn.functional.normalize(torch.cross(fallback_up[degenerate_mask], valid_rays[degenerate_mask]), dim=1)
    y_axis = torch.nn.functional.normalize(torch.cross(valid_rays, x_axis), dim=1)
    z_axis = valid_rays

    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3)

    # Step 5: apply covariance transformation: Sigma = R S S^T R^T
    S_matrices = torch.zeros((S.shape[0], 3, 3), device=device)
    S_matrices[:, 0, 0] = S[:, 0]
    S_matrices[:, 1, 1] = S[:, 1]
    S_matrices[:, 2, 2] = S[:, 2]

    covariances = R @ S_matrices @ S_matrices.transpose(1, 2) @ R.transpose(1, 2)
    colors = rgb.view(-1, 3).float() / 255.0
    opacities = torch.ones((centers.shape[0], 1))
    rotation = matrix_to_quaternion(R)

    return SplatFile(
        centers=centers.cpu().numpy(),
        covariances=covariances.cpu().numpy(),
        rgbs=colors.cpu().numpy(),
        opacities=opacities.cpu().numpy(),
        rotations=rotation.cpu().numpy(),
        scales=S.cpu().numpy(),
    )

def mask_splat(splat: SplatFile, mask: np.ndarray) -> SplatFile:
    H, W = mask.shape
    valid_mask = mask>0
    centers = splat.centers
    covariances = splat.covariances
    rgbs = splat.rgbs
    opacity = splat.opacities
    scales = splat.scales
    rotations = splat.rotations

    centers = centers.reshape(H, W, 3)[valid_mask]
    covariances = covariances.reshape(H, W, 3, 3)[valid_mask]
    rgbs = rgbs.reshape(H, W, 3)[valid_mask]
    opacity = opacity.reshape(H, W, 1)[valid_mask]
    scales = scales.reshape(H, W, 3)[valid_mask]
    rotations = rotations.reshape(H, W, 4)[valid_mask]

    splat = {
        "centers": centers,
        "covariances": covariances,
        "rgbs": rgbs,
        "opacities": opacity,
        "scales": scales,
        "rotations": rotations
    }
    return SplatFile(**splat)

def merge_splats(splat1: SplatFile, splat2: SplatFile) -> SplatFile:
    return SplatFile(
        centers=np.concatenate([splat1.centers, splat2.centers], axis=0),
        covariances=np.concatenate([splat1.covariances, splat2.covariances], axis=0),
        rgbs=np.concatenate([splat1.rgbs, splat2.rgbs], axis=0),
        opacities=np.concatenate([splat1.opacities, splat2.opacities], axis=0),
        scales=np.concatenate([splat1.scales, splat2.scales], axis=0),
        rotations=np.concatenate([splat1.rotations, splat2.rotations], axis=0)
    )