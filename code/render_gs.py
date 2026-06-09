# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

from gsplat_renderer import GsplatRenderer
from plyfile import PlyData
import torch
import imageio
import numpy as np
import math
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
        description="A simple argparse example"
    )
parser.add_argument(
        "--input",
        type=str,
        default=None,
    )
parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
args = parser.parse_args()

os.makedirs('output_videos', exist_ok=True)


renderer = GsplatRenderer(make_scale_invariant=True, background_color=[1., 1., 1.]).to(torch.device('cuda'))
renderer.eval()


num_frames = 40
K = torch.tensor(np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
cam_pose = torch.tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
cam_pose = cam_pose.repeat(1, num_frames, 1, 1)

DIST = 1
for i in range(num_frames):
    cam_pose[:, i, 2, 3] += float(i)/num_frames*DIST


last_pose = cam_pose[:, -1:]          # [B,1,4,4]
B = last_pose.size(0)
num_views = 72

eye3 = torch.eye(3, device=last_pose.device, dtype=last_pose.dtype).view(1,1,3,3).expand_as(last_pose[..., :3, :3])
last_pose[..., :3, :3] = eye3         

# Angles (no repetition of start and end)
angles = torch.arange(num_views, device=last_pose.device, dtype=last_pose.dtype) / num_views * 2 * math.pi
if num_views > 1:
    angles = angles[:-1]                   # [N]
N = angles.numel()

# Rotation matrix around world Y-axis (4x4)
c, s = torch.cos(angles), torch.sin(angles)
rot_mats = torch.zeros(N, 4, 4, device=last_pose.device, dtype=last_pose.dtype)
rot_mats[:, 0, 0] =  c;  rot_mats[:, 0, 2] =  s
rot_mats[:, 1, 1] =  1
rot_mats[:, 2, 0] = -s;  rot_mats[:, 2, 2] =  c
rot_mats[:, 3, 3] =  1
rot_mats = rot_mats.unsqueeze(0)           # [1,N,4,4]

# Apply rotations (keep original position, only rotate orientation; if want to rotate around camera's own axis/orbit, change multiplication order or translation)
new_poses = last_pose @ rot_mats           # [B,N,4,4]

cam_pose = torch.cat([cam_pose, new_poses], dim=1)


K = K.repeat(1, cam_pose.shape[1], 1, 1)
near = torch.tensor(0.0001).repeat(cam_pose.shape[1]).unsqueeze(0)
far = torch.tensor(30.).repeat(cam_pose.shape[1]).unsqueeze(0)

K = K.cuda()
cam_pose = cam_pose.cuda()
near = near.cuda()
far = far.cuda()

def quaternion_to_rotation_matrix(q):
    """
    q: [N, 4] (w, x, y, z)
    returns: [N, 3, 3]
    """
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
    ], dim=-1).reshape(-1, 3, 3)

    return R


def load_gaussians_from_ply(
    ply_path: str,
    device: str = "cuda",
    use_covariances: bool = False
):
    """
    Load Gaussians from PLY file.
    
    Args:
        ply_path: Path to PLY file
        device: Device to load tensors to
        use_covariances: If True, returns Gaussians dataclass with covariances (slow).
                        If False, returns tuple (means, scales, quats, colors, opacities) (fast).
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    N = len(v)

    # --------------------------------------------------
    # Means
    # --------------------------------------------------
    means = torch.from_numpy(
        np.stack([v["x"], v["y"], v["z"]], axis=1)
    ).float()

    # --------------------------------------------------
    # Opacity (inverse sigmoid in PLY)
    # --------------------------------------------------
    opacities = torch.from_numpy(v["opacity"]).float()
    opacities = torch.sigmoid(opacities)

    # --------------------------------------------------
    # Scales (log-space in PLY)
    # --------------------------------------------------
    scales = torch.from_numpy(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    ).float()
    scales = torch.exp(scales)

    # --------------------------------------------------
    # Rotation (quaternions in PLY)
    # --------------------------------------------------
    quats = torch.from_numpy(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    ).float()
    quats = quats / quats.norm(dim=1, keepdim=True)

    # --------------------------------------------------
    # SH degree 0 (DC only) -> RGB
    # --------------------------------------------------
    # Convert SH to RGB directly for faster rendering
    C0 = 0.28209479177387814
    colors = torch.zeros((N, 3), dtype=torch.float32)
    colors[:, 0] = torch.from_numpy(v["f_dc_0"]) * C0 + 0.5
    colors[:, 1] = torch.from_numpy(v["f_dc_1"]) * C0 + 0.5
    colors[:, 2] = torch.from_numpy(v["f_dc_2"]) * C0 + 0.5
    
    # Add batch dimension and move to device
    means = means.unsqueeze(0).to(device)
    scales = scales.unsqueeze(0).to(device)
    quats = quats.unsqueeze(0).to(device)
    colors = colors.unsqueeze(0).to(device)
    opacities = opacities.unsqueeze(0).to(device)

    if use_covariances:
        # For backward compatibility - compute covariances (SLOW!)
        R = quaternion_to_rotation_matrix(quats.squeeze(0))
        S = torch.diag_embed(scales.squeeze(0) ** 2)
        covariances = R @ S @ R.transpose(1, 2)
        
        harmonics = torch.zeros((N, 3, 1), dtype=torch.float32)
        harmonics[:, 0, 0] = torch.from_numpy(v["f_dc_0"])
        harmonics[:, 1, 0] = torch.from_numpy(v["f_dc_1"])
        harmonics[:, 2, 0] = torch.from_numpy(v["f_dc_2"])
        
        return Gaussians(
            means=means,
            covariances=covariances.unsqueeze(0).to(device),
            harmonics=harmonics.unsqueeze(0).to(device),
            opacities=opacities,
        )
    else:
        # Return direct parameters (FAST!)
        return means, scales, quats, colors, opacities


# Load Gaussians directly (fast path - no covariance computation)
means, scales, quats, colors, opacities = load_gaussians_from_ply(args.input, use_covariances=False)
   

video_frames = []

with torch.cuda.amp.autocast(dtype=torch.float):
    for i in tqdm(range(cam_pose.shape[1])):
        render_output = renderer.forward_direct(
            means,
            scales,
            quats,
            colors,
            opacities,
            cam_pose[:, i:i+1],   # keep batch dim
            K[:, i:i+1],
            near[:, i:i+1],
            far[:, i:i+1],
            (512, 512),
            depth_mode=None,
        )

        frames = render_output.color[0]  # [C, H, W]

        frame = frames.permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().cpu().numpy()
        

        video_frames.append(frame)

video_frames = np.concatenate(video_frames, axis=0)  # [N, H, W, 3]

# Save as MP4 (auto-select encoder)
imageio.mimwrite(args.output,  video_frames, fps=12, quality=10, macro_block_size=None)  # avoid size alignment issues
print("Rendering complete")
        



                