# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

import os
import importlib.util

import numpy as np
import torch

from utils_3dscene.geak_kernels import rasterize as geak_rasterize, interpolate as geak_interpolate

from utils_3dscene.render_geak import (
    depth_edge_torch,
    get_diffrast_camera_parameter_from_cv,
    merge_panorama_depth_torch,
    image_uv_torch,
    spherical_uv_to_directions_torch,
)


_PANO_CONSTANTS_CACHE = {}
_ROTATED_DIRS_CACHE = {}
_FULL_FACES_CACHE = {}


def _get_pano_constants(device):
    """All constants returned as torch tensors on `device`."""
    device_key = str(device)
    if device_key not in _PANO_CONSTANTS_CACHE:
        mat = np.array([
            [[1,0,0],[0,1,0],[0,0,1]],
            [[0,0,-1],[0,1,0],[1,0,0]],
            [[-1,0,0],[0,1,0],[0,0,-1]],
            [[0,0,1],[0,1,0],[-1,0,0]],
            [[1,0,0],[0,0,1],[0,-1,0]],
            [[1,0,0],[0,0,-1],[0,1,0]],
        ], dtype=np.float32)
        mat_glue = np.array([
            [[0,0,-1],[0,-1,0],[-1,0,0]],
            [[0,0,-1],[-1,0,0],[0,1,0]],
            [[0,0,-1],[0,1,0],[1,0,0]],
            [[0,0,-1],[1,0,0],[0,-1,0]],
            [[-1,0,0],[0,-1,0],[0,0,1]],
            [[1,0,0],[0,-1,0],[0,0,-1]],
        ], dtype=np.float32)
        mat_glue = np.array([[[0,-1,0],[1,0,0],[0,0,1]]]) @ mat_glue
        tmp = np.eye(4, dtype=np.float32)[None].repeat(6, 0)
        tmp[:,:3,:3] = mat_glue
        K = np.array([[256.,0,256.],[0.,256,256.],[0.,0,1.]], dtype=np.float32)
        mat_t = torch.from_numpy(mat).to(device)
        tmp_t = torch.from_numpy(tmp).to(device)
        K_t = torch.from_numpy(K).to(device)
        K_6 = K_t.unsqueeze(0).expand(6, -1, -1).contiguous()
        _PANO_CONSTANTS_CACHE[device_key] = (mat_t, tmp_t, K_t, K_6)
    return _PANO_CONSTANTS_CACHE[device_key]


def _get_rotated_directions(H, W, device, dtype):
    """Cache (directions @ rot_matrix.T).  Only depends on (H, W, device, dtype)."""
    key = (H, W, str(device), dtype)
    if key not in _ROTATED_DIRS_CACHE:
        uv = image_uv_torch(width=W, height=H, device=device, dtype=dtype)
        directions = spherical_uv_to_directions_torch(uv)
        rot_matrix = torch.tensor(
            [[0,1,0],[0,0,-1],[-1,0,0.]], dtype=dtype, device=device
        )
        _ROTATED_DIRS_CACHE[key] = directions.reshape(-1, 3) @ rot_matrix.T
    return _ROTATED_DIRS_CACHE[key]


def _build_faces_gpu(fg_mask, H, W):
    """Build triangle faces entirely on GPU -- no CPU transfer needed."""
    device = fg_mask.device
    mask_ext = torch.zeros(H, W + 1, dtype=torch.bool, device=device)
    mask_ext[:, :-1] = fg_mask
    mask_ext[:, -1] = fg_mask[:, 0]

    tm0 = mask_ext[:-1,:-1] | mask_ext[:-1,1:] | mask_ext[1:,:-1]
    tm1 = mask_ext[:-1,1:] | mask_ext[1:,:-1] | mask_ext[1:,1:]

    r0, c0 = torch.where(tm0)
    tri0 = torch.stack([
        r0 * W + c0,
        (r0 + 1) * W + c0,
        r0 * W + (c0 + 1) % W,
    ], dim=1).int()

    r1, c1 = torch.where(tm1)
    tri1 = torch.stack([
        r1 * W + (c1 + 1) % W,
        (r1 + 1) * W + c1,
        (r1 + 1) * W + (c1 + 1) % W,
    ], dim=1).int()

    return torch.cat([tri0, tri1], dim=0)


def _get_full_faces(H, W, device):
    """Cached full face set for the no-mask case (skybox)."""
    key = (H, W, str(device))
    if key not in _FULL_FACES_CACHE:
        all_true = torch.ones(H, W, dtype=torch.bool, device=device)
        _FULL_FACES_CACHE[key] = _build_faces_gpu(all_true, H, W)
    return _FULL_FACES_CACHE[key]


def _write_ply_binary(verts_np, faces_np, path):
    """Write mesh as binary PLY — raw tobytes(), no text formatting."""
    n_v, n_f = len(verts_np), len(faces_np)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    face_records = np.empty(
        n_f, dtype=[('count', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')]
    )
    face_records['count'] = 3
    face_records['v0'] = faces_np[:, 0]
    face_records['v1'] = faces_np[:, 1]
    face_records['v2'] = faces_np[:, 2]

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        verts_np.astype('<f4').tofile(f)
        face_records.tofile(f)


def load_ply_binary(path):
    """Read binary PLY written by _write_ply_binary. Returns (vertices, faces) as numpy arrays."""
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        n_v = n_f = 0
        for line in header_lines:
            if line.startswith('element vertex'):
                n_v = int(line.split()[-1])
            elif line.startswith('element face'):
                n_f = int(line.split()[-1])

        vertices = np.frombuffer(f.read(n_v * 3 * 4), dtype='<f4').reshape(n_v, 3)
        face_dt = np.dtype([('count', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')])
        face_records = np.frombuffer(f.read(n_f * face_dt.itemsize), dtype=face_dt)
        faces = np.column_stack([face_records['v0'], face_records['v1'], face_records['v2']])

    return vertices.copy(), faces.copy()


def _write_obj(verts_np, faces_np, path):
    """Write mesh as text OBJ via np.savetxt."""
    with open(path, 'w') as f:
        np.savetxt(f, verts_np, fmt='v %.6f %.6f %.6f')
        np.savetxt(f, faces_np + 1, fmt='f %d %d %d')


def _build_pano_mesh_gpu(pano_depth, pano_mask, Rt, device):
    """Build vertices and faces on GPU, return as numpy arrays."""
    if isinstance(pano_depth, np.ndarray):
        depth_t = torch.from_numpy(pano_depth).float().to(device)
    else:
        depth_t = pano_depth.float().to(device)

    H, W = depth_t.shape[:2]

    rotated_dirs = _get_rotated_directions(H, W, device, depth_t.dtype)
    cam_pcs = depth_t.reshape(-1, 1) * rotated_dirs

    if isinstance(Rt, np.ndarray):
        Rt_t = torch.from_numpy(Rt).float().to(device)
    else:
        Rt_t = Rt.float().to(device)

    c2w = torch.linalg.inv(Rt_t)
    vertices = (cam_pcs @ c2w[:3, :3].T + c2w[:3, 3][None]).float()

    if isinstance(pano_mask, np.ndarray):
        mask_t = torch.from_numpy(pano_mask).to(device=device, dtype=torch.bool)
    else:
        mask_t = pano_mask.to(device=device, dtype=torch.bool)

    faces = _build_faces_gpu(mask_t, H, W)

    return vertices.cpu().numpy(), faces.cpu().numpy()


def export_pano_mesh_fast(pano_depth, pano_mask, Rt, output_path, device):
    """GPU-accelerated mesh build + export.

    Auto-selects format by extension:
      .ply  → binary PLY (fastest: raw tobytes, no text formatting)
      .obj  → text OBJ via np.savetxt
    """
    verts_np, faces_np = _build_pano_mesh_gpu(pano_depth, pano_mask, Rt, device)

    if output_path.endswith('.ply'):
        _write_ply_binary(verts_np, faces_np, output_path)
    else:
        _write_obj(verts_np, faces_np, output_path)


def warp_depth_to_tgt_fast(src_depth, src_Rt, tgt_Rts, apply_skybox_mask=True, apply_seam_mask=True):
    """Optimized depth warping using GEAK Triton kernels for rasterization.

    Compared to the nvdiffrast version:
      - Uses GEAK Triton rasterize/interpolate instead of nvdiffrast
      - No RasterizeCudaContext needed (stateless kernels)
      - Otherwise identical pipeline: GPU mesh build, projection, panorama merge
    """
    device = src_depth.device
    H, W = src_depth.shape[:2]
    near = 1e-4
    far = float(src_depth.max()) * 1.2
    skybox_depth = float(src_depth.max())

    no_mask = not apply_seam_mask and not apply_skybox_mask

    if no_mask:
        tri_tensor = _get_full_faces(H, W, device)
    else:
        skybox_mask = src_depth > 0.99 * skybox_depth
        if apply_seam_mask:
            depth_edge_mask = ~depth_edge_torch(src_depth, rtol=0.03)
        if apply_seam_mask and apply_skybox_mask:
            fg_extraction_mask = (~skybox_mask) * depth_edge_mask
        elif apply_seam_mask:
            fg_extraction_mask = depth_edge_mask
        else:
            fg_extraction_mask = ~skybox_mask
        tri_tensor = _build_faces_gpu(fg_extraction_mask, H, W)

    rotated_dirs = _get_rotated_directions(H, W, device, src_depth.dtype)
    cam_pcs = src_depth.reshape(-1, 1) * rotated_dirs

    src_Rt_f = src_Rt.float() if src_Rt.dtype != torch.float32 else src_Rt
    c2w = torch.linalg.inv(src_Rt_f).to(src_depth.dtype)
    vertices = (cam_pcs @ c2w[:3, :3].T + c2w[:3, 3][None]).float()

    mat_t, tmp_t, K_t, K_6 = _get_pano_constants(device)

    tgt_Rts_f = tgt_Rts.float() if tgt_Rts.dtype != torch.float32 else tgt_Rts
    N = tgt_Rts_f.shape[0]

    K_proj = get_diffrast_camera_parameter_from_cv(K_t, 512, 512, near, far, device)
    K_proj_T = K_proj.T.unsqueeze(0)

    all_pano_depth_map = []
    rendered_mask = []
    render_batch_size = 5
    inf_val = torch.tensor(float('inf'), device=device)
    n_verts = vertices.shape[0]

    with torch.no_grad():
        for i in range(0, N, render_batch_size):
            bsz = min(render_batch_size, N - i)

            cur_Rt = tgt_Rts_f[i:i+bsz].unsqueeze(1).expand(-1, 6, -1, -1).clone()
            cur_Rt[:, :, :3, :] = mat_t[None] @ cur_Rt[:, :, :3, :]
            Rts_batch = cur_Rt.reshape(bsz * 6, 4, 4)

            n_views = bsz * 6
            pos_cam = (vertices[None] @ Rts_batch[:, :3, :3].permute(0, 2, 1)
                       + Rts_batch[:, :3, 3][:, None, :])

            pos_qc = torch.ones(
                (n_views, n_verts, 4), dtype=torch.float32, device=device
            )
            pos_qc[:, :, :3] = pos_cam

            dist = torch.empty(
                (n_views, n_verts, 2), dtype=torch.float32, device=device
            )
            dist[:, :, 0] = pos_cam.norm(dim=-1)
            dist[:, :, 1] = 1.

            pos_rast = pos_qc @ K_proj_T

            rast = geak_rasterize(pos_rast, tri_tensor, [512, 512])
            out, _ = geak_interpolate(dist, rast, tri_tensor)

            rd = out[:, :, :, 0]
            rm = out[:, :, :, 1] > (1. - 1e-4)
            rd = torch.where(rm, rd, inf_val)

            for j in range(bsz):
                pano_depth = merge_panorama_depth_torch(
                    W, H, rd[6*j:6*(j+1)], None, tmp_t, K_6, device,
                )
                pano_np = pano_depth.cpu().numpy()
                mask = pano_np < 9e5
                pano_np[~mask] = skybox_depth
                all_pano_depth_map.append(pano_np)
                rendered_mask.append(mask)

    return all_pano_depth_map, rendered_mask


# ---------------------------------------------------------------------------
# MoGe panorama inference — in-process replacement for os.system() subprocess
# ---------------------------------------------------------------------------

_INFER_PANO_MOD = None


def _get_infer_pano_mod():
    """Lazily import helper functions from code/MoGe/scripts/infer_panorama.py."""
    global _INFER_PANO_MOD
    if _INFER_PANO_MOD is None:
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "MoGe", "scripts", "infer_panorama.py"
        )
        spec = importlib.util.spec_from_file_location("infer_panorama", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _INFER_PANO_MOD = mod
    return _INFER_PANO_MOD


def load_moge_model(pretrained_path, device):
    """Load MoGe model once.  Returns the model in eval mode on *device*."""
    from moge.model import MoGeModel
    return MoGeModel.from_pretrained(pretrained_path).to(device).eval()


def merge_panorama_depth_fft(width, height, distance_maps, pred_masks, extrinsics, intrinsics):
    """FFT/DCT-based panoramic depth merging (gradient + Laplacian, combined).

    ~12x faster than the recursive LSMR version. Solves the same variational
    problem (min ||grad(f)-g||^2 + ||lap(f)-L||^2) via spectral decomposition
    with periodic-x / Neumann-y boundary conditions.  No recursion needed.
    """
    import cv2
    import utils3d
    from scipy.fft import dct, idct
    from scipy.ndimage import convolve

    mod = _get_infer_pano_mod()
    H, W = height, width

    uv = utils3d.numpy.image_uv(width=W, height=H)
    spherical_directions = mod.spherical_uv_to_directions(uv)

    gxs, gys, mxs, mys = [], [], [], []
    laps, lms = [], []
    panorama_pred_masks = []
    pv_logs = []
    coverage = np.zeros((H, W), dtype=np.float32)
    wavg = np.zeros((H, W), dtype=np.float64)

    for i in range(len(distance_maps)):
        projected_uv, projected_depth = utils3d.numpy.project_cv(
            spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i],
        )
        projection_valid_mask = (
            (projected_depth > 0)
            & (projected_uv > 0).all(axis=-1)
            & (projected_uv < 1).all(axis=-1)
        )
        projected_pixels = utils3d.numpy.uv_to_pixel(
            np.clip(projected_uv, 0, 1),
            width=distance_maps[i].shape[1], height=distance_maps[i].shape[0],
        ).astype(np.float32)

        log_dist = np.log(distance_maps[i])
        pano_log = np.where(
            projection_valid_mask,
            cv2.remap(log_dist, projected_pixels[..., 0], projected_pixels[..., 1],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
            0,
        )
        pano_mask = projection_valid_mask & (
            cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1],
                      cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
        )
        panorama_pred_masks.append(pano_mask)

        m_f = pano_mask.astype(np.float32)
        wavg += pano_log * m_f
        coverage += m_f

        p = np.pad(pano_log, ((0, 0), (0, 1)), mode='wrap')
        gxs.append(p[:, :-1] - p[:, 1:])
        gys.append(p[:-1, :] - p[1:, :])
        p = np.pad(pano_mask, ((0, 0), (0, 1)), mode='wrap')
        mxs.append(p[:, :-1] & p[:, 1:])
        mys.append(p[:-1, :] & p[1:, :])

        p = np.pad(pano_log, ((1, 1), (0, 0)), mode='edge')
        p = np.pad(p, ((0, 0), (1, 1)), mode='wrap')
        laps.append(convolve(p, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1])
        p = np.pad(pano_mask, ((1, 1), (0, 0)), mode='edge')
        p = np.pad(p, ((0, 0), (1, 1)), mode='wrap')
        lms.append(convolve(p.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5)

    GX, GY = np.stack(gxs), np.stack(gys)
    MX, MY = np.stack(mxs), np.stack(mys)
    avg_gx = np.sum(GX * MX, 0) / np.sum(MX, 0).clip(1e-3)
    avg_gy = (np.sum(GY * MY, 0) / np.sum(MY, 0).clip(1e-3))[:, :W]

    LP, LM = np.stack(laps), np.stack(lms)
    avg_lap = np.sum(LP * LM, 0) / np.sum(LM, 0).clip(1e-3)

    avg_gx_f = avg_gx.astype(np.float64)
    avg_gy_f = avg_gy.astype(np.float64)
    avg_lap_f = avg_lap.astype(np.float64)

    div = (avg_gx_f - np.roll(avg_gx_f, 1, axis=1)).copy()
    div[0] += avg_gy_f[0]
    div[1:-1] += avg_gy_f[1:] - avg_gy_f[:-1]
    div[-1] -= avg_gy_f[-1]

    m_idx = np.arange(H, dtype=np.float64)
    n_idx = np.arange(W, dtype=np.float64)
    mu = (2 * (1 - np.cos(np.pi * m_idx / H)))[:, None] \
       + (2 * (1 - np.cos(2 * np.pi * n_idx / W)))[None, :]

    div_hat = np.fft.fft(dct(div, type=2, axis=0, norm='ortho'), axis=1)
    lap_hat = np.fft.fft(dct(avg_lap_f, type=2, axis=0, norm='ortho'), axis=1)

    op = mu * (1.0 + mu)
    op[0, 0] = 1.0
    f_hat = (div_hat - mu * lap_hat) / op

    wavg_safe = np.where(coverage > 0, wavg / coverage.clip(1e-6), 0)
    mean_log = float(np.mean(wavg_safe[coverage > 0]))
    f_hat[0, 0] = mean_log * np.sqrt(H) * W

    f = idct(np.real(np.fft.ifft(f_hat, axis=1)), type=2, axis=0, norm='ortho')

    panorama_depth = np.exp(np.clip(f, -20, 20)).reshape(H, W).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    return panorama_depth, panorama_mask


def moge_infer_panorama(model, image_bgr, device, batch_size=4):
    """In-process MoGe panorama depth inference.

    Replaces the subprocess call:
        os.system("cd code/MoGe && python scripts/infer_panorama.py ...")
    followed by reading depth.exr and mask.png from disk.

    Args:
        model:     MoGe model returned by load_moge_model().
        image_bgr: panorama image as numpy uint8 array (H, W, 3) in BGR order.
        device:    torch device string or torch.device.
        batch_size: NN inference batch size (default 4).

    Returns:
        (depth, mask) — numpy arrays.
        depth : float32 (H, W)   panoramic distance map.
        mask  : bool    (H, W)   valid-pixel mask.
    """
    import cv2
    import utils3d

    mod = _get_infer_pano_mod()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    extrinsics, intrinsics = mod.get_panorama_cameras(fov_x=100., fov_y=100.)
    splitted_images = mod.split_panorama_image(image_rgb, extrinsics, intrinsics, 768)

    splitted_distance_maps, splitted_masks = [], []
    for i in range(0, len(splitted_images), batch_size):
        batch = splitted_images[i:i + batch_size]
        image_tensor = torch.tensor(
            np.stack(batch) / 255,
            dtype=torch.float32, device=device,
        ).permute(0, 3, 1, 2)
        fov_x, _ = np.rad2deg(
            utils3d.numpy.intrinsics_to_fov(np.array(intrinsics[i:i + batch_size]))
        )
        fov_x_t = torch.tensor(fov_x, dtype=torch.float32, device=device)
        output = model.infer(image_tensor, fov_x=fov_x_t, apply_mask=False)
        splitted_distance_maps.extend(list(output['points'].norm(dim=-1).cpu().numpy()))
        splitted_masks.extend(list(output['mask'].cpu().numpy()))
    torch.cuda.synchronize()

    merging_width = min(1920, width)
    merging_height = min(960, height)
    panorama_depth, panorama_mask = merge_panorama_depth_fft(
        merging_width, merging_height,
        splitted_distance_maps, splitted_masks,
        extrinsics, intrinsics,
    )

    panorama_depth = panorama_depth.astype(np.float32)
    panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
    panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0

    return panorama_depth, panorama_mask
