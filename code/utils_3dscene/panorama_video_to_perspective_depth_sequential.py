# Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import os
import sys
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
sys.path.append("code/MoGe")
sys.path.append("code")
from utils_3dscene.pipeline_utils_3dscene import get_video_frames, warp_depth_to_tgt, depth_edge, optimize_depth, optimize_depth_v2
from utils_3dscene.render_geak import get_mesh_from_pano_Rt, depth_edge_torch
from utils_3dscene.helper_funcs import warp_depth_to_tgt_fast, _build_pano_mesh_gpu, _write_ply_binary, load_moge_model, moge_infer_panorama, merge_panorama_depth_fft
import argparse
import numpy as np
import torch
import cv2
OPTIMAL_SPLIT_FRAME_SIZE = 49


# 
#moge_model_path = os.path.abspath("./code/MoGe/checkpoints/model.pt")
moge_model_path = os.path.abspath("checkpoints/moge/model.pt")
def apply_warp_fix(warped_depth, warped_mask):
    warped_depth_valid = warped_depth[warped_mask]
    warped_depth[~warped_mask] = warped_depth_valid.max() * 2.
    return warped_depth
def main(args):

    device = args.device
    video_path = args.video_path
    camera_path = args.camera_path
    anchor_frame_depth_paths = args.anchor_frame_depth_paths
    anchor_frame_mask_paths = args.anchor_frame_mask_paths
    anchor_frame_indices = args.anchor_frame_indices

    output_dir = args.output_dir
    depth_estimation_interval = args.depth_estimation_interval
    # each frame is cut into 15views. which is fixed. 
    width = args.width
    height = args.height
    
    video_frames = get_video_frames(video_path)
    anchor_depths = [cv2.resize(cv2.imread(i, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH),(width,height)) for i in anchor_frame_depth_paths]
    anchor_masks = []
    for i,p in enumerate(anchor_frame_mask_paths):
        if os.path.exists(p):
            anchor_masks.append(cv2.resize(cv2.imread(p, cv2.IMREAD_UNCHANGED),(width,height))>127)
        else:
            anchor_masks.append(anchor_depths[i] < 0.9 * anchor_depths[i].max())

    all_cameras = np.load(camera_path)["arr_0"]

    all_generated_frames = []
    N = len(video_frames)
    N_anchors = len(anchor_frame_indices)
    os.makedirs(output_dir, exist_ok=True)
    moge_output_dir = os.path.join(output_dir, "moge")
    data_output_dir = os.path.join(output_dir, "data")
    optimized_depth_dir = os.path.join(output_dir, "data", "optimized_depths")
    mv_rgb_dir = os.path.join(output_dir, "data", "mv_rgb")
    mv_depth_dir = os.path.join(output_dir, "data", "mv_depth")
    os.makedirs(moge_output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(optimized_depth_dir, exist_ok=True)
    os.makedirs(mv_rgb_dir, exist_ok=True)
    os.makedirs(mv_depth_dir, exist_ok=True)

    moge_model = load_moge_model(moge_model_path, device)

    last_optimized_depth = []
    last_optimized_mask = []
    last_optimized_Rt = []
    for i in range(N):
        if i in anchor_frame_indices:
            cur_frame = video_frames[i]
            anchor_index = -1
            anchor_depth = None
            anchor_mask = None
            for j in range(N_anchors):
                if anchor_frame_indices[j] == i:
                    anchor_index = anchor_frame_indices[j]
                    anchor_depth = anchor_depths[j]
                    anchor_mask = anchor_masks[j]
                    break
            
            cur_camera = all_cameras[i]

            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}.exr"), cv2.resize(anchor_depth,(width, height)))
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}_mask.png"), cv2.resize(anchor_mask.astype(np.uint8)*255,(width, height)))
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}_rgb.png"),cv2.resize(cur_frame,(width,height)))

            last_optimized_depth.append(cv2.resize(anchor_depth,(width, height)))
            last_optimized_mask.append(cv2.resize(anchor_mask.astype(np.uint8)*255,(width, height))>127)
            last_optimized_Rt.append(cur_camera)
        elif i % depth_estimation_interval == 0:
            cur_frame = video_frames[i]
            if len(last_optimized_depth) > 0:
                anchor_depth = last_optimized_depth[-1]
                anchor_mask = last_optimized_mask[-1]
                anchor_camera = last_optimized_Rt[-1]
            cur_camera = all_cameras[i]
            
            # --- MoGe inference ---
            cur_depth, cur_fgmask = moge_infer_panorama(
                moge_model,
                cv2.resize(cur_frame, (width, height)),
                device,
            )

            # --- depth_edge ---
            cur_seam_mask = ~depth_edge(cur_depth, rtol=0.05)


            # import nvdiffrast.torch as dr
            # glctx = dr.RasterizeCudaContext(device=device)
            # --- warp_depth_to_tgt ---
            anchor_warp_apply_fg_mask = (~anchor_mask).sum() > 1000
            warped_depth, warped_depth_mask = warp_depth_to_tgt_fast(torch.from_numpy(anchor_depth).to(device), torch.from_numpy(anchor_camera).to(device), torch.from_numpy(cur_camera).to(device)[None], apply_skybox_mask=anchor_warp_apply_fg_mask)

            
            
            # --- get_mesh (warped) ---
            warped_verts, warped_faces = _build_pano_mesh_gpu(warped_depth[0], warped_depth_mask[0], cur_camera, device)
            _write_ply_binary(warped_verts, warped_faces, os.path.join(optimized_depth_dir, f"{i:04d}_warped_mesh.ply"))

            cur_depth_fixed = apply_warp_fix(cur_depth, cur_fgmask)

            # --- optimize_depth ---
            # optimized_depth, optimized_mask = optimize_depth(warped_depth[0],cur_depth_fixed, warped_depth_mask[0], cur_seam_mask, cur_fgmask)
            optimized_depth, optimized_mask = optimize_depth_v2(warped_depth[0],cur_depth_fixed, warped_depth_mask[0], cur_seam_mask, cur_fgmask)

            # --- depth_edge + mesh (optimized) ---
            optimized_depth_edge = (~depth_edge(optimized_depth, rtol=0.05))
            opt_verts, opt_faces = _build_pano_mesh_gpu(optimized_depth, warped_depth_mask[0] * optimized_depth_edge, cur_camera, device)
            _write_ply_binary(opt_verts, opt_faces, os.path.join(optimized_depth_dir, f"{i:04d}_mesh_estim.ply"))

            # --- skybox warp ---
            skybox_depth = torch.from_numpy(np.ones_like(anchor_depth) * anchor_depth.max() * 2).to(device)
            skybox_warped_depth, _ = warp_depth_to_tgt_fast(skybox_depth, torch.from_numpy(anchor_camera).to(device), torch.from_numpy(cur_camera).to(device)[None], apply_skybox_mask=False, apply_seam_mask=False)
            optimized_depth[~optimized_mask] = skybox_warped_depth[0][~optimized_mask]

            # --- save I/O ---
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}.exr"), cv2.resize(optimized_depth,(width, height)))
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}_mask.png"), cv2.resize(optimized_mask.astype(np.uint8)*255,(width, height)))
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}_rgb.png"),cv2.resize(cur_frame,(width,height)))

            last_optimized_depth.append(cv2.resize(optimized_depth,(width, height)))
            last_optimized_mask.append(cv2.resize(optimized_mask.astype(np.uint8)*255,(width, height))>127)
            last_optimized_Rt.append(cur_camera)

if __name__ == "__main__":
    '''
        device = args.device
        video_path = args.video_path
        camera_path = args.camera_path
        anchor_frame_depth_paths = args.anchor_frame_depth_paths
        anchor_frame_indices = args.anchor_frame_indices

        output_dir = args.output_dir
        depth_estimation_interval = args.depth_estimation_interval
        # each frame is cut into 15views. which is fixed. 
        width = args.width
        height = args.height
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", type=str, default="cuda:1")
    #
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--anchor_frame_depth_paths", type=str, nargs="+", required=True)
    parser.add_argument("--anchor_frame_mask_paths", type=str, nargs="+", required=True)
    parser.add_argument("--anchor_frame_indices", type=int, nargs="+", default=[0])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--depth_estimation_interval", type=int, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=960)

    args = parser.parse_args()
    main(args)
            

    
    