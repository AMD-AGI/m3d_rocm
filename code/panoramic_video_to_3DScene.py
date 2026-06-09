# Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.append("./DiffSynth-Studio")
import argparse
import cv2
        
def main(args):
    device = args.device
    step1_output_dir = os.path.abspath(args.inout_dir)

    prompt_path = os.path.join(args.inout_dir, 'prompt.txt')

    with open(os.path.abspath(prompt_path),"r",encoding="utf-8") as f:
        prompt=f.read()
        print(f"prompt is {prompt}")

    generated_dir = os.path.join(step1_output_dir, "generated")
    condition_dir = os.path.join(step1_output_dir, "condition")
    
    generated_video_path = os.path.join(generated_dir,"generated.mp4")
    width_following = 1440
    height_following = 720


        
    camera_path = os.path.join(condition_dir,"cameras.npz")
    os.system(f"python code/utils_3dscene/panorama_video_to_perspective_depth_sequential.py \
        --device {device} \
        --camera_path {camera_path} \
        --video_path {generated_video_path} \
        --anchor_frame_depth_paths \'{os.path.join(condition_dir,'firstframe_depth.exr')}\' \
        --anchor_frame_mask_paths \'{os.path.join(condition_dir,'firstframe_mask.png')}\' \
        --anchor_frame_indices 0 \
        --output_dir {os.path.join(step1_output_dir,'geom_optim')} \
        --depth_estimation_interval 10 \
        --width {width_following} \
        --height {height_following} \
    ")

    os.system(
        f"python code/utils_3dscene/gs_optim_datagen.py \
            --optimized_depth_dir {os.path.join(step1_output_dir,'geom_optim/data/optimized_depths')} \
            --camera_path {os.path.join(step1_output_dir,'condition/cameras.npz')} \
            --output_dir {os.path.join(step1_output_dir,'geom_optim/data')} \
        "
    )


    cmd = f"python code/enhance_with_osediff.py --inout_dir {step1_output_dir}"
    os.system(cmd)


    gs_input_dir = os.path.join(step1_output_dir, 'geom_optim/data')


    flags = (
        f"-s {gs_input_dir} "
        "--save_iterations 3000 6000 9000 12000 15000 --test_iterations 3000 "
        "--sh_degree 0 --densify_from_iter 500 --densify_until_iter 1501 --iterations 3000 --eval "
        f"--img_sample_interval 1 --num_views_per_view 3 --num_of_point_cloud 3000000 --device {device} "
    )

    gs_output_dir = os.path.join(step1_output_dir, 'geom_optim/output')
    cmd = f"cd ./code/pano_gs && python train.py -m {gs_output_dir} {flags}"

    os.system(cmd)

    compact_src = os.path.join(gs_output_dir, 'point_cloud/compact/point_cloud.ply')
    iter_src = os.path.join(gs_output_dir, 'point_cloud/iteration_3000/point_cloud.ply')
    src = compact_src if os.path.exists(compact_src) else iter_src
    dst = os.path.join(step1_output_dir, 'generated_3dgs_opt.ply')
    if os.path.exists(src):
        os.system(f"cp {src} {dst}")
        print(f"Copied output PLY -> {dst}")
    else:
        print(f"WARNING: {src} not found, skipping")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="the device on which the 3d scene generation runs")
    parser.add_argument("--inout_dir", type=str, default="./output/example1", help="the directory storing the input and output result")
    parser.add_argument("--resolution", type=int, default=720, help="the working resolution of the 3D scene generation")
    parser.add_argument("--enhance_input_dir", type=str, default=None, help="Override input dir for OSEDiff enhancement (default: <inout_dir>/geom_optim/data/mv_rgb_orig)")
    args = parser.parse_args()
    

    main(args)
