# Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import sys
sys.path.append("code")
import re
import glob
import subprocess
from multiprocessing import Pool, cpu_count

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from utils_3dscene.pipeline_utils_3dscene import generate_panovideo_data, csv_cam_to_opencv, data_convert
import cv2
import numpy as np
import torch
import math
import utils3d
import trimesh
import torchvision
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import pandas as pd

def read_cam_csv(pose_file):
    cam_params = []
    with open(pose_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            frame_data = line.strip().split(',')
            if len(frame_data) != 3:
                continue
            
            # Parse position
            pos = frame_data[1].split()
            X = float(pos[0].split('=')[1])
            Y = float(pos[1].split('=')[1])
            Z = float(pos[2].split('=')[1])
            
            
            # Parse rotation
            rot = frame_data[2].split()
            P = float(rot[0].split('=')[1])
            Yaw = float(rot[1].split('=')[1])
            R = float(rot[2].split('=')[1])
            
            cam_params.append((X, Y, Z, P, Yaw, R))
    return cam_params


def generate_fit_data_new(args):
    optimized_depth_dir = args.optimized_depth_dir
    camera_path = args.camera_path
    output_dir = args.output_dir

    all_files = os.listdir(optimized_depth_dir)
    all_exr = [i for i in all_files if i.endswith("exr")]
    all_exr.sort()
    all_Rts_ = np.load(camera_path)["arr_0"]
    N = len(all_exr)
    all_rgbs = []
    all_depths = []
    all_Rts = []

    for i in range(N):
        cur_exr_name = all_exr[i]
        cur_index = int(cur_exr_name.split(".")[0])
        cur_exr_path = os.path.join(optimized_depth_dir, cur_exr_name)
        cur_rgb_path = cur_exr_path.replace(".exr","_rgb.png")

        cur_exr = cv2.imread(cur_exr_path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        cur_rgb = cv2.imread(cur_rgb_path, cv2.IMREAD_UNCHANGED)
        cur_camera = all_Rts_[cur_index]

        all_rgbs.append(cur_rgb)
        all_depths.append(cur_exr)
        all_Rts.append(cur_camera)

    all_rgbs = np.stack(all_rgbs, axis=0)
    all_depths = np.stack(all_depths, axis=0)
    Rts = np.stack(all_Rts)

    K = np.array([
        [512.,0,512.],
        [0.,512,512.],
        [0.,0,1.],
    ])
    #splitted_images, generate_panovideo_data(all_rgbs[:5], all_depths[:5], K, Rts[:5], (512,512), 4)
    all_splitted_images, all_splitted_depths, all_splitted_Rts, all_splitted_Ks = generate_panovideo_data(all_rgbs[:], all_depths[:], K, Rts[:], (1024,1024), 1,correct_pano_depth_=False)
    splitted_image_dir = os.path.join(output_dir, "mv_rgb_orig")
    splitted_depth_dir = os.path.join(output_dir, "mv_depth")

    os.makedirs(splitted_image_dir, exist_ok=True)
    os.makedirs(splitted_depth_dir, exist_ok=True)
    all_splitted_Rts = np.stack(all_splitted_Rts,axis=0)
    # print(all_splitted_Rts.shape)
    # np.savez(os.path.join(output_dir, "world_matrix_new.npz"), all_splitted_Rts)

    all_splitted_c2ws = np.linalg.inv(all_splitted_Rts)
    pos = all_splitted_c2ws[:,:3,3]
    pos_mean = pos.mean(axis=0)
    pos_scale = np.linalg.norm(pos[-1] - pos[0]) / 10.

    all_splitted_c2ws[:,:3,3] = (all_splitted_c2ws[:,:3,3] - pos_mean)/pos_scale
    all_splitted_Rts = np.linalg.inv(all_splitted_c2ws)
    N_ = len(all_splitted_depths)
    for j in range(N_):
        all_splitted_depths[j] /= pos_scale
    # np.savez(os.path.join(output_dir, "world_matrix_new.npz"), all_splitted_Rts)
    # exit()
    

    N_split = len(all_splitted_images)
    for j in range(N_split):
        #print(all_splitted_depths[0].dtype,all_splitted_depths[0].shape)
        cv2.imwrite(os.path.join(splitted_image_dir, f"{j:04d}.png"), (all_splitted_images[j]).astype(np.uint8))
        cv2.imwrite(os.path.join(splitted_depth_dir, f"{j:04d}.exr"), (all_splitted_depths[j]))
    with open(os.path.join(output_dir, "para.json"),"w" ) as F_:
        dic = {
            "image_width": int(K[0,0]*2),
            "image_height": int(K[0,0]*2),
            "focal_length_in_pixel": K[0,0]
        }
        F_.write(json.dumps(dic,indent=4))

    arr = data_convert(np.stack(all_splitted_Rts,axis=0))
    np.savez(os.path.join(output_dir, "world_matrix.npz"), arr)
    print("split complete")

    

if __name__ == "__main__":
    '''
    optimized_depth_dir = args.optimized_depth_dir
    camera_path = args.camera_path
    output_dir = args.output_dir
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimized_depth_dir', type=str, default='/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output/A_small_anime_village_with_thatched-roof_houses,_a_windmill,_and_a_field_of_flowers_stretching_to_the_horizon,_ultra-detailed,_warm_lighting,_cozy_atmosphere_superres/geom_optim/data/optimized_depths')
    parser.add_argument('--camera_path', type=str, default='/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output/A_small_anime_village_with_thatched-roof_houses,_a_windmill,_and_a_field_of_flowers_stretching_to_the_horizon,_ultra-detailed,_warm_lighting,_cozy_atmosphere_superres/condition/cameras.npz')
    parser.add_argument('--output_dir', type=str, default='/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output/A_small_anime_village_with_thatched-roof_houses,_a_windmill,_and_a_field_of_flowers_stretching_to_the_horizon,_ultra-detailed,_warm_lighting,_cozy_atmosphere_superres/geom_optim/data')
    args = parser.parse_args()
    generate_fit_data_new(args)