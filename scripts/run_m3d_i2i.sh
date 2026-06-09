# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

output_dir=output/i2i_case1
export PYTHONPATH=$(pwd)/code/MoGe:$PYTHONPATH
echo $PYTHONPATH


python code/panoramic_image_generation.py \
    --mode=i2p \
    --input_image_path="data/case1.png" \
    --output_path=$output_dir

torchrun --nproc_per_node 1 code/panoramic_image_to_video.py \
  --inout_dir=$output_dir  \
  --resolution=720 \
    --use_5b_model

python code/panoramic_video_to_3DScene.py \
    --inout_dir=$output_dir \
    --resolution=720

python code/render_gs.py --input $output_dir/generated_3dgs_opt.ply --output $output_dir/rendered_video.mp4
