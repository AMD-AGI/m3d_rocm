import os
import sys
# 
base_dir = "/mnt/workspace/zhongqi.yang/VideoInpainting_new/output/case_forward_moving/vc_output/mv_rgb"
cases = os.listdir(base_dir)
begin_index = 36
for i in range(begin_index, 61):
    os.system(f"python scripts/infer.py --input {os.path.join(base_dir, f'{i:04d}.png')} --output /mnt/workspace/zhongqi.yang/VideoInpainting_new/output/moge_perview_estimate/ --maps --glb --ply")