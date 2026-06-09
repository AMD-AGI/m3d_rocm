import os
import sys
base_dir = "/mnt/workspace/zhongqi.yang/VideoInpainting_new/ViewCrafter/extern/MoGe/example_images/pano_examples"
cases = os.listdir(base_dir)
for c in cases:
    os.system(f"python scripts/infer_panorama.py --input {os.path.join(base_dir, c)} --output /mnt/workspace/zhongqi.yang/VideoInpainting_new/ViewCrafter/extern/MoGe/outputs/pano/ --maps --glb --ply")