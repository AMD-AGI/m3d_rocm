# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
One-time preprocessing step after downloading OSEDiff models.
Strips optimizer state from the RAM checkpoint and saves model weights
only in safetensors format, reducing file size from ~5.6GB to ~2GB
and load time from ~22s to ~2s.

Usage:
    python preprocess_models.py
    python preprocess_models.py --ram_path /path/to/ram_swin_large_14m.pth
"""

import argparse
import os
import time
import torch
from safetensors.torch import save_file

def preprocess_ram(ram_path: str):
    out_path = os.path.splitext(ram_path)[0] + '_model_only.safetensors'
    if os.path.exists(out_path):
        print(f'Already exists, skipping: {out_path}')
        return out_path

    print(f'Loading {ram_path} ({os.path.getsize(ram_path)/1e6:.0f} MB)...')
    t0 = time.time()
    ckpt = torch.load(ram_path, map_location='cpu', weights_only=False)
    print(f'  Loaded in {time.time()-t0:.1f}s')

    state_dict = ckpt['model']
    # safetensors requires contiguous, non-shared tensors
    state_dict = {k: v.clone().contiguous().float() for k, v in state_dict.items()}

    print(f'Saving model-only weights to {out_path}...')
    t1 = time.time()
    save_file(state_dict, out_path)
    print(f'  Saved in {time.time()-t1:.1f}s ({os.path.getsize(out_path)/1e6:.0f} MB)')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ram_path', type=str,
                        default='preset/models/ram_swin_large_14m.pth')
    args = parser.parse_args()

    t0 = time.time()
    out = preprocess_ram(args.ram_path)
    print(f'\nDone in {time.time()-t0:.1f}s total. Preprocessed model: {out}')
