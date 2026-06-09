#!/bin/bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
set -e  # Stop on any error

# Fix git "dubious ownership" errors that occur when cloning inside Docker
# (files owned by a different uid than the current user).
git config --global --add safe.directory '*'
apt install cmake

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

mkdir -p libs
cd libs

# Install flash-attention from local source.
# First run compiles HIP kernels (~30 min). Subsequent runs reuse the build cache.
echo "Installing flash-attention..."
if [ ! -d "flash-attention" ]; then
    git clone https://github.com/Dao-AILab/flash-attention
fi
if ! python -c "import flash_attn" 2>/dev/null; then
    cd flash-attention
    MAX_JOBS=16 python setup.py install
    cd ..
else
    echo "flash-attention already installed, skipping build."
fi


if [ ! -d "simple-knn" ]; then
    git clone https://github.com/amd-wangfan/simple-knn
fi
cd simple-knn
pip install -e . --no-build-isolation
touch simple_knn/__init__.py  # required for editable install to resolve the package
cd ..


if [ ! -d "taming-transformers" ]; then
    git clone https://github.com/CompVis/taming-transformers.git
fi
cd taming-transformers
find taming -type d -exec touch {}/__init__.py \;
pip install -e . --no-build-isolation
cd ..

# Build gsplat from source (ROCm fork). HIP kernel compilation takes ~20-30 min.
echo "Installing gsplat (ROCm, build from source)..."
if [ ! -d "gsplat" ]; then
    git clone --recurse-submodules https://github.com/rocm/gsplat.git
fi
if ! python -c "import gsplat" 2>/dev/null; then
    cd gsplat/gsplat/cuda/csrc/third_party/glm
    cmake -DGLM_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -B build . -DCMAKE_INSTALL_PREFIX:PATH=~/.local
    cmake --build build -- all
    cmake --build build -- install
    cd ../../../../..  # back to libs/gsplat
    MAX_JOBS=16 python setup.py bdist_wheel
    pip install dist/amd_gsplat*.whl
    cd ..
else
    echo "gsplat already installed, skipping build."
fi

echo "Installing Python packages..."
pip install diffusers transformers==4.56.0 modelscope einops opencv-python open3d py360convert scikit-image==0.25.2 plyfile \
            peft decord ffmpeg protobuf sentencepiece trimesh xfuser OmegaConf \
            jaxtyping SwissArmyTransformer wandb imageio[ffmpeg] pyrender pytorch-lightning==1.4.2 \
            torchmetrics==0.7.0 openai-clip kornia open-clip-torch==2.7.0 fairscale loralib 



cd "$ROOT_DIR"

cd code/DiffSynth-Studio/
pip install -e . --no-build-isolation
pip uninstall -y apex

echo "Pre download models"
cd "$ROOT_DIR"
python code/download_checkpoints.py

echo "✓ Installation complete!"
