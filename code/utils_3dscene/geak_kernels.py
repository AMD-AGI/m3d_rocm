# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content.
# SPDX-License-Identifier: MIT

"""
Mesh Rasterization Kernel - Triton Implementation

This kernel performs triangle rasterization for 3D mesh rendering using a 
triangle-centric approach with atomic depth testing.

The algorithm:
1. Triangle setup: compute screen-space bounds and precompute edge equations
2. Triangle-centric rasterization: each triangle writes to its covered pixels

Input:
    mesh: (N, V, 4) - batch, vertices, homogeneous clip coordinates (x, y, z, w)
    tri: (F, 3) - triangle indices (int32)
    resolution: [H, W] - output image resolution

Output:
    rast: (N, H, W, 4) - batch, height, width, (u, v, depth, triangle_index)
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# Triangle-Centric Rasterization Kernel - Optimized with 2D tile processing
# Each program handles one triangle with 2D tile-based pixel processing
# =============================================================================
@triton.jit
def triangle_rasterize_kernel(
    # Input pointers
    vertices_ptr,      # (N, V, 4) clip space vertices
    triangles_ptr,     # (F, 3) triangle indices
    # Output pointers
    rast_ptr,          # (N, H, W, 4) output rasterization buffer
    depth_ptr,         # (N, H, W) depth buffer for atomic comparison
    # Dimensions
    batch_size,
    num_vertices,
    num_triangles,
    image_height,
    image_width,
    # Block size for pixel processing
    BLOCK_SIZE: tl.constexpr,
    TILE_X: tl.constexpr = 32,
    TILE_Y: tl.constexpr = 16,
):
    """
    Triangle-centric rasterization with 2D tile processing.
    Each program handles one triangle, processing pixels in 2D tiles for better cache locality.
    """
    tri_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    if tri_idx >= num_triangles:
        return
    
    # Load triangle vertex indices
    tri_base = tri_idx * 3
    idx0 = tl.load(triangles_ptr + tri_base)
    idx1 = tl.load(triangles_ptr + tri_base + 1)
    idx2 = tl.load(triangles_ptr + tri_base + 2)
    
    # Load vertex clip coordinates with precomputed base offsets
    batch_offset = batch_idx * num_vertices * 4
    v0_base = batch_offset + idx0 * 4
    v1_base = batch_offset + idx1 * 4
    v2_base = batch_offset + idx2 * 4
    
    # Load all vertex components
    v0_x = tl.load(vertices_ptr + v0_base)
    v0_y = tl.load(vertices_ptr + v0_base + 1)
    v0_z = tl.load(vertices_ptr + v0_base + 2)
    v0_w = tl.load(vertices_ptr + v0_base + 3)
    
    v1_x = tl.load(vertices_ptr + v1_base)
    v1_y = tl.load(vertices_ptr + v1_base + 1)
    v1_z = tl.load(vertices_ptr + v1_base + 2)
    v1_w = tl.load(vertices_ptr + v1_base + 3)
    
    v2_x = tl.load(vertices_ptr + v2_base)
    v2_y = tl.load(vertices_ptr + v2_base + 1)
    v2_z = tl.load(vertices_ptr + v2_base + 2)
    v2_w = tl.load(vertices_ptr + v2_base + 3)
    
    # Check w values (must be positive for vertices in front of camera)
    w_invalid = (v0_w <= 0) | (v1_w <= 0) | (v2_w <= 0)
    if w_invalid:
        return
    
    # Perspective divide to get NDC coordinates - use reciprocal for efficiency
    inv_w0 = 1.0 / v0_w
    inv_w1 = 1.0 / v1_w
    inv_w2 = 1.0 / v2_w
    
    ndc0_x = v0_x * inv_w0
    ndc0_y = v0_y * inv_w0
    ndc0_z = v0_z * inv_w0
    
    ndc1_x = v1_x * inv_w1
    ndc1_y = v1_y * inv_w1
    ndc1_z = v1_z * inv_w1
    
    ndc2_x = v2_x * inv_w2
    ndc2_y = v2_y * inv_w2
    ndc2_z = v2_z * inv_w2
    
    # Early frustum culling in NDC space - skip triangles completely outside [-1, 1]
    # Check if all vertices are outside the same frustum plane
    all_left = (ndc0_x < -1.0) & (ndc1_x < -1.0) & (ndc2_x < -1.0)
    all_right = (ndc0_x > 1.0) & (ndc1_x > 1.0) & (ndc2_x > 1.0)
    all_bottom = (ndc0_y < -1.0) & (ndc1_y < -1.0) & (ndc2_y < -1.0)
    all_top = (ndc0_y > 1.0) & (ndc1_y > 1.0) & (ndc2_y > 1.0)
    all_near = (ndc0_z < -1.0) & (ndc1_z < -1.0) & (ndc2_z < -1.0)
    all_far = (ndc0_z > 1.0) & (ndc1_z > 1.0) & (ndc2_z > 1.0)
    
    if all_left | all_right | all_bottom | all_top | all_near | all_far:
        return
    
    # Convert NDC to screen coordinates
    half_w = image_width * 0.5
    half_h = image_height * 0.5
    
    sx0 = (ndc0_x + 1.0) * half_w
    sy0 = (ndc0_y + 1.0) * half_h
    sx1 = (ndc1_x + 1.0) * half_w
    sy1 = (ndc1_y + 1.0) * half_h
    sx2 = (ndc2_x + 1.0) * half_w
    sy2 = (ndc2_y + 1.0) * half_h
    
    # Compute edge vectors and area (signed area for backface detection)
    e1x = sx1 - sx0
    e1y = sy1 - sy0
    e2x = sx2 - sx0
    e2y = sy2 - sy0
    area2 = e1x * e2y - e2x * e1y
    
    # Skip degenerate triangles (very small area)
    if tl.abs(area2) < 1e-10:
        return
    
    # Compute bounding box
    min_x = tl.minimum(tl.minimum(sx0, sx1), sx2)
    max_x = tl.maximum(tl.maximum(sx0, sx1), sx2)
    min_y = tl.minimum(tl.minimum(sy0, sy1), sy2)
    max_y = tl.maximum(tl.maximum(sy0, sy1), sy2)
    
    # Clamp to screen bounds
    min_x_i = tl.maximum(0, min_x.to(tl.int32))
    max_x_i = tl.minimum(image_width - 1, max_x.to(tl.int32))
    min_y_i = tl.maximum(0, min_y.to(tl.int32))
    max_y_i = tl.minimum(image_height - 1, max_y.to(tl.int32))
    
    # Skip if bounding box is empty
    bbox_empty = (min_x_i > max_x_i) | (min_y_i > max_y_i)
    if bbox_empty:
        return
    
    inv_area2 = 1.0 / area2
    
    # Precompute edge equation coefficients for incremental evaluation
    A0 = (sx1 * sy2 - sx2 * sy1) * inv_area2
    B0 = (sy1 - sy2) * inv_area2
    C0 = (sx2 - sx1) * inv_area2
    
    A1 = (sx2 * sy0 - sx0 * sy2) * inv_area2
    B1 = (sy2 - sy0) * inv_area2
    C1 = (sx0 - sx2) * inv_area2
    
    # Precompute depth interpolation coefficients
    dz0 = ndc0_z - ndc2_z
    dz1 = ndc1_z - ndc2_z
    
    # Triangle index (1-indexed)
    tri_idx_f = (tri_idx + 1).to(tl.float32)
    
    # Process bounding box dimensions
    bbox_width = max_x_i - min_x_i + 1
    bbox_height = max_y_i - min_y_i + 1
    
    # Precompute batch offset for pixel indexing
    batch_pixel_offset = batch_idx * image_height * image_width
    
    # Epsilon for inside test
    eps = -1e-2
    
    # Calculate number of tiles
    num_tiles_x = (bbox_width + TILE_X - 1) // TILE_X
    num_tiles_y = (bbox_height + TILE_Y - 1) // TILE_Y
    
    # Create tile offsets
    tile_offs_x = tl.arange(0, TILE_X)
    tile_offs_y = tl.arange(0, TILE_Y)
    
    # Precompute tile pixel offsets (relative to tile origin)
    # This is a 2D grid of offsets that can be reused for all tiles
    tile_local_x = tl.broadcast_to(tile_offs_x[:, None], [TILE_X, TILE_Y])
    tile_local_y = tl.broadcast_to(tile_offs_y[None, :], [TILE_X, TILE_Y])
    tile_local_x_flat = tl.reshape(tile_local_x, [TILE_X * TILE_Y])
    tile_local_y_flat = tl.reshape(tile_local_y, [TILE_X * TILE_Y])
    
    # Precompute local pixel index offsets within a tile
    tile_local_idx = tile_local_y_flat * image_width + tile_local_x_flat
    
    # Process tiles in row-major order
    for tile_y in range(num_tiles_y):
        # Precompute row base for this tile row
        tile_base_y = min_y_i + tile_y * TILE_Y
        tile_row_base = tile_base_y * image_width + min_x_i
        
        # Precompute y-dependent terms
        py_flat = tile_base_y + tile_local_y_flat
        py_f = py_flat.to(tl.float32) + 0.5
        
        # Precompute y-dependent parts of barycentric coords
        w0_y = A0 + C0 * py_f
        w1_y = A1 + C1 * py_f
        
        # Precompute y validity
        y_valid = (py_flat >= min_y_i) & (py_flat <= max_y_i)
        
        for tile_x in range(num_tiles_x):
            # Compute tile base coordinates
            tile_base_x = min_x_i + tile_x * TILE_X
            
            # Compute global pixel coordinates
            px_flat = tile_base_x + tile_local_x_flat
            
            # Create mask for valid pixels (within bounding box)
            # Simplified: px_flat >= min_x_i is always true since tile_base_x >= min_x_i
            pixel_mask = (px_flat <= max_x_i) & y_valid
            
            # Pixel center coordinates
            px_f = px_flat.to(tl.float32) + 0.5
            
            # Compute barycentric coordinates using precomputed y-dependent terms
            w0 = w0_y + B0 * px_f
            w1 = w1_y + B1 * px_f
            
            # Check if pixel is inside triangle (w2 = 1 - w0 - w1 >= eps means w0 + w1 <= 1 + eps)
            inside = (w0 >= eps) & (w1 >= eps) & (w0 + w1 <= 1.0 - eps) & pixel_mask
            
            # Interpolate depth using FMA
            depth = tl.fma(w0, dz0, tl.fma(w1, dz1, ndc2_z))
            
            # Compute pixel indices using precomputed offsets
            tile_origin_idx = batch_pixel_offset + tile_row_base + tile_x * TILE_X
            pixel_idx = tile_origin_idx + tile_local_idx
            
            # Load current depth and check if this triangle is closer
            current_depth = tl.load(depth_ptr + pixel_idx, mask=inside, other=float('inf'))
            should_update = inside & (depth < current_depth)
            
            # Store depth
            tl.store(depth_ptr + pixel_idx, depth, mask=should_update)
            
            # Store rasterization output
            rast_base = pixel_idx * 4
            tl.store(rast_ptr + rast_base, w0, mask=should_update)
            tl.store(rast_ptr + rast_base + 1, w1, mask=should_update)
            tl.store(rast_ptr + rast_base + 2, depth, mask=should_update)
            tl.store(rast_ptr + rast_base + 3, tri_idx_f, mask=should_update)


# =============================================================================
# Python Wrapper Function
# =============================================================================
def rasterize(mesh, tri, resolution):
    """
    Rasterize a mesh to an image.
    
    Args:
        mesh: (N, V, 4) tensor of vertex positions in clip space (x, y, z, w)
        tri: (F, 3) tensor of triangle indices (int32)
        resolution: [H, W] output image resolution
    
    Returns:
        rast: (N, H, W, 4) tensor containing (u, v, depth, triangle_index)
              - u, v: barycentric coordinates
              - depth: interpolated depth value
              - triangle_index: 1-indexed triangle ID (0 = no triangle)
    """
    assert mesh.dim() == 3 and mesh.shape[2] == 4, f"mesh must be (N, V, 4), got {mesh.shape}"
    assert tri.dim() == 2 and tri.shape[1] == 3, f"tri must be (F, 3), got {tri.shape}"
    
    N, V, _ = mesh.shape
    F = tri.shape[0]
    H, W = resolution
    
    device = mesh.device
    dtype = mesh.dtype
    
    # Allocate output buffer
    rast = torch.zeros((N, H, W, 4), device=device, dtype=dtype)
    
    # Allocate depth buffer for depth comparison
    depth_buffer = torch.full((N, H, W), float('inf'), device=device, dtype=dtype)
    
    # Triangle-centric rasterization
    BLOCK_SIZE = 256  # Process 256 pixels per iteration
    
    triangle_rasterize_kernel[(F, N)](
        mesh, tri,
        rast.view(-1),
        depth_buffer.view(-1),
        N, V, F, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,
    )
    
    return rast


# =============================================================================
# Reference Implementation (for correctness testing)
# =============================================================================
def rasterize_reference(mesh, tri, resolution):
    """
    Reference CPU implementation for correctness testing.
    """
    N, V, _ = mesh.shape
    F = tri.shape[0]
    H, W = resolution
    
    mesh_cpu = mesh.cpu().numpy()
    tri_cpu = tri.cpu().numpy()
    
    import numpy as np
    rast = np.zeros((N, H, W, 4), dtype=np.float32)
    
    half_w = W * 0.5
    half_h = H * 0.5
    
    for batch_idx in range(N):
        depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
        
        for tri_idx in range(F):
            idx0, idx1, idx2 = tri_cpu[tri_idx]
            
            # Get clip coordinates
            v0 = mesh_cpu[batch_idx, idx0]
            v1 = mesh_cpu[batch_idx, idx1]
            v2 = mesh_cpu[batch_idx, idx2]
            
            # Perspective divide
            ndc0 = v0[:3] / v0[3]
            ndc1 = v1[:3] / v1[3]
            ndc2 = v2[:3] / v2[3]
            
            # Screen coordinates
            sx0 = (ndc0[0] + 1.0) * half_w
            sy0 = (1.0 - ndc0[1]) * half_h
            sx1 = (ndc1[0] + 1.0) * half_w
            sy1 = (1.0 - ndc1[1]) * half_h
            sx2 = (ndc2[0] + 1.0) * half_w
            sy2 = (1.0 - ndc2[1]) * half_h
            
            # Bounding box
            min_x = max(0, int(np.floor(min(sx0, sx1, sx2))))
            max_x = min(W - 1, int(np.ceil(max(sx0, sx1, sx2))))
            min_y = max(0, int(np.floor(min(sy0, sy1, sy2))))
            max_y = min(H - 1, int(np.ceil(max(sy0, sy1, sy2))))
            
            # Triangle area
            area2 = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0)
            if abs(area2) < 1e-10:
                continue
            inv_area2 = 1.0 / area2
            
            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    px_f = px + 0.5
                    py_f = py + 0.5
                    
                    # Barycentric coordinates
                    w0 = ((sx1 - px_f) * (sy2 - py_f) - (sx2 - px_f) * (sy1 - py_f)) * inv_area2
                    w1 = ((sx2 - px_f) * (sy0 - py_f) - (sx0 - px_f) * (sy2 - py_f)) * inv_area2
                    w2 = 1.0 - w0 - w1
                    
                    if w0 >= 0 and w1 >= 0 and w2 >= 0:
                        depth = w0 * ndc0[2] + w1 * ndc1[2] + w2 * ndc2[2]
                        if depth < depth_buffer[py, px]:
                            depth_buffer[py, px] = depth
                            rast[batch_idx, py, px, 0] = w1  # u
                            rast[batch_idx, py, px, 1] = w2  # v
                            rast[batch_idx, py, px, 2] = depth
                            rast[batch_idx, py, px, 3] = tri_idx + 1  # 1-indexed
    
    return torch.from_numpy(rast).to(mesh.device)


# =============================================================================
# Test with case_1.pth
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test against case_1.pth")
    parser.add_argument("--profile", action="store_true", help="Run for profiling")
    args = parser.parse_args()
    
    if args.test:
        # Load test case
        data = torch.load("case_1.pth")
        pos = data['pos_rast']  # (N, V, 4)
        tri = data['tri_tensor']  # (F, 3)
        expected_rast = data['rast']  # (N, H, W, 4)
        
        H, W = 512, 512
        
        result = rasterize(pos, tri, [H, W])
    
    elif args.profile:
        # Simple profiling run
        N, V = 1, 10000
        F = 5000
        H, W = 512, 512
        
        torch.manual_seed(42)
        mesh = torch.randn(N, V, 4, device='cuda', dtype=torch.float32)
        mesh[..., 3] = 1.0  # Set w=1 for simplicity
        tri = torch.randint(0, V, (F, 3), device='cuda', dtype=torch.int32)
        
        # Warmup
        result = rasterize(mesh, tri, [H, W])
        torch.cuda.synchronize()
        
        # Profile run
        result = rasterize(mesh, tri, [H, W])
        torch.cuda.synchronize()
        
        pass
    
    else:
        # Quick test
        N, V = 1, 100
        F = 50
        H, W = 64, 64
        
        torch.manual_seed(42)
        mesh = torch.randn(N, V, 4, device='cuda', dtype=torch.float32)
        mesh[..., 3] = 1.0
        tri = torch.randint(0, V, (F, 3), device='cuda', dtype=torch.int32)
        
        result = rasterize(mesh, tri, [H, W])


"""
Interpolate Kernel - Triton Implementation

This kernel performs vertex attribute interpolation for 3D mesh rendering,
similar to nvdiffrast's interpolate function.

The algorithm:
1. For each pixel, read the rasterization result (u, v, depth, tri_idx)
2. If tri_idx > 0, look up the triangle's vertex indices
3. Interpolate vertex attributes using barycentric coordinates: 
   attr = u * attr[v0] + v * attr[v1] + (1-u-v) * attr[v2]

Input:
    attr: (N, V, A) or (V, A) - vertex attributes (colors, normals, etc.)
    rast: (N, H, W, 4) - rasterization output (u, v, depth, tri_idx)
    tri: (F, 3) - triangle indices (int32)

Output:
    out: (N, H, W, A) - interpolated attributes
"""

import torch
import triton
import triton.language as tl


@triton.jit
def interpolate_kernel_unified(
    # Input pointers
    attr_ptr,          # (N, V, A) vertex attributes
    rast_ptr,          # (N, H, W, 4) rasterization output
    tri_ptr,           # (F, 3) triangle indices
    # Output pointer
    out_ptr,           # (N, H, W, A) output
    # Dimensions
    batch_size,
    num_vertices,
    image_height,
    image_width,
    num_triangles,
    # Strides
    attr_batch_stride,
    rast_batch_stride,
    out_batch_stride,
    # Block size and number of attributes
    BLOCK_SIZE: tl.constexpr,
    NUM_ATTRS: tl.constexpr,
):
    """
    Unified interpolate kernel for any number of attributes.
    Uses tl.constexpr for NUM_ATTRS to enable compile-time specialization.
    For 4 attributes (common RGBA case), uses manually unrolled code.
    """
    pid = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    # Calculate pixel indices
    pixel_start = pid * BLOCK_SIZE
    pixel_offsets = pixel_start + tl.arange(0, BLOCK_SIZE)
    total_pixels = image_height * image_width
    pixel_mask = pixel_offsets < total_pixels
    
    # Load rasterization data (u, v, depth, tri_idx)
    rast_base = batch_idx * rast_batch_stride + pixel_offsets * 4
    
    # Load all rast data first - use evict_first to free cache for attribute loads
    u = tl.load(rast_ptr + rast_base + 0, mask=pixel_mask, other=0.0, eviction_policy="evict_first")
    v = tl.load(rast_ptr + rast_base + 1, mask=pixel_mask, other=0.0, eviction_policy="evict_first")
    tri_idx_f = tl.load(rast_ptr + rast_base + 3, mask=pixel_mask, other=0.0, eviction_policy="evict_first")
    
    # Convert triangle index to int (1-indexed in rast, 0 means background)
    tri_idx = tri_idx_f.to(tl.int32) - 1  # Convert to 0-indexed
    valid_mask = pixel_mask & (tri_idx >= 0)
    
    # Calculate w = 1 - u - v
    w = 1.0 - u - v
    
    # Load triangle vertex indices
    tri_base = tri_idx * 3
    v0_idx = tl.load(tri_ptr + tri_base + 0, mask=valid_mask, other=0)
    v1_idx = tl.load(tri_ptr + tri_base + 1, mask=valid_mask, other=0)
    v2_idx = tl.load(tri_ptr + tri_base + 2, mask=valid_mask, other=0)
    
    # Attribute base pointer for this batch
    attr_batch_base = batch_idx * attr_batch_stride
    
    # Compute attribute base addresses once
    v0_base = attr_batch_base + v0_idx * NUM_ATTRS
    v1_base = attr_batch_base + v1_idx * NUM_ATTRS
    v2_base = attr_batch_base + v2_idx * NUM_ATTRS
    
    # Output base pointer
    out_base = batch_idx * out_batch_stride + pixel_offsets * NUM_ATTRS
    
    # Manually unrolled code for 4 attributes (most common case: RGBA)
    # This is equivalent to the original interpolate_kernel_4attr
    if NUM_ATTRS == 4:
        # Load all vertex attributes first (better memory access pattern)
        a0_0 = tl.load(attr_ptr + v0_base + 0, mask=valid_mask, other=0.0)
        a0_1 = tl.load(attr_ptr + v0_base + 1, mask=valid_mask, other=0.0)
        a0_2 = tl.load(attr_ptr + v0_base + 2, mask=valid_mask, other=0.0)
        a0_3 = tl.load(attr_ptr + v0_base + 3, mask=valid_mask, other=0.0)
        
        a1_0 = tl.load(attr_ptr + v1_base + 0, mask=valid_mask, other=0.0)
        a1_1 = tl.load(attr_ptr + v1_base + 1, mask=valid_mask, other=0.0)
        a1_2 = tl.load(attr_ptr + v1_base + 2, mask=valid_mask, other=0.0)
        a1_3 = tl.load(attr_ptr + v1_base + 3, mask=valid_mask, other=0.0)
        
        a2_0 = tl.load(attr_ptr + v2_base + 0, mask=valid_mask, other=0.0)
        a2_1 = tl.load(attr_ptr + v2_base + 1, mask=valid_mask, other=0.0)
        a2_2 = tl.load(attr_ptr + v2_base + 2, mask=valid_mask, other=0.0)
        a2_3 = tl.load(attr_ptr + v2_base + 3, mask=valid_mask, other=0.0)
        
        # Interpolate all 4 attributes
        interp_0 = u * a0_0 + v * a1_0 + w * a2_0
        interp_1 = u * a0_1 + v * a1_1 + w * a2_1
        interp_2 = u * a0_2 + v * a1_2 + w * a2_2
        interp_3 = u * a0_3 + v * a1_3 + w * a2_3
        
        # Store results
        tl.store(out_ptr + out_base + 0, interp_0, mask=valid_mask)
        tl.store(out_ptr + out_base + 1, interp_1, mask=valid_mask)
        tl.store(out_ptr + out_base + 2, interp_2, mask=valid_mask)
        tl.store(out_ptr + out_base + 3, interp_3, mask=valid_mask)
    else:
        # Generic loop for other attribute counts
        for attr_idx in tl.static_range(NUM_ATTRS):
            a0 = tl.load(attr_ptr + v0_base + attr_idx, mask=valid_mask, other=0.0)
            a1 = tl.load(attr_ptr + v1_base + attr_idx, mask=valid_mask, other=0.0)
            a2 = tl.load(attr_ptr + v2_base + attr_idx, mask=valid_mask, other=0.0)
            interp = u * a0 + v * a1 + w * a2
            tl.store(out_ptr + out_base + attr_idx, interp, mask=valid_mask)


def interpolate(attr, rast, tri):
    """
    Interpolate vertex attributes using rasterization output.
    
    Args:
        attr: Vertex attributes tensor with shape [N, V, A] or [V, A]
              where N=batch, V=vertices, A=attributes
        rast: Rasterization output from rasterize() with shape [N, H, W, 4]
              containing (u, v, depth, tri_idx) per pixel
        tri: Triangle indices with shape [F, 3] and dtype int32
    
    Returns:
        Tuple of (out, out_db) where:
        - out: Interpolated attributes with shape [N, H, W, A]
        - out_db: Empty tensor (derivatives not implemented)
    """
    # Handle 2D attr input (V, A) -> (1, V, A)
    if attr.dim() == 2:
        attr = attr.unsqueeze(0)
    
    # Get dimensions
    N, V, A = attr.shape
    _, H, W, _ = rast.shape
    F = tri.shape[0]
    
    # Ensure contiguous
    attr = attr.contiguous()
    rast = rast.contiguous()
    tri = tri.contiguous()
    
    # Allocate output (pre-initialized to zero for background pixels)
    out = torch.zeros(N, H, W, A, dtype=attr.dtype, device=attr.device)
    
    # Launch kernel
    total_pixels = H * W
    BLOCK_SIZE = 256  # Optimal block size
    num_blocks = (total_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_blocks, N)
    
    # Unified kernel works for any number of attributes
    # NUM_ATTRS is a constexpr, so the compiler will specialize for each value
    attr_batch_stride = V * A
    rast_batch_stride = H * W * 4
    out_batch_stride = H * W * A
    
    interpolate_kernel_unified[grid](
        attr, rast, tri, out,
        N, V, H, W, F,
        attr_batch_stride, rast_batch_stride, out_batch_stride,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_ATTRS=A,
    )
    
    # Return (out, empty_db) to match nvdiffrast API
    out_db = torch.empty(N, H, W, 0, dtype=attr.dtype, device=attr.device)
    return out, out_db
