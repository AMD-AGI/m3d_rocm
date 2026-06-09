import os
import sys
import torch
import numpy as np
import trimesh
from numbers import Number
from typing import *
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import trimesh
import pyrender
import math
import json
import cv2
import utils3d
import nvdiffrast.torch as dr
from scipy.ndimage.morphology import distance_transform_edt
# do some random sample upon the mesh surface;
# we can as well as remove some non-valid boundary points from the depth map;
# what are those guys up to.
# apply some interesting lora.
def get_diffrast_camera_parameter_from_cv(K, H, W, near, far, device):
    K_ = torch.zeros((4,4), dtype=torch.float32, device=device)
    K_[0,0] = K[0,0] * 2. / W
    K_[1,1] = K[1,1] * 2. / H
    K_[2,2] = (far + near) / (far - near) 
    K_[2,3] = - 2. * (near * far) / (far - near) 
    K_[3,2] = 1.
    return K_
def get_mesh_render(mesh, Ks, Rts, H, W, device):
    if isinstance(Ks, np.ndarray):
        Ks = torch.from_numpy(Ks).float().to(device)
        Rts = torch.from_numpy(Rts).float().to(device)
    glctx = dr.RasterizeCudaContext(device=device)
    pos_tensor = torch.from_numpy(np.array(mesh.vertices)).float().to(device)
    col_tensor = torch.from_numpy(np.array(mesh.visual.vertex_colors)).float().to(device)
    tri_tensor = torch.from_numpy(np.array(mesh.faces)).int().to(device)
    if col_tensor.max() > 5:
        col_tensor /= 255.
    pos_qc = torch.ones((pos_tensor.shape[0],4),dtype=torch.float32, device=device)
    near = 0.
    far = 1e3
    N = Rts.shape[0]
    all_images = []

    for i in range(N):
        pos_tensor_cam = pos_tensor @ Rts[i,:3,:3].T + Rts[i:i+1,:3,3]
        K_ = get_diffrast_camera_parameter_from_cv(Ks[i], H, W, near, far, device)
        pos_qc[:,:3] = pos_tensor_cam
        pos_rast = (pos_qc @ K_.T)[None]
        rast, _ = dr.rasterize(glctx, pos_rast, tri_tensor, resolution=[H, W])
        out, _ = dr.interpolate(col_tensor[None], rast, tri_tensor)

        img = out.cpu().numpy()[0, :, :, :] # Flip vertically.
        #img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
        all_images.append(img)
    return np.stack(all_images, axis=0)

def spherical_uv_to_directions_torch(uv):
    # device = uv.device
    theta, phi = (1 - uv[..., 0]) * (2 * torch.pi), uv[..., 1] * torch.pi
    directions = torch.stack([torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)], axis=-1)
    return directions

def image_uv_torch(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:

    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, device=device, dtype=dtype)
    v = torch.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv
def get_pano_pcs_torch(pano_depth):
    device = pano_depth.device
    H, W = pano_depth.shape
    uv = image_uv_torch(width=W, height=H, device=device)
    directions = spherical_uv_to_directions_torch(uv)

    vertices_cam = pano_depth[:,:,None] * directions
    return vertices_cam.reshape((-1,3))
def get_world_pcs_pano_torch(c2w, depth):
    device = depth.device
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(depth.dtype).to(device)
    # 提取深度在相机系下的点云；
    rot_matrix = torch.tensor([
        [0,1,0],[0,0,-1],[-1,0,0.]
    ], dtype=depth.dtype,device=device)

    cam_pcs = get_pano_pcs_torch(depth) @ rot_matrix.T

    # print(cam_pcs.dtype, c2w.dtype)
    world_pcs = cam_pcs @ c2w[:3,:3].T + c2w[:3,3][None]
    return world_pcs
def uv_to_pixel_torch(
    uv,
    width,
    height,
):
    """
    Args:
        pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    if not isinstance(width, int):
        pixel = uv * torch.stack([width, height], dim=-1).to(uv.dtype).to(uv.device) - 0.5
    else:
        Ns = uv.ndim
        tmp = torch.tensor([width, height]).to(uv.dtype).to(uv.device)
        for i in range(Ns - 1):
            tmp = tmp.unsqueeze(0)
        pixel = uv * tmp - 0.5
    return pixel

def directions_to_spherical_uv_torch(directions):
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)
    u = 1 - torch.arctan2(directions[..., 1], directions[..., 0]) / (2 * torch.pi) % 1.0
    v = torch.arccos(directions[..., 2]) / torch.pi
    return torch.stack([u, v], dim=-1)

def generate_masked_video(rgb, depth, end_angle, frame_number, movement_ratio):
    if isinstance(depth, np.ndarray):
        device = "cuda:0"
        depth = torch.from_numpy(depth).to(torch.float32).to(device)
    else:
        device = depth.device
    rot_matrix = torch.tensor([
        [0,1,0],[0,0,-1],[-1,0,0.]
    ], dtype=depth.dtype,device=device)
    depth_edge_mask = torch.from_numpy(~depth_edge(depth.cpu().numpy(), rtol=0.03)).reshape(-1).to(device)
    c2w_wrd = torch.eye(4).to(torch.float32).to(device)
    wrd_pcs = get_world_pcs_pano_torch(c2w_wrd, depth)
    wrd_rgb = rgb.reshape((-1,3))
    H, W = depth.shape[:2]
    mid_pivot = (H//2, int((end_angle/360.) * W + W//2) % W)
    depth_value = depth[mid_pivot[0], mid_pivot[1]]
    all_canvas = torch.zeros((frame_number, H, W, 3), dtype=torch.float32, device=device)
    all_canvas_mask = torch.zeros((frame_number, H, W), dtype=torch.bool, device=device)
    for i in range(frame_number):
        R = torch.tensor([
            [math.cos(math.radians(end_angle)), 0, -math.sin(math.radians(end_angle))],
            [0, 1., 0.],
            [math.sin(math.radians(end_angle)), 0, math.cos(math.radians(end_angle))],
        ]).to(torch.float32).to(device)
        t = torch.tensor([math.sin(math.radians(end_angle)), 0, math.cos(math.radians(end_angle))]).float().to(device) * (i/(frame_number-1)) * movement_ratio * depth_value
        c2w = torch.eye(4).float().to(device)
        c2w[:3,:3] = R.T
        c2w[:3,3] = t
        w2c = torch.linalg.inv(c2w)

        cur_pcs_cam = wrd_pcs @ w2c[:3,:3].T + w2c[:3,3][None]
        pcs_cur_c2w_sph = cur_pcs_cam @ rot_matrix

        pcs_cur_c2w_norm = torch.linalg.norm(pcs_cur_c2w_sph, dim=-1)
        pcs_cur_c2w_direction = pcs_cur_c2w_sph / (pcs_cur_c2w_norm[...,None] + 1e-8)

        pcs_cur_c2w_uv = directions_to_spherical_uv_torch(pcs_cur_c2w_direction)
        pcs_cur_c2w_pix = uv_to_pixel_torch(pcs_cur_c2w_uv, width=W, height=H).to(torch.float32)
        pcs_cur_c2w_pix_long = (pcs_cur_c2w_pix + 0.5).to(torch.int32)

        mask_valid = (pcs_cur_c2w_pix_long[:,1] < H - 1) * (pcs_cur_c2w_pix_long[:,0] < W - 1) * (pcs_cur_c2w_pix_long[:,1] > 0) * (pcs_cur_c2w_pix_long[:,0] > 0) * (pcs_cur_c2w_norm > 0) 
        if i >= 1:
            mask_valid = mask_valid * depth_edge_mask
        pcs_norm_valid = pcs_cur_c2w_norm[mask_valid]
        pcs_pix_valid = pcs_cur_c2w_pix_long[mask_valid]
        pcs_rgb_valid = wrd_rgb[mask_valid]

        sort_index = torch.argsort(pcs_norm_valid,descending=True)
        pcs_pix_valid = pcs_pix_valid[sort_index]
        all_canvas[i, pcs_pix_valid[:,1], pcs_pix_valid[:,0]] = pcs_rgb_valid[sort_index]
        all_canvas_mask[i, pcs_pix_valid[:,1], pcs_pix_valid[:,0]] = True
    # print()
    return all_canvas, all_canvas_mask


def write_pointcloud(pc,filename, rgb=None):
    with open(filename,"w") as file:
        N = pc.shape[0]
        for i in range(N):
            if rgb is None:
                file.write(f"v {pc[i,0]} {pc[i,1]} {pc[i,2]}\n")
            else:
                file.write(f"v {pc[i,0]} {pc[i,1]} {pc[i,2]} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")
# from trellis to anything.
# super earth is the best.
def merge_videos(vid_path1, vid_path2, out_path):
    vid1 = cv2.VideoCapture(vid_path1)
    vid2 = cv2.VideoCapture(vid_path2)
    if vid1.isOpened() and vid2.isOpened():
        fps = vid1.get(cv2.CAP_PROP_FPS)
        width = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        while vid1.isOpened():
            ret1, frame1 = vid1.read()
            if not ret1:
                
                break
            out.write(frame1)
        while vid2.isOpened():
            ret2, frame2 = vid2.read()
            if not ret2:
                
                break
            out.write(frame2)
        out.release()
def write_video(frames, out_path, fps = 10):
    width, height = frames[0].shape[1], frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
def get_video_frames(vid_path):
    vid = cv2.VideoCapture(vid_path)
    frames = []
    if vid.isOpened():
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                
                break
            frames.append(frame)

    return frames

# it shall be so.
# somehow make some little tidy-up.
'''
def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(
                            f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}"
                        )
            else:
                transformer_ = unwrap_model(
                    self.accelerator, self.components.transformer
                ).__class__.from_pretrained(self.args.model_path, subfolder="transformer")
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)
# interesting.

'''

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as file:
        d = json.load(file)
    K = np.zeros((3,3), dtype=np.float32)
    K[0,0] = d["focal_length_in_pixel"]
    K[1,1] = d["focal_length_in_pixel"]
    K[0,2] = d["image_width"]/2.
    K[1,2] = d["image_height"]/2.
    K[2,2] = 1.
    return K

def save_intrinsic(intrinsic_path, K):
    dic = {
        "focal_length_in_pixel": float((K[0,0]+K[1,1])/2.),
        "image_width": int(K[0,2]) * 2,
        "image_height": int(K[1,2]) * 2
    }
    with open(intrinsic_path, 'w') as file:
        file.write(json.dumps(dic,indent=4))

def Depth_View_Transfer(depth0, K0, Rt0, K1, Rt1, threshold = 1.):
    pts_wrd = raw_depth_to_pointcloud(depth0[None,:,:], K0[None], Rt0[None])[0]
    # get topology
    H, W = pts_wrd.shape[:2]
    pts_topology = np.zeros((H - 1, W - 1, 2, 3), dtype=np.int32)
    index_ = np.arange(H*W).astype(np.int32).reshape(H,W)
    pts_topology[:,:,0,0] = index_[:-1,:-1]
    pts_topology[:,:,0,2] = index_[:-1,1:]
    pts_topology[:,:,0,1] = index_[1:,:-1]

    pts_topology[:,:,1,0] = index_[:-1,1:]
    pts_topology[:,:,1,1] = index_[1:,:-1]
    pts_topology[:,:,1,2] = index_[1:,1:]

    d00 = depth0[:-1,:-1,0]
    d01 = depth0[:-1,1:,0]
    d10 = depth0[1:,:-1,0]
    d11 = depth0[1:,1:,0]
    
    mask0 = (np.abs(d00 - d10)<threshold) * (np.abs(d00 - d01)<threshold) * (np.abs(d01 - d10)<threshold)
    mask1 = (np.abs(d11 - d10)<threshold) * (np.abs(d11 - d01)<threshold) * (np.abs(d01 - d10)<threshold)
    
    pts_wrd = pts_wrd.reshape(-1,3)
    topo0 = pts_topology[:,:,0][mask0]
    topo1 = pts_topology[:,:,1][mask1]
    topo = np.concatenate([topo0, topo1], axis=0)
    # print("sdada, ",topo.shape, pts_topology.shape, mask0.shape, topo0.shape)
    mesh_trimesh = trimesh.Trimesh(vertices=pts_wrd, faces=topo)
    # mesh.export("../debug/sov.obj")
    print(f"mesh construction complete; mesh info: {mesh_trimesh.vertices.shape} {mesh_trimesh.faces.shape}")
    
    
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
    scene.add(mesh)
    # convert K1 to pyrender camera;

    if K1.ndim == 2:
        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
        scene.add(mesh)

        aspect_ratio = float(W)/H
        yfov = 2.*math.atan(float(H)/(K1[1,1]*2.))
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)

        camera_pose = np.eye(4)
        camera_pose[:3,:3] = Rt1[:3,:3].transpose(1,0)
        camera_pose[:3, 3] = -camera_pose[:3,:3] @ Rt1[:3,3]
        camera_pose[:,1] *= -1.
        camera_pose[:,2] *= -1.

        scene.add(camera, pose=camera_pose)
        r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        _, depth = r.render(scene)
    else:
        nc = None
        r = None
        depth_list = []
        
        N = K1.shape[0]
        for i in range(N):
            
            aspect_ratio = float(W)/H
            yfov = 2.*math.atan(float(H)/(K1[i,1,1]*2.))
            camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)

            camera_pose = np.eye(4)
            camera_pose[:3,:3] = Rt1[i,:3,:3].transpose(1,0)
            camera_pose[:3, 3] = -camera_pose[:3,:3] @ Rt1[i,:3,3]
            camera_pose[:,1] *= -1.
            camera_pose[:,2] *= -1.
            if nc is None:
                nc = scene.add(camera, pose=camera_pose)
            
            else:
                print("cam set")
                scene.set_pose(nc, pose=camera_pose)
            if r is None:
                r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
            _, depth = r.render(scene)
            # scene.remove_node(nc)
            depth_list.append(depth)
        depth = np.stack(depth_list, axis=0)
    print(f"depth render complete, depth range: {depth.max()} {depth.min()} {depth.shape}")
    return mesh_trimesh, depth


def depth_to_inverse(depth_map, return_max_min=False):
    depth_map_crop = depth_map.copy()
    depth_map_crop[depth_map_crop<1e-4] = 1e-4
    depth_map_crop = 1./depth_map_crop
    max_, min_ = depth_map_crop.max(), depth_map_crop.min()
    depth_map_crop = (depth_map_crop - min_)/(max_-min_)*255
    depth_map_crop = depth_map_crop.astype(np.uint8)
    if return_max_min:
        return depth_map_crop, max_, min_
    return depth_map_crop

def raw_depth_to_pointcloud(depthlist, K, Rt):
    B = len(depthlist)
    H, W = depthlist[0].shape[:2]
    xy_canvas = np.ones((H, W, 3), dtype=np.float32)
    xy_canvas[:,:,0] = np.linspace(0,W,num=W)[None,:]
    xy_canvas[:,:,1] = np.linspace(0,H,num=H)[:,None]
    world_coords_list = []
    for i in range(B):
        cur_depth_map = depthlist[i].astype(np.float32)[:,:,:1]
        
        cur_K = K[i]
        cur_Rt = Rt[i]
        cur_c2w = np.zeros((3,4), dtype=np.float32)
        cur_c2w[:,:3] = cur_Rt[:3,:3].transpose(1,0)
        cur_c2w[:,3:] = -cur_Rt[:3,:3].transpose(1,0)@cur_Rt[:3,3:]
        depth_vec = xy_canvas * cur_depth_map[:,:,:1]

        cur_K_inv = np.linalg.inv(cur_K).transpose(1,0)[None,None]

        world_coords = (depth_vec[...,None,:]@cur_K_inv)@cur_c2w[:,:3].transpose(1,0)[None,None] + cur_c2w[:,3][None,None,None]
        world_coords = world_coords[:,:,0,:]
        world_coords_list.append(world_coords)
        # ok then;
        # what else?

    return world_coords_list

def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Return x view of the input array with x sliding window of the given kernel size and stride.
    The sliding window is performed over the given axis, and the window dimension is append to the end of the output array's shape.

    Args:
        x (np.ndarray): input array with shape (..., axis_size, ...)
        kernel_size (int): size of the sliding window
        stride (int): stride of the sliding window
        axis (int): axis to perform sliding window over
    
    Returns:
        a_sliding (np.ndarray): view of the input array with shape (..., n_windows, ..., kernel_size), where n_windows = (axis_size - kernel_size + 1) // stride
    """
    assert x.shape[axis] >= window_size, f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    axis = axis % x.ndim
    shape = (*x.shape[:axis], (x.shape[axis] - window_size + 1) // stride, *x.shape[axis + 1:], window_size)
    strides = (*x.strides[:axis], stride * x.strides[axis], *x.strides[axis + 1:], x.strides[axis])
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding


def sliding_window_nd(x: np.ndarray, window_size: Tuple[int,...], stride: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def sliding_window_2d(x: np.ndarray, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)


def max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        padding_arr = np.full((*x.shape[:axis], padding, *x.shape[axis + 1:]), fill_value=fill_value, dtype=x.dtype)
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool


def max_pool_nd(x: np.ndarray, kernel_size: Tuple[int,...], stride: Tuple[int,...], padding: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x


def max_pool_2d(x: np.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)

def depth_edge(depth: np.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.
    
    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff = (max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (max_pool_2d(np.where(mask, depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(np.where(mask, -depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def points_to_normals(point: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Calculate normal map from point map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

    Args:
        point (np.ndarray): shape (height, width, 3), point map
    Returns:
        normal (np.ndarray): shape (height, width, 3), normal map. 
    """
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack([
        np.cross(up, left, axis=-1),
        np.cross(left, down, axis=-1),
        np.cross(down, right, axis=-1),
        np.cross(right, up, axis=-1),
    ])
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    valid = np.stack([
        mask[:-2, 1:-1] & mask[1:-1, :-2],
        mask[1:-1, :-2] & mask[2:, 1:-1],
        mask[2:, 1:-1] & mask[1:-1, 2:],
        mask[1:-1, 2:] & mask[:-2, 1:-1],
    ]) & mask[None, 1:-1, 1:-1]
    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    
    if has_mask:
        normal_mask =  valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    else:
        return normal
# 这也没啥屁用啊。
def generate_mesh_from_depth(depth0, K0, Rt0, threshold = 0.03):
    pts_wrd = raw_depth_to_pointcloud(depth0[None], K0[None], Rt0[None])[0]
    depth_edge_mask = ~depth_edge(depth0[:,:,0], rtol = threshold, kernel_size = 3, mask = None)
    # get topology
    H, W = pts_wrd.shape[:2]
    pts_topology = np.zeros((H - 1, W - 1, 2, 3), dtype=np.int32)
    index_ = np.arange(H*W).astype(np.int32).reshape(H,W)
    pts_topology[:,:,0,0] = index_[:-1,:-1]
    pts_topology[:,:,0,2] = index_[:-1,1:]
    pts_topology[:,:,0,1] = index_[1:,:-1]

    pts_topology[:,:,1,0] = index_[:-1,1:]
    pts_topology[:,:,1,1] = index_[1:,:-1]
    pts_topology[:,:,1,2] = index_[1:,1:]

    d00 = depth_edge_mask[:-1,:-1]
    d01 = depth_edge_mask[:-1,1:]
    d10 = depth_edge_mask[1:,:-1]
    d11 = depth_edge_mask[1:,1:]
    
    mask0 = d00 * d10 * d01# ((d00 - d10)) * (np.abs(d00 - d01)<threshold) * (np.abs(d01 - d10)<threshold)
    mask1 = d11 * d01 * d10# (np.abs(d11 - d10)<threshold) * (np.abs(d11 - d01)<threshold) * (np.abs(d01 - d10)<threshold)
    
    pts_wrd = pts_wrd.reshape(-1,3)
    topo0 = pts_topology[:,:,0][mask0]
    topo1 = pts_topology[:,:,1][mask1]
    topo = np.concatenate([topo0, topo1], axis=0)
    # print("sdada, ",topo.shape, pts_topology.shape, mask0.shape, topo0.shape)
    mesh_trimesh = trimesh.Trimesh(vertices=pts_wrd, faces=topo)
    # mesh.export("../debug/sov.obj")
    
    return mesh_trimesh

def generate_mesh_from_depth_with_mask(depth0, K0, Rt0, mask0, threshold = 0.03):
    pts_wrd = raw_depth_to_pointcloud(depth0[None], K0[None], Rt0[None])[0]
    depth_edge_mask = ~depth_edge(depth0[:,:,0], rtol = threshold, kernel_size = 3, mask = None)
    # get topology
    H, W = pts_wrd.shape[:2]
    pts_topology = np.zeros((H - 1, W - 1, 2, 3), dtype=np.int32)
    index_ = np.arange(H*W).astype(np.int32).reshape(H,W)
    pts_topology[:,:,0,0] = index_[:-1,:-1]
    pts_topology[:,:,0,2] = index_[:-1,1:]
    pts_topology[:,:,0,1] = index_[1:,:-1]

    pts_topology[:,:,1,0] = index_[:-1,1:]
    pts_topology[:,:,1,1] = index_[1:,:-1]
    pts_topology[:,:,1,2] = index_[1:,1:]

    d00 = depth_edge_mask[:-1,:-1]
    d01 = depth_edge_mask[:-1,1:]
    d10 = depth_edge_mask[1:,:-1]
    d11 = depth_edge_mask[1:,1:]

    m00 = mask0[:-1,:-1]
    m01 = mask0[:-1,1:]
    m10 = mask0[1:,:-1]
    m11 = mask0[1:,1:]
    
    mask0 = d00 * d10 * d01 * (m00 + m10 + m01)# ((d00 - d10)) * (np.abs(d00 - d01)<threshold) * (np.abs(d01 - d10)<threshold)
    mask1 = d11 * d01 * d10 * (m11 + m10 + m01)# (np.abs(d11 - d10)<threshold) * (np.abs(d11 - d01)<threshold) * (np.abs(d01 - d10)<threshold)
    
    pts_wrd = pts_wrd.reshape(-1,3)
    topo0 = pts_topology[:,:,0][mask0]
    topo1 = pts_topology[:,:,1][mask1]
    topo = np.concatenate([topo0, topo1], axis=0)
    # print("sdada, ",topo.shape, pts_topology.shape, mask0.shape, topo0.shape)
    mesh_trimesh = trimesh.Trimesh(vertices=pts_wrd, faces=topo)
    # mesh.export("../debug/sov.obj")
    
    return mesh_trimesh

def generate_colored_pointcloud(rgb, depth, K, Rt, threshold=0.03):
    pts_wrd = raw_depth_to_pointcloud(depth[None], K[None], Rt[None])[0]
    depth_edge_mask = ~depth_edge(depth[:,:,0], rtol = threshold, kernel_size = 3, mask = None)
    pts_wrd_valid = pts_wrd[depth_edge_mask]
    rgb_valid = rgb[depth_edge_mask]
    return pts_wrd_valid, rgb_valid


def read_blender_camera_as_opencv(filepath, resolution=512, focal_length=341.58):
    Rt = np.load(filepath)["arr_0"].reshape(-1,4,4)
    Rt[:,:3,:3] = Rt[:,:3,:3].transpose(0,2,1)

    blender_to_cv_matrix = np.array(
        [
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1]
        ],dtype=np.float32
    )
    Rt_cv = Rt.copy()
    Rt_cv[:, :, :3] = blender_to_cv_matrix[None]@Rt[:,:,:3] 
    Rt_cv[:, :3, 3:] = -Rt_cv[:,:3,:3]@Rt_cv[:,:3,3:]
    # Rt_cv = blender_to_cv_matrix[None]@Rt
    if isinstance(resolution, int):
        K = np.array([
            [focal_length,0,resolution/2.],
            [0,focal_length,resolution/2.],
            [0,0,1]
        ],dtype=np.float32)[None].repeat(Rt_cv.shape[0], 0)
    else:
        # given as (width, height)
        K = np.array([
            [focal_length,0,resolution[0]/2.],
            [0,focal_length,resolution[1]/2.],
            [0,0,1]
        ],dtype=np.float32)[None].repeat(Rt_cv.shape[0], 0)
    return K,Rt_cv.astype(np.float32)
# 那些比较摆的互联网厂子。
# 有没有这一说
# 瞧瞧吧。那些高精尖的，做新事情的，都不要再考虑了。
# 那个需要一个卷的环境。
# 我半分钟都不想卷。
# 你总得生活的。这种逼玩意搞得人都不想活。
# 而且说实话。这个加钱的事情多半是饼
# 我很确定。

def mesh_to_depth(mesh, K, Rt, H, W):
    scene = pyrender.Scene()
    mesh_ = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_)
    # convert K1 to pyrender camera;

    aspect_ratio = float(W)/H
    yfov = 2.*math.atan(float(H)/(K[1,1]*2.))
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)

    camera_pose = np.eye(4)
    camera_pose[:3,:3] = Rt[:3,:3].transpose(1,0)
    camera_pose[:3, 3] = -camera_pose[:3,:3] @ Rt[:3,3]
    camera_pose[:,1] *= -1.
    camera_pose[:,2] *= -1.

    scene.add(camera, pose=camera_pose)
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    _, depth = r.render(scene)
    return depth


def depth_and_rgb_to_pointcloud(depths, rgbs, Ks, Rts, masks=None):
    # some kind of loose end.
    N = Ks.shape[0]
    all_pts = None
    all_rgbs = None
    partial_mesh = None

    for i in range(N):
        # pts_wrd = raw_depth_to_pointcloud(depths[i:i+1], Ks[i:i+1], Rts[i:i+1])[0]
        # generate_mesh_from_depth(depths[i], Ks[i], Rts[i])
        print(f"{i}...")
        cur_pts = raw_depth_to_pointcloud(depths[i:i+1], Ks[i:i+1], Rts[i:i+1])[0]
        cur_rgbs = rgbs[i]
        if all_pts is None:
            if masks is not None:
                all_pts = cur_pts[masks[i]]
                all_rgbs = cur_rgbs[masks[i]]
                if N>1:
                    partial_mesh = generate_mesh_from_depth_with_mask(depths[i], Ks[i], Rts[i], masks[i])
            else:
                all_pts = cur_pts.reshape(-1,3)
                all_rgbs = cur_rgbs.reshape(-1,3) # .astype(np.float32)/255.
                if N>1:
                    partial_mesh = generate_mesh_from_depth(depths[i], Ks[i], Rts[i])
        else:
            H, W = depths[i].shape[:2]
            cur_partial_mesh_depth = mesh_to_depth(partial_mesh, Ks[i], Rts[i], H, W)
            cur_partial_mesh_depth[cur_partial_mesh_depth<1e-3] = 1e3
            if masks is not None:
                cur_mask = masks[i] * (depths[i][...,0] < cur_partial_mesh_depth + 1e-3)
            else:
                cur_mask = depths[i][...,0] < cur_partial_mesh_depth + 1e-3
            cur_pts = cur_pts[cur_mask]
            cur_rgbs = cur_rgbs[cur_mask]
            # (depth0, K0, Rt0, mask0, threshold = 0.03):
            cur_mesh = generate_mesh_from_depth_with_mask(depths[i], Ks[i], Rts[i], cur_mask)
            # all_pts = np.concatenate([all_pts, cur_pts], axis=0)
            partial_mesh = trimesh.util.concatenate([cur_mesh, partial_mesh])
            all_pts = np.concatenate([all_pts, cur_pts], axis=0)
            all_rgbs = np.concatenate([all_rgbs, cur_rgbs], axis=0)
    return all_pts, all_rgbs


def filter_pointcloud(depths, Ks, Rts, pcs, pcs_view_id, device="cuda:0"):
    # 接受点云，然后用深度图filter掉
    # depths: [B, H, W]
    # 被至少一个其他视角观察到，才被视为可靠。
    if isinstance(depths, np.ndarray):
        depths = torch.from_numpy(depths).to(device).float()
        Ks = torch.from_numpy(Ks).to(device).float()
        Rts = torch.from_numpy(Rts).to(device).float()
        pcs = torch.from_numpy(pcs).to(device).float()
    
    H, W = depths.shape[1:]
    all_valid_pts = []
    N = Ks.shape[0]
    fov_radius = 1.2 * math.atan(math.sqrt(H*float(H) + W*W)/(float(Ks[0,0,0]) * 2.))
    fov_cos = math.cos(fov_radius)
    all_mask = torch.zeros((pcs.shape[0], ), dtype=torch.bool, device=device)
    for i in range(N):
        print(f"{i}...")
        cur_c2w = torch.linalg.inv(Rts[i])
        cam_axis = Rts[i][2,:3]
        cam_pos = cur_c2w[:3,3]

        pcs_vec = pcs - cam_pos[None]
        pcs_vec_norm = torch.linalg.norm(pcs_vec, dim=-1)
        pcs_vec = pcs_vec / pcs_vec_norm[:,None]

        pcs_vec_dot = torch.sum(pcs_vec * cam_axis[None], dim=-1)
        
        mask_candidate = (pcs_vec_dot >= fov_cos) * (~(pcs_view_id == i))
        # 应该照这样去画饼
        cur_pcs_valid = pcs[mask_candidate]
        tmp = torch.zeros((cur_pcs_valid.shape[0],), dtype=torch.bool, device=device)
        # project the pcs onto the canvas;
        cur_pcs_cam = cur_pcs_valid @ Rts[i][:3,:3].T + Rts[i][:3,3][None]
        cur_pcs_pix = cur_pcs_cam @ (Ks[i])[:3,:3].T
        cur_pcs_depth = cur_pcs_pix[:,2]
        cur_pcs_uv = cur_pcs_pix[:,:2] / cur_pcs_pix[:,2:]
        
        # 首先这东西肯定是有用的
        # 第二这东西的局限性估计就是，只能拿来搞大结构了。
        # 给几个建模选项，允许在生成mesh的时候自动摊平光滑组，让这玩意尽可能落地。
        # anyway，这个事情本身不关我事。
        mask_valid = (cur_pcs_uv[:,0] >= 1.) * (cur_pcs_uv[:,0] < W-1) * (cur_pcs_uv[:,1] >= 1.) * (cur_pcs_uv[:,1] < H-1) * (cur_pcs_depth > 1e-3)

        valid_uv = cur_pcs_uv[mask_valid].long()
        valid_depth = cur_pcs_depth[mask_valid]

        sample_depth = depths[i][valid_uv[:,1], valid_uv[:,0]]
        mask_depth_valid = sample_depth + 5e-2 > valid_depth 
        unn = mask_valid.clone()
        mask_valid[unn] = mask_depth_valid
        tmp[mask_valid] = True
        all_mask[mask_candidate] = all_mask[mask_candidate] + tmp
    return all_mask
# 不如直接randomly sample 3000w 个点算了。
# ok...
# what else?


# 调用这个的时候，从最后一帧往前添加
# 算了，没那个必要。
# ok,先做mesh depth，然后用它filter点云。
def filter_pointcloud_from_depth(depths, Ks, Rts, rgbs=None, device="cuda:0", sample_size=5e6):
    '''
    depths: [B,H,W]
    Ks: [B,3,3]
    Rts: [B,4,4]
    rgbs: [B,H,W,3]
    '''
    if isinstance(depths, np.ndarray):
        depths = torch.from_numpy(depths).float().to(device)
        Ks = torch.from_numpy(Ks).float().to(device)
        Rts = torch.from_numpy(Rts).float().to(device)
        if rgbs is not None:
            rgbs = torch.from_numpy(rgbs).float().to(device)
    N = Ks.shape[0]
    all_pcs = []
    all_rgbs = []
    # filter_pointcloud(depths, Ks, Rts, pcs, pcs_view_id, device="cuda:0"):
    last_pcs_wrd = None
    for i in range(N):
        H, W = depths.shape[1:]
        canvas = torch.ones((H, W, 3), dtype=torch.float32, device=device)
        # and what else?
        canvas[:,:,0] = torch.linspace(0, W-1, W).to(device)[None,:]
        canvas[:,:,1] = torch.linspace(0, H-1, H).to(device)[:,None]
        canvas = canvas * depths[i][:,:,None]

        c2w = torch.linalg.inv(Rts[i])
        pcs_cam = canvas @ (torch.linalg.inv(Ks[i]).T)[None]
        # print(pcs_cam.shape, c2w.shape)
        
        pcs_wrd = pcs_cam @ c2w[:3,:3].T + c2w[:3,3][None]
        pcs_wrd = pcs_wrd.reshape(-1,3)
        pcs_rgb = rgbs[i].reshape(-1,3)

        # 对后续视角：
        # 将在前一个视角不可见的点加入点集。
        if i>0:
            cur_pcs_cam = pcs_wrd @ Rts[i-1][:3,:3].T + Rts[i-1][:3,3][None]
            cur_pcs_pix = cur_pcs_cam @ (Ks[i-1])[:3,:3].T
            cur_pcs_depth = cur_pcs_pix[:,2]
            cur_pcs_uv = cur_pcs_pix[:,:2] / cur_pcs_pix[:,2:]
            mask_valid = (cur_pcs_uv[:,0] >= 1.) * (cur_pcs_uv[:,0] < W-1) * (cur_pcs_uv[:,1] >= 1.) * (cur_pcs_uv[:,1] < H-1) * (cur_pcs_depth > 1e-3)
            sample_uv = cur_pcs_uv[mask_valid].long()
            p1 = pcs_wrd[~mask_valid]
            p1c = pcs_rgb[~mask_valid]

            if sample_uv.shape[0]>0:
                relative_ratio = 0.1
                sample_depth = depths[i-1][sample_uv[:,1], sample_uv[i,0]]
                # mask_depth = (cur_pcs_depth[mask_valid] > sample_depth+2.)
                mask_depth = (cur_pcs_depth[mask_valid] - sample_depth)/(sample_depth+1e-3) > relative_ratio
                mask_valid[mask_valid.clone()] = mask_depth

                p2 = pcs_wrd[mask_valid]   
                p2c = pcs_rgb[mask_valid]   
                pcs_wrd = torch.cat([p1,p2], dim=0)
                pcs_rgb = torch.cat([p1c,p2c], dim=0)
            else:
                pcs_wrd = p1
                pcs_rgb = p1c


        all_pcs.append(pcs_wrd)
        all_rgbs.append(pcs_rgb)
    all_pcs = torch.cat(all_pcs, dim=0)
    all_rgbs = torch.cat(all_rgbs, dim=0)
    # print(torch.randperm(all_pcs.shape[0]).shape)
    if sample_size == -1:
        randind = torch.arange(all_pcs.shape[0])
    else:    
        randind = torch.randperm(all_pcs.shape[0])[:sample_size]
    

    return all_pcs[randind], all_rgbs[randind]


def filter_pointcloud_from_depth_fb_separate(depths, Ks, Rts, rgbs=None, device="cuda:0", threshold=2.):
    '''
    depths: [B,H,W]
    Ks: [B,3,3]
    Rts: [B,4,4]
    rgbs: [B,H,W,3]
    '''
    if isinstance(depths, np.ndarray):
        depths = torch.from_numpy(depths).float().to(device)
        Ks = torch.from_numpy(Ks).float().to(device)
        Rts = torch.from_numpy(Rts).float().to(device)
        if rgbs is not None:
            rgbs = torch.from_numpy(rgbs).float().to(device)
    N = Ks.shape[0]
    all_pcs_fg = []
    all_rgbs_fg = []

    all_pcs_bg = []
    all_rgbs_bg = []
    # filter_pointcloud(depths, Ks, Rts, pcs, pcs_view_id, device="cuda:0"):
    last_pcs_wrd = None
    for i in range(N):
        bg_mask = depths[i] > threshold
        H, W = depths.shape[1:]
        canvas = torch.ones((H, W, 3), dtype=torch.float32, device=device)
        # and what else?
        canvas[:,:,0] = torch.linspace(0, W-1, W).to(device)[None,:]
        canvas[:,:,1] = torch.linspace(0, H-1, H).to(device)[:,None]
        canvas = canvas * depths[i][:,:,None]

        c2w = torch.linalg.inv(Rts[i])
        pcs_cam = canvas @ (torch.linalg.inv(Ks[i]).T)[None]
        # print(pcs_cam.shape, c2w.shape)
        
        pcs_wrd = pcs_cam @ c2w[:3,:3].T + c2w[:3,3][None]
        pcs_wrd = pcs_wrd.reshape(-1,3)
        pcs_rgb = rgbs[i].reshape(-1,3)
        bg_mask = bg_mask.reshape(-1)
        all_pcs_fg.append(pcs_wrd[~bg_mask])
        all_pcs_bg.append(pcs_wrd[bg_mask])
        all_rgbs_fg.append(pcs_rgb[~bg_mask])
        all_rgbs_bg.append(pcs_rgb[bg_mask])
    all_pcs_bg = torch.cat(all_pcs_bg, dim=0)
    all_rgbs_bg = torch.cat(all_rgbs_bg, dim=0)
    fov_x = math.degrees(math.atan(math.radians(W/(Ks[0,0,0]*2.))))
    bg_samples = min(all_pcs_bg.shape[0], int((360./fov_x)*(H*W)))
    index = torch.randperm(all_pcs_bg.shape[0])[:bg_samples]
    all_pcs_bg = all_pcs_bg[index]
    all_rgbs_bg = all_rgbs_bg[index]
    return all_pcs_fg, all_rgbs_fg, all_pcs_bg, all_rgbs_bg

def render_pointcloud(H, W, pcs, Ks, Rts, rgbs=None, device="cuda:0"):
    if isinstance(pcs, np.ndarray) or isinstance(Ks, np.ndarray) or isinstance(Rts, np.ndarray):
        pcs = torch.from_numpy(pcs).float().to(device)
        Ks = torch.from_numpy(Ks).float().to(device)
        Rts = torch.from_numpy(Rts).float().to(device)
        if rgbs is not None:
            rgbs = torch.from_numpy(rgbs).float().to(device)

    N = Ks.shape[0]
    all_frames = []
    all_masks = []
    for i in range(N):
        print(f"{i}...")
        cur_canvas = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        sdadsa = torch.zeros((H, W, 1), dtype=torch.float32, device=device)
        cur_depth_canvas = 1e5 * torch.ones((H, W), dtype=torch.float32, device=device)
        pcs_cam = pcs @ Rts[i,:3,:3].T + Rts[i,:3,3][None, :]
        pcs_pix = pcs_cam @ (Ks[i])[:3,:3].T
        pcs_depth = pcs_pix[:,2]
        pcs_uv = pcs_pix[:,:2] / pcs_pix[:,2:]        

        cur_mask = (pcs_uv[:,0] >= 1.) * (pcs_uv[:,0] < W-1) * (pcs_uv[:,1] >= 1.) * (pcs_uv[:,1] < H-1) * (pcs_depth > 1e-3)
        cur_pcs_uv = (pcs_uv[cur_mask] + 0.5).long()
        cur_pcs_depth = pcs_depth[cur_mask]
        rgb_valid = rgbs[cur_mask]

        agsort = torch.argsort(cur_pcs_depth,descending=True)
        cur_canvas[cur_pcs_uv[agsort,1], cur_pcs_uv[agsort,0]] = rgb_valid[agsort]
        sdadsa[cur_pcs_uv[agsort,1], cur_pcs_uv[agsort,0]] = 1.

        all_frames.append(cur_canvas * 2. - 1.)
        all_masks.append(sdadsa)
    
    return torch.stack(all_frames, dim=0).permute(0,3,1,2), torch.stack(all_masks, dim=0).permute(0,3,1,2)

def proj_points_on_image(H, W, pcs, K, Rt):

    pcs_cam = pcs @ Rt[:3,:3].T + Rt[:3,3][None, :]
    pcs_pix = pcs_cam @ K[:3,:3].T
    pcs_depth = pcs_pix[:,2]
    pcs_uv = pcs_pix[:,:2] / pcs_pix[:,2:]        

    cur_mask = (pcs_uv[:,0] >= 1.) * (pcs_uv[:,0] < W-1) * (pcs_uv[:,1] >= 1.) * (pcs_uv[:,1] < H-1) * (pcs_depth > 1e-3)
    if isinstance(pcs, np.ndarray):
        cur_pcs_uv = (pcs_uv[cur_mask] + 0.5).astype(np.int32)
    else:
        cur_pcs_uv = (pcs_uv[cur_mask] + 0.5).long()
    cur_pcs_depth = pcs_depth[cur_mask]
    
    return cur_pcs_uv, cur_pcs_depth, cur_mask
# by applying some lora to allow panorama generation.
# no such comments.

def render_pointcloud_with_mesh_filter(H, W, pcs, Ks, Rts, rgbs=None, device="cuda:0", mesh=None):

    # mesh_to_depth(mesh, K, Rt, H, W):
    
    if isinstance(pcs, np.ndarray) or isinstance(Ks, np.ndarray) or isinstance(Rts, np.ndarray):
        pcs = torch.from_numpy(pcs).float().to(device)
        Ks = torch.from_numpy(Ks).float().to(device)
        Rts = torch.from_numpy(Rts).float().to(device)
        if rgbs is not None:
            rgbs = torch.from_numpy(rgbs).float().to(device)
    
    Ks_np = Ks.cpu().numpy()
    Rts_np = Rts.cpu().numpy()
    Nf = Ks.shape[0]
    all_frames_filter_depth = []
    for i in range(Nf):
        meshdepth = mesh_to_depth(mesh, Ks_np[i], Rts_np[i], H, W)
        all_frames_filter_depth.append(torch.from_numpy(meshdepth).to(device))

    N = Ks.shape[0]
    all_frames = []
    all_masks = []
    for i in range(N):
        print(f"{i}...")
        cur_canvas = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        sdadsa = torch.zeros((H, W, 1), dtype=torch.float32, device=device)
        cur_depth_canvas = 1e5 * torch.ones((H, W), dtype=torch.float32, device=device)
        pcs_cam = pcs @ Rts[i,:3,:3].T + Rts[i,:3,3][None, :]
        pcs_pix = pcs_cam @ (Ks[i])[:3,:3].T
        pcs_depth = pcs_pix[:,2]
        pcs_uv = pcs_pix[:,:2] / pcs_pix[:,2:]     

        cur_mask = (pcs_uv[:,0] >= 1.) * (pcs_uv[:,0] < W-1) * (pcs_uv[:,1] >= 1.) * (pcs_uv[:,1] < H-1) * (pcs_depth > 1e-3)
        cur_pcs_uv = (pcs_uv[cur_mask] + 0.5).long()
        cur_pcs_depth = pcs_depth[cur_mask]
        if i>0:
            sampled_depth = all_frames_filter_depth[i][cur_pcs_uv[:,1], cur_pcs_uv[:,0]]
        else:
            sampled_depth = torch.ones_like(cur_pcs_depth) * 1e5
        unn = (cur_pcs_depth - sampled_depth)
        # print(unn.max(), unn.min())
        if i>0:
            mask_depth = ((cur_pcs_depth - sampled_depth) < 0.3) * ((cur_pcs_depth - sampled_depth) > -1)
        else:
            mask_depth = ((cur_pcs_depth - sampled_depth) < 0.3)

        cur_pcs_uv = cur_pcs_uv[mask_depth] 
        cur_pcs_depth = cur_pcs_depth[mask_depth]

        rgb_valid = rgbs[cur_mask]
        rgb_valid = rgb_valid[mask_depth]

        agsort = torch.argsort(cur_pcs_depth,descending=True)
        cur_canvas[cur_pcs_uv[agsort,1], cur_pcs_uv[agsort,0]] = rgb_valid[agsort]
        sdadsa[cur_pcs_uv[agsort,1], cur_pcs_uv[agsort,0]] = 1.

        all_frames.append(cur_canvas * 2. - 1.)
        all_masks.append(sdadsa)
    
    return torch.stack(all_frames, dim=0).permute(0,3,1,2), torch.stack(all_masks, dim=0).permute(0,3,1,2)

def sample_from_aux_frame(canvas, K, Rt, canvas_aux, K_aux, Rt_aux):
    H, W = canvas.shape[:2]
    device = canvas.device
    d_canvas = torch.ones((H, W, 3), dtype=torch.float32, device=device)
    d_canvas[:,:,0] = torch.linspace(0, W-1, W).to(device)[None,:]
    d_canvas[:,:,1] = torch.linspace(0, H-1, H).to(device)[:,None]
    K_inv = torch.linalg.inv(K)
    rays_d_cam0 = d_canvas @ (K_inv[:3,:3].T)[None]
    rays_d_wrd = rays_d_cam0 @ (Rt[None,:3,:3])
    rays_d_cam1 = rays_d_wrd @ (Rt_aux[:3,:3].T)[None]
    rays_d_pix1 = rays_d_cam1 @ (K_aux[:3,:3].T)[None]
    
    mask_candidate = (rays_d_pix1[:,:,2] > 1e-3)
    rays_d_selected0 = rays_d_pix1[mask_candidate]
    rays_d_selected0_uv = rays_d_selected0[:,:2]/rays_d_selected0[:,2:]

    mask_candidate1 = (rays_d_selected0_uv[:,0] > 0) * (rays_d_selected0_uv[:,1] > 0) * (rays_d_selected0_uv[:,0] < W-1) * (rays_d_selected0_uv[:,1] < H-1)
    uu = mask_candidate.clone()
    mask_candidate[uu] = mask_candidate1
    rays_d_selected0_uv = rays_d_selected0_uv[mask_candidate1]
    # print(canvas_aux.device, rays_d_selected0_uv.device, canvas_aux.shape, rays_d_selected0_uv.shape, mask_candidate.shape)
    color_sample = canvas_aux[rays_d_selected0_uv[:,1].long(), rays_d_selected0_uv[:,0].long()]
    canvas[mask_candidate] = color_sample
    return canvas, mask_candidate


def render_pointcloud_with_aux_frames(H, W, pcs, Ks, Rts, rgbs=None, device="cuda:0", aux_frames=None, Ks_aux=None, Rts_aux=None):
    if isinstance(pcs, np.ndarray) or isinstance(Ks, np.ndarray) or isinstance(Rts, np.ndarray):
        pcs = torch.from_numpy(pcs).float().to(device)
        Ks = torch.from_numpy(Ks).float().to(device)
        Rts = torch.from_numpy(Rts).float().to(device)
        if rgbs is not None:
            rgbs = torch.from_numpy(rgbs).float().to(device)

    N = Ks.shape[0]
    all_frames = []
    all_masks = []
    for i in range(N):
        print(f"{i}...")
        cur_canvas = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        sdadsa = torch.zeros((H, W, 1), dtype=torch.float32, device=device)
        cur_depth_canvas = 1e5 * torch.ones((H, W), dtype=torch.float32, device=device)
        pcs_cam = pcs @ Rts[i,:3,:3].T + Rts[i,:3,3][None, :]
        pcs_pix = pcs_cam @ (Ks[i])[:3,:3].T
        pcs_depth = pcs_pix[:,2]
        pcs_uv = pcs_pix[:,:2] / pcs_pix[:,2:]        

        cur_mask = (pcs_uv[:,0] >= 1.) * (pcs_uv[:,0] < W-1) * (pcs_uv[:,1] >= 1.) * (pcs_uv[:,1] < H-1) * (pcs_depth > 1e-3)
        cur_pcs_uv = (pcs_uv[cur_mask] + 0.5).long()
        cur_pcs_depth = pcs_depth[cur_mask]
        rgb_valid = rgbs[cur_mask]

        agsort = torch.argsort(cur_pcs_depth,descending=True)
        cur_canvas[cur_pcs_uv[agsort,1], cur_pcs_uv[agsort,0]] = rgb_valid[agsort]
        sdadsa[cur_pcs_uv[agsort,1], cur_pcs_uv[agsort,0]] = 1.

        cur_canvas_, cur_mask_ = sample_from_aux_frame(cur_canvas, Ks[i], Rts[i], aux_frames[i], Ks_aux[i], Rts_aux[i])
        cur_canvas[cur_mask_] = cur_canvas_[cur_mask_]
        sdadsa[cur_mask_] = True
        # Npc = cur_pcs_uv.shape[0]
        # print(Npc)
        # for j in range(Npc):
        # if cur_depth_canvas[cur_pcs_uv[j,1], cur_pcs_uv[j,0]] > pcs_depth[j]:
        #    cur_depth_canvas[cur_pcs_uv[j,1], cur_pcs_uv[j,0]] = pcs_depth[j]
        #    cur_canvas[cur_pcs_uv[j,1], cur_pcs_uv[j,0]] = rgb_valid[j]
        all_frames.append(cur_canvas * 2. - 1.)
        all_masks.append(sdadsa)
    
    return torch.stack(all_frames, dim=0).permute(0,3,1,2), torch.stack(all_masks, dim=0).permute(0,3,1,2)


def image_uv(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Get image space UV grid, ranging in [0, 1]. 

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        np.ndarray: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = np.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, dtype=dtype)
    v = np.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    return np.stack([u, v], axis=2)

def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions
import warnings
class no_warnings:
    def __init__(self, action: str = 'ignore', **kwargs):
        self.action = action
        self.filter_kwargs = kwargs
    
    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(self.action, **self.filter_kwargs)
                return fn(*args, **kwargs)
        return wrapper  
    
    def __enter__(self):
        self.warnings_manager = warnings.catch_warnings()
        self.warnings_manager.__enter__()
        warnings.simplefilter(self.action, **self.filter_kwargs)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.warnings_manager.__exit__(exc_type, exc_val, exc_tb)
def project_cv(
        points: np.ndarray,
        extrinsics: np.ndarray = None,
        intrinsics: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D following the OpenCV convention

    Args:
        points (np.ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last
            dimension is 4, the points are assumed to be in homogeneous coordinates
        extrinsics (np.ndarray): [..., 4, 4] extrinsics matrix
        intrinsics (np.ndarray): [..., 3, 3] intrinsics matrix

    Returns:
        uv_coord (np.ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (np.ndarray): [..., N] linear depth
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    if extrinsics is not None:
        points = points @ extrinsics.swapaxes(-1, -2)
    points = points[..., :3] @ intrinsics.swapaxes(-1, -2)
    with no_warnings():
        uv_coord = points[..., :2] / points[..., 2:]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth

def uv_to_pixel(
    uv: np.ndarray,
    width: Union[int, np.ndarray],
    height: Union[int, np.ndarray]
) -> np.ndarray:
    """
    Args:
        pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    pixel = uv * np.stack([width, height], axis=-1).astype(uv.dtype) - 0.5
    return pixel
def depthwarp_new(K0,Rt0,K1,Rt1,depth0,depth1,img0):
    # K&Rt: intrinsic and w2c matrix;
    # depthlist: a list of depthmap,given as raw depth, 3-channel np.float32 array
    # index0: the index of view from which the depth is warped
    # index1: the index of view to which the depth is warped

    H,W=depth0.shape[:2]

    xy_canvas = np.ones((H, W, 3), dtype=np.float32)
    xy_canvas[:,:,0] = np.linspace(0,W,num=W)[None,:]
    xy_canvas[:,:,1] = np.linspace(0,H,num=H)[:,None]
    pts_pixspace = xy_canvas * depth0[:,:,:1]

    Rt0_inv = np.zeros((3,4), dtype=np.float32)
    Rt0_inv[:3,:3] = Rt0[:3,:3].transpose(1,0)
    Rt0_inv[:,3:] = -Rt0[:3,:3].transpose(1,0)@Rt0[:3,3:]
    pts_camspace = (pts_pixspace[:,:,None,:]@np.linalg.inv(K0.transpose(1,0))[None,None])[:,:,0,:]
    pts_wrdspace = (pts_camspace[:,:,None,:]@(Rt0_inv[:3,:3].transpose(1,0))[None,None] + Rt0_inv[:3,3][None,None,None])[:,:,0,:]
    pts_cam2 = (pts_wrdspace[:,:,None,:]@Rt1[:3,:3].transpose(1,0)[None,None] + Rt1[:3,3][None,None,None])[:,:,0,:]
    pts_pix2 = (pts_cam2[:,:,None,:]@K1.transpose(1,0)[None,None])[:,:,0,:]
    
    pts_pix2_depth = pts_pix2[:,:,2:]
    # print(pts_pix2_depth.max(), pts_pix2_depth.min())
    pts_pix2 = pts_pix2/pts_pix2[:,:,2:]
    pts_pix2 = pts_pix2.astype(np.int32)[...,:2]
    
    # canvas for visualize
    canvas = np.zeros((H,W,3),dtype=img0.dtype)
    mask = ((pts_pix2[...,0]>=0) * (pts_pix2[...,0]<W)) * ((pts_pix2[...,1]>=0) * (pts_pix2[...,1]<H))
    
    
    valid = pts_pix2[mask]
    depth0_sample = pts_pix2_depth[...,0][mask]
    depth1_sample = depth1[...,0][valid[:,1],valid[:,0]]
    # q = depth0_sample <= (depth1_sample + 10)
    # print(q.shape,q.sum())
    # that is what i know.
    # and what else?
    #mask[mask] = q
    mask = mask * (pts_pix2_depth[...,0] > 0.)
    # print(f"shape chekc: {pts_pix2.shape} {mask.shape} ")
    valid = pts_pix2[mask]
    img0_color_sample = img0[mask]
    mask_return = np.zeros_like(mask)
    mask_return[valid[:,1],valid[:,0]] = True
    canvas[valid[:,1],valid[:,0]] = img0_color_sample
    return canvas, mask_return
def merge_panorama_image(width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics_: List[np.ndarray], init_frame=None, init_mask=None, mode="overwrite"):
    # preprocess:
    N = len(intrinsics_)
    intrinsics = []
    H, W = distance_maps[0].shape[:2]
    for i in range(N):
        ss = intrinsics_[i].copy()
        ss[0,...] /= W
        ss[1,...] /= H
        intrinsics.append(ss)
    uv = image_uv(width=width, height=height)
    spherical_directions = spherical_uv_to_directions(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    
    panorama_log_distance_map = init_frame
    panorama_pred_mask = init_mask
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        # print(projected_uv.shape, projected_depth.shape)
        projection_valid_mask = (projected_depth >= 0) & (projected_uv >= 0).all(axis=-1) & (projected_uv <= 1).all(axis=-1)        
        projected_pixels = uv_to_pixel(np.clip(projected_uv, 0, 1), width=distance_maps[i].shape[1], height=distance_maps[i].shape[0]).astype(np.float32)

        panorama_log_distance_map_r = np.where(projection_valid_mask, cv2.remap(distance_maps[i][...,0], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_log_distance_map_g = np.where(projection_valid_mask, cv2.remap(distance_maps[i][...,1], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_log_distance_map_b = np.where(projection_valid_mask, cv2.remap(distance_maps[i][...,2], projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        
        panorama_log_distance_map_ = np.stack([panorama_log_distance_map_r, panorama_log_distance_map_g, panorama_log_distance_map_b], axis=-1)
        panorama_pred_mask_ = projection_valid_mask & (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)
        # print(panorama_log_distance_map.shape, panorama_pred_mask.shape)
        # print(panorama_pred_mask_.dtype, panorama_pred_mask_.shape)
        # cv2.imwrite("./sdasd.png", panorama_pred_mask_.astype(np.uint8)*255)
        if panorama_log_distance_map is None:
            panorama_log_distance_map = panorama_log_distance_map_
            panorama_pred_mask = panorama_pred_mask_
        else:
            if mode=="overwrite":
                panorama_log_distance_map[panorama_pred_mask_] = panorama_log_distance_map_[panorama_pred_mask_]
            else:
                uu = panorama_pred_mask_ * panorama_pred_mask
                uu1 = (~panorama_pred_mask_) * panorama_pred_mask
                uu2 = (panorama_pred_mask_) * (~panorama_pred_mask)
                panorama_log_distance_map[uu] = (panorama_log_distance_map_[uu] + panorama_log_distance_map[uu])/2.
                panorama_log_distance_map[uu2] = (panorama_log_distance_map_[uu2])
            panorama_pred_mask += panorama_pred_mask_
        
    return panorama_log_distance_map, panorama_pred_mask
def unproject_cv(
    uv_coord: np.ndarray,
    depth: np.ndarray = None,
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None
) -> np.ndarray:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    Args:
        uv_coord (np.ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (np.ndarray): [..., N] depth value
        extrinsics (np.ndarray): [..., 4, 4] extrinsics matrix
        intrinsics (np.ndarray): [..., 3, 3] intrinsics matrix

    Returns:
        points (np.ndarray): [..., N, 3] 3d points
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    points = np.concatenate([uv_coord, np.ones_like(uv_coord[..., :1])], axis=-1)
    points = points @ np.linalg.inv(intrinsics).swapaxes(-1, -2)
    if depth is not None:
        points = points * depth[..., None]
    if extrinsics is not None:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        points = (points @ np.linalg.inv(extrinsics).swapaxes(-1, -2))[..., :3]
    return points
def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)

def image_border_padding(img, padding_pix = 3):
    img[:padding_pix, padding_pix:-padding_pix] = img[padding_pix, padding_pix:-padding_pix][None]
    img[-padding_pix:, padding_pix:-padding_pix] = img[-padding_pix, padding_pix:-padding_pix][None]
    img[padding_pix:-padding_pix, :padding_pix] = img[padding_pix:-padding_pix, padding_pix][:, None]
    img[padding_pix:-padding_pix, -padding_pix:] = img[padding_pix:-padding_pix, -padding_pix][:, None]
    img[:padding_pix, :padding_pix] = img[padding_pix, padding_pix][None, None]
    img[:padding_pix, -padding_pix:] = img[padding_pix, -padding_pix][None,None]
    img[-padding_pix:, :padding_pix] = img[-padding_pix, padding_pix][None,None]
    img[-padding_pix:, -padding_pix:] = img[-padding_pix, -padding_pix][None, None]
    return img
    
def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray, intrinsics_: np.ndarray, resolution: Union[int,tuple]):
    # print(extrinsics.shape, intrinsics_.shape)
    height, width = image.shape[:2]

    N = len(intrinsics_)
    intrinsics = []
    W, H = resolution
    for i in range(N):
        ss = intrinsics_[i].copy()
        ss[0,...] /= W
        ss[1,...] /= H
        intrinsics.append(ss)

    if isinstance(resolution, int):
        uv = image_uv(width=resolution, height=resolution)
    else:
        uv = image_uv(width=resolution[0], height=resolution[1])
    splitted_images = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)

        splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_NEAREST) 
        splitted_image = image_border_padding(splitted_image, 2)   
        splitted_images.append(splitted_image)
    return splitted_images

# 给定首帧深度，一组点云和对应的rgb，远景深度的threshold，以及包括首帧在内的相机内外参。
# 首先，将首帧深度根据threshold切成两部分，将所有投影位于远景部分的点云作为远景点云，所有投影位于近景部分的点云作为近景点云。
# 接着，根据首帧深度生成两种mesh，一种有切割，一种没有切割。
# 对于每帧后续相机，计算两种mesh在当前相机下的深度；计算远景点云在没切割的深度图上的投影，如果投影在前景深度区域，则不选，否则选上，计算近景点云在切割的深度图上的投影，如果投影在前景深度区域，则不选，否则选上。

def split_pointclouds_based_on_first_frame(depth, pcs, rgbs, thres, intrinsics, extrinsics):

    w2c_ref = extrinsics[0]
    c2w_ref = np.linalg.inv(w2c_ref)
    pos = c2w_ref[:3,3]

    dist = np.linalg.norm(pcs - pos[None], axis=-1)
    mask = dist < thres*0.98

    fg_pcs = pcs[mask]
    fg_rgbs = rgbs[mask]

    bg_pcs = pcs[~mask]
    bg_rgbs = rgbs[~mask]

    bmask = depth<thres*0.7
    mesh_with_cut = generate_mesh_from_depth(depth[...,None], intrinsics[0], extrinsics[0], threshold = 0.05)
    mesh_wo_cut = generate_mesh_from_depth_with_mask(depth[...,None], intrinsics[0], extrinsics[0],bmask, threshold=1e4)
    mesh_wo_cut.export("../debug/A.obj")
    H, W = depth.shape[:2]
    N = intrinsics.shape[0]

    all_selected_pcs = []
    all_selected_rgbs = []
    # print(f"pts shape check: {bg_pcs.shape} {bg_rgbs.shape}")
    for i in range(N):
        depth_with_cut = mesh_to_depth(mesh_with_cut, intrinsics[i], extrinsics[i], H, W)
        depth_wo_cut = mesh_to_depth(mesh_wo_cut, intrinsics[i], extrinsics[i], H, W)

        # select foreground pointclouds

        proj_valid_uv, proj_valid_depth, proj_valid_mask = proj_points_on_image(H, W, fg_pcs, intrinsics[i], extrinsics[i])
        sampled_depth_with_cut = depth_with_cut[proj_valid_uv[:,1], proj_valid_uv[:,0]]
        valid_mask_fg = np.abs(sampled_depth_with_cut - proj_valid_depth) < 0.5
        fg_pcs_proj = fg_pcs[proj_valid_mask][valid_mask_fg]
        fg_rgbs_proj = fg_rgbs[proj_valid_mask][valid_mask_fg]

        # select background pointclouds

        proj_valid_uv, proj_valid_depth, proj_valid_mask = proj_points_on_image(H, W, bg_pcs, intrinsics[i], extrinsics[i])
        sampled_depth_wo_cut = depth_wo_cut[proj_valid_uv[:,1], proj_valid_uv[:,0]]
        valid_mask_bg = sampled_depth_wo_cut < 1e-3
        # print(f"sda: {valid_mask_bg.sum()}")
        bg_pcs_proj = bg_pcs[proj_valid_mask][valid_mask_bg]
        bg_rgbs_proj = bg_rgbs[proj_valid_mask][valid_mask_bg]

        cur_valid_pcs = np.concatenate([fg_pcs_proj, bg_pcs_proj],axis=0)
        cur_valid_rgb = np.concatenate([fg_rgbs_proj, bg_rgbs_proj],axis=0)

        all_selected_pcs.append(cur_valid_pcs)
        all_selected_rgbs.append(cur_valid_rgb)
    return all_selected_pcs, all_selected_rgbs

def split_pointclouds_based_on_first_frame_torch(depth, pcs, rgbs, thres, intrinsics, extrinsics):
    device = depth.device
    w2c_ref = extrinsics[0]
    c2w_ref = torch.linalg.inv(w2c_ref)
    pos = c2w_ref[:3,3]

    dist = (pcs - pos[None]).norm(dim=-1)
    mask = dist < thres*0.98

    fg_pcs = pcs[mask]
    fg_rgbs = rgbs[mask]

    bg_pcs = pcs[~mask]
    bg_rgbs = rgbs[~mask]

    depth_np = depth.cpu().numpy()
    intrinsics_np = intrinsics.cpu().numpy()
    extrinsics_np = extrinsics.cpu().numpy()
    bmask = depth_np<thres*0.7
    mesh_with_cut = generate_mesh_from_depth(depth_np[...,None], intrinsics_np[0], extrinsics_np[0], threshold = 0.05)
    mesh_wo_cut = generate_mesh_from_depth_with_mask(depth_np[...,None], intrinsics_np[0], extrinsics_np[0],bmask, threshold=1e4)
    # mesh_wo_cut.export("../debug/A.obj")
    H, W = depth.shape[:2]
    N = intrinsics.shape[0]

    all_selected_pcs = []
    all_selected_rgbs = []
    # print(f"pts shape check: {bg_pcs.shape} {bg_rgbs.shape}")
    for i in range(N):
        depth_with_cut = torch.from_numpy(mesh_to_depth(mesh_with_cut, intrinsics_np[i], extrinsics_np[i], H, W)).float().to(device)
        depth_wo_cut = torch.from_numpy(mesh_to_depth(mesh_wo_cut, intrinsics_np[i], extrinsics_np[i], H, W)).float().to(device)

        # select foreground pointclouds

        proj_valid_uv, proj_valid_depth, proj_valid_mask = proj_points_on_image(H, W, fg_pcs, intrinsics[i], extrinsics[i])
        sampled_depth_with_cut = depth_with_cut[proj_valid_uv[:,1], proj_valid_uv[:,0]]
        valid_mask_fg = torch.abs(sampled_depth_with_cut - proj_valid_depth) < 0.5
        fg_pcs_proj = fg_pcs[proj_valid_mask][valid_mask_fg]
        fg_rgbs_proj = fg_rgbs[proj_valid_mask][valid_mask_fg]

        # select background pointclouds

        proj_valid_uv, proj_valid_depth, proj_valid_mask = proj_points_on_image(H, W, bg_pcs, intrinsics[i], extrinsics[i])
        sampled_depth_wo_cut = depth_wo_cut[proj_valid_uv[:,1], proj_valid_uv[:,0]]
        valid_mask_bg = sampled_depth_wo_cut < 1e-3
        # print(f"sda: {valid_mask_bg.sum()}")
        bg_pcs_proj = bg_pcs[proj_valid_mask][valid_mask_bg]
        bg_rgbs_proj = bg_rgbs[proj_valid_mask][valid_mask_bg]

        cur_valid_pcs = torch.cat([fg_pcs_proj, bg_pcs_proj],dim=0)
        cur_valid_rgb = torch.cat([fg_rgbs_proj, bg_rgbs_proj],dim=0)

        all_selected_pcs.append(cur_valid_pcs)
        all_selected_rgbs.append(cur_valid_rgb)
    return all_selected_pcs, all_selected_rgbs





def get_colored_mesh_from_pano(pano_depth, pano_rgb, pano_mask, origin_position=np.zeros((3,),dtype=np.float32)):
    # ok...
    # pano_depth: [H, W] numpy array
    # pano_rgb: [H, W, 3] numpy array
    # spherical_uv_to_directions(uv: np.ndarray):
    H, W = pano_depth.shape
    uv = image_uv(width=W, height=H)
    directions = spherical_uv_to_directions(uv)
    vertices = pano_depth[:,:,None] * directions + origin_position
    
    pts_topology = np.zeros((H - 1, W - 1, 2, 3), dtype=np.int32)
    index_ = np.arange(H*W).astype(np.int32).reshape(H,W)
    pts_topology[:,:,0,0] = index_[:-1,:-1]
    pts_topology[:,:,0,2] = index_[:-1,1:]
    pts_topology[:,:,0,1] = index_[1:,:-1]

    pts_topology[:,:,1,0] = index_[:-1,1:]
    pts_topology[:,:,1,1] = index_[1:,:-1]
    pts_topology[:,:,1,2] = index_[1:,1:]

    topo_mask_0 = pano_mask[:-1,:-1] * pano_mask[:-1,1:] * pano_mask[1:,:-1]
    topo_mask_1 = pano_mask[:-1,1:] * pano_mask[1:,:-1] * pano_mask[1:,1:]

    topo0 = pts_topology[:,:,0][topo_mask_0]
    topo1 = pts_topology[:,:,1][topo_mask_1]

    topo = np.concatenate([topo0, topo1], axis=0)

    vertices = vertices.reshape(-1,3)
    rgbs = pano_rgb.reshape(-1,3)
    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=topo, vertex_colors=rgbs)

    print(f"mesh construction complete; mesh info: {mesh_trimesh.vertices.shape} {mesh_trimesh.faces.shape}")

    return mesh_trimesh

def render_mesh(mesh_trimesh, Ks, Rts, H, W):
    '''
    material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1, 1, 1, 1.],
            metallicFactor=0.0,
            roughnessFactor=0.0,
            smooth=False,
            alphaMode='OPAQUE')
    
    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, material=material)
    camera = pyrender.IntrinsicsCamera(320, 240, 320, 320, znear=1, zfar=1000)
    renderer = pyrender.OffscreenRenderer(viewport_width=320, viewport_height=320)

    scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=np.ones(3))  
    
    '''
    material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1, 1, 1, 1.],
            metallicFactor=1.0,
            roughnessFactor=1.0,
            smooth=False,
            alphaMode='OPAQUE')
    # mesh_trimesh.visual.vertex_colors = mesh_trimesh.visual.vertex_colors.astype(np.float32)/255.
    ssd = mesh_trimesh.visual.vertex_colors.astype(np.float32)
    # print("sdadsada: ",ssd[ssd<254].max())
    # mesh_trimesh.visual.vertex_colors = ssd
    
    # mesh_trimesh.visual.vertex_colors[:, :3] = np.array([[10,20,30]])
    
    # convert K1 to pyrender camera;

    rgb_list = []
    depth_list = []
    mask_list = []
    N = Ks.shape[0]
    scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=np.ones(3))
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, material=material)
    scene.add(mesh)
    nc = None
    r = None
    for i in range(N):
        
        aspect_ratio = float(W)/H
        yfov = 2.*math.atan(float(H)/(Ks[i,1,1]*2.))
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)

        camera_pose = np.eye(4)
        camera_pose[:3,:3] = Rts[i,:3,:3].transpose(1,0)
        camera_pose[:3, 3] = -camera_pose[:3,:3] @ Rts[i,:3,3]
        camera_pose[:,1] *= -1.
        camera_pose[:,2] *= -1.
        if nc is None:
            nc = scene.add(camera, pose=camera_pose)
        else:
            scene.set_pose(nc, pose=camera_pose)
        if r is None:
            r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        rgb, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
        # scene.remove_node(nc)
        rgb_list.append(rgb)
        # print(rgb.shape, rgb.dtype)
        depth_list.append(depth)
        mask = depth > 1e-3
        mask_list.append(mask)
        # r.delete()

    rgb = np.stack(rgb_list, axis=0)
    depth = np.stack(depth_list, axis=0)
    mask = np.stack(mask_list, axis=0)
    r.delete()
    return rgb, depth, mask
def get_panorama_cameras_elevation_full(fov_x=90.,fov_y=90., end_angle=270.,view_num=36, elevation_angle=[45.,],elevation_view_size=[4,]):
    # begin view:[0,0,0]
    # currently we assume things move around z axis.
    vertices = []

    x_axis = np.array([1.,0,0], dtype=np.float32)
    y_axis = np.array([0.,1,0], dtype=np.float32)
    z_axis = np.array([0.,0,1.], dtype=np.float32)

    for elev, elev_size in zip(elevation_angle, elevation_view_size):
        for i in range(elev_size):
            cur_vertex = math.cos(math.radians(elev)) * (x_axis * math.cos(math.radians(end_angle+i*360./elev_size)) + y_axis * math.sin(math.radians(end_angle+i*360./elev_size))) + math.sin(math.radians(elev)) * z_axis
            vertices.append(cur_vertex)
            cur_vertex = math.cos(math.radians(-elev)) * (x_axis * math.cos(math.radians(end_angle+i*360./elev_size)) + y_axis * math.sin(math.radians(end_angle+i*360./elev_size))) + math.sin(math.radians(-elev)) * z_axis
            vertices.append(cur_vertex)

    for i in range(view_num):
        cur_vertex = x_axis * math.cos(math.radians(end_angle+i*360./view_num)) + y_axis * math.sin(math.radians(end_angle+i*360./view_num))
        vertices.append(cur_vertex)
    vertices = np.stack(vertices)


    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(fov_x), fov_y=np.deg2rad(fov_y))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)
    
def generate_masked_video_mesh(rgb, depth, end_angle, frame_number, movement_ratio):
    # (width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics_: List[np.ndarray], init_frame=None, init_mask=None, mode="overwrite"):
    H, W = rgb.shape[:2]
    mid_pivot = (int(H//2 * 1.1), int((end_angle/360.) * W + W//2) % W)
    depth_value = depth[mid_pivot[0], mid_pivot[1]]

    depth = np.concatenate([depth[:,mid_pivot[1]:],depth[:,:mid_pivot[1]]], axis=1)
    rgb = np.concatenate([rgb[:,mid_pivot[1]:],rgb[:,:mid_pivot[1]]], axis=1)

    depth = np.concatenate([depth[:,W//2:],depth[:,:W//2]], axis=1)
    rgb = np.concatenate([rgb[:,W//2:],rgb[:,:W//2]], axis=1)

    rot_matrix = np.array([
        [0,1,0],[0,0,-1],[-1,0,0.]
    ], dtype=np.float32)
    # pano_depth, pano_rgb, pano_mask, origin_position=np.zeros((3,),dtype=np.float32)
    depth_edge_mask = ~depth_edge(depth, rtol=0.1)
    mesh_masked = get_colored_mesh_from_pano(depth, rgb, depth_edge_mask)
    mesh_full = get_colored_mesh_from_pano(depth, rgb, np.ones_like(depth_edge_mask))
    # mesh_masked.vertices = mesh_masked.vertices @ rot_matrix.T
    # mesh_full.vertices = mesh_full.vertices @ rot_matrix.T
    
    Rts, Ks = get_panorama_cameras_elevation_full(end_angle = 180., view_num=5)
    Ks = np.stack(Ks, axis=0)
    
    
    H, W = 512,512
    Ks[:,:2] *= H
    all_merged_frames = []
    all_merged_masks = []
    for i in range(frame_number):
        Rts = np.linalg.inv(Rts)
        Rts[:,0,3] = -depth_value * (i/(frame_number-1.)) * movement_ratio
        Rts = np.linalg.inv(Rts)
        if i == 0:
            images,_, mask = render_mesh(mesh_full, Ks, Rts, H, W)
        else:
            images,_, mask = render_mesh(mesh_masked, Ks, Rts, H, W)
        Rts[:,:3,3] = 0.
        mask = mask.astype(np.float32)[:,:,:,None].repeat(3,3)
        # print(f"sdasdfafa: {images.max()} {images.min()}")
        merged_frame,_ = merge_panorama_image(2048, 1024, [images[i].astype(np.float32)/255. for i in range(images.shape[0])], [np.ones(images[i].shape[:2], dtype=np.bool_) for i in range(images.shape[0])], [Rts[i] for i in range(Rts.shape[0])], [Ks[i] for i in range(Ks.shape[0])])
        merged_mask,_ = merge_panorama_image(2048, 1024, [mask[i] for i in range(mask.shape[0])], [np.ones(images[i].shape[:2], dtype=np.bool_) for i in range(images.shape[0])], [Rts[i] for i in range(Rts.shape[0])], [Ks[i] for i in range(Ks.shape[0])])
        merged_mask = merged_mask[:,:,0] > 0.5
        all_merged_frames.append(merged_frame)
        all_merged_masks.append(merged_mask)
    return np.stack(all_merged_frames, axis=0), np.stack(all_merged_masks, axis=0)
def get_diffrast_camera_parameter_from_cv(K, H, W, near, far, device):
    K_ = torch.zeros((4,4), dtype=torch.float32, device=device)
    K_[0,0] = K[0,0] * 2. / W
    K_[1,1] = K[1,1] * 2. / H
    K_[2,2] = (far + near) / (far - near) 
    K_[2,3] = - 2. * (near * far) / (far - near) 
    K_[3,2] = 1.
    return K_
def fill_image_nvdiffrast(merged_frame, merged_mask):
    to_fill = (~merged_mask[:,1:-1,1:-1]) * merged_mask[:,:-2,1:-1]
    merged_frame[:,1:-1,1:-1][to_fill] = merged_frame[:,:-2,1:-1][to_fill]

    to_fill = (~merged_mask[:,1:-1,1:-1]) * merged_mask[:,2:,1:-1]
    merged_frame[:,1:-1,1:-1][to_fill] = merged_frame[:,2:,1:-1][to_fill]

    to_fill = (~merged_mask[:,1:-1,1:-1]) * merged_mask[:,1:-1,2:]
    merged_frame[:,1:-1,1:-1][to_fill] = merged_frame[:,1:-1,2:][to_fill]

    to_fill = (~merged_mask[:,1:-1,1:-1]) * merged_mask[:,1:-1,:-2]
    merged_frame[:,1:-1,1:-1][to_fill] = merged_frame[:,1:-1,:-2][to_fill]
    
    merged_mask[:,1:-1,1:-1] = merged_mask[:,1:-1,1:-1] + merged_mask[:,1:-1,:-2] + merged_mask[:,1:-1,2:] + merged_mask[:,:-2,1:-1] + merged_mask[:,2:,1:-1]
    return merged_frame, merged_mask
def get_mesh_render(mesh, Ks, Rts, H, W, near, far, device):
    with torch.no_grad():
        if isinstance(Ks, np.ndarray):
            Ks = torch.from_numpy(Ks).float().to(device)
            Rts = torch.from_numpy(Rts).float().to(device)
        glctx = dr.RasterizeCudaContext(device=device)
        pos_tensor = torch.from_numpy(np.array(mesh.vertices)).float().to(device)
        col_tensor = torch.from_numpy(np.array(mesh.visual.vertex_colors)).float().to(device)
        tri_tensor = torch.from_numpy(np.array(mesh.faces)).int().to(device)
        if col_tensor.max() > 5:
            col_tensor /= 255.
        pos_qc = torch.ones((pos_tensor.shape[0],4),dtype=torch.float32, device=device)

        N = Rts.shape[0]
        all_images = []

        for i in range(N):
            pos_tensor_cam = pos_tensor @ Rts[i,:3,:3].T + Rts[i:i+1,:3,3]
            K_ = get_diffrast_camera_parameter_from_cv(Ks[i], H, W, near, far, device)
            pos_qc[:,:3] = pos_tensor_cam
            pos_rast = (pos_qc @ K_.T)[None]
            rast, _ = dr.rasterize(glctx, pos_rast, tri_tensor, resolution=[H, W])
            out, _ = dr.interpolate(col_tensor[None], rast, tri_tensor)
            # out = dr.antialias(out, rast, pos_rast, tri_tensor, topology_hash=None, pos_gradient_boost=1.0)
            #out = dr.antialias(out, rast, pos_rast, tri_tensor, topology_hash=None, pos_gradient_boost=1.0)
            img = out.cpu().numpy()[0, :, :, :] # Flip vertically.
            #img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
            all_images.append(img)
        all_images = np.stack(all_images, axis=0)
        all_canvas = all_images[:,:,:,:3]
        all_canvas_mask = all_images[:,:,:,3] > 0.9
        #all_canvas,all_canvas_mask = fill_image_nvdiffrast(all_canvas,all_canvas_mask)
        return all_canvas,all_canvas_mask
def floodfill_new(image, mask):
    # image: [H,W,3] np.array
    # mask: [H,W] np.array
    # fill image pixels outside the mask;
    img_np = np.array(image)[:,:,:3]
    mask_np = ~(np.array(mask) > 0)
    # print(mask_np.shape)
    _,ind = distance_transform_edt(~mask_np,return_indices = True)
    # print(img_np[ind[0],ind[1]].shape,ind.shape)
    img_np[~mask_np] = img_np[ind[0],ind[1]][~mask_np]
    return img_np # Image.fromarray(img_np.astype(np.uint8)) 

def generate_masked_video_mesh_nvdiffrast(rgb, depth, end_angle, frame_number, movement_ratio, near, far, device):
    # (width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics_: List[np.ndarray], init_frame=None, init_mask=None, mode="overwrite"):
    H, W = rgb.shape[:2]
    mid_pivot = (int(H//2 * 1.1), int((end_angle/360.) * W + W//2) % W)
    depth_value = depth[mid_pivot[0], mid_pivot[1]]

    depth = np.concatenate([depth[:,mid_pivot[1]:],depth[:,:mid_pivot[1]]], axis=1)
    rgb = np.concatenate([rgb[:,mid_pivot[1]:],rgb[:,:mid_pivot[1]]], axis=1)

    depth = np.concatenate([depth[:,W//2:],depth[:,:W//2]], axis=1)
    rgb = np.concatenate([rgb[:,W//2:],rgb[:,:W//2]], axis=1)

    rot_matrix = np.array([
        [0,1,0],[0,0,-1],[-1,0,0.]
    ], dtype=np.float32)
    # pano_depth, pano_rgb, pano_mask, origin_position=np.zeros((3,),dtype=np.float32)
    depth_edge_mask = ~depth_edge(depth, rtol=0.1)
    mesh_masked = get_colored_mesh_from_pano(depth, rgb, depth_edge_mask)
    mesh_full = get_colored_mesh_from_pano(depth, rgb, np.ones_like(depth_edge_mask))
    # mesh_masked.vertices = mesh_masked.vertices @ rot_matrix.T
    # mesh_full.vertices = mesh_full.vertices @ rot_matrix.T
    
    Rts, Ks = get_panorama_cameras_elevation_full(end_angle = 180., view_num=5)
    Ks = np.stack(Ks, axis=0)
    
    
    H, W = 512,512
    Ks[:,:2] *= H
    all_merged_frames = []
    all_merged_masks = []
    for i in range(frame_number):
        print(f"{i}...")
        Rts = np.linalg.inv(Rts)
        Rts[:,0,3] = -depth_value * (i/(frame_number-1.)) * movement_ratio
        Rts = np.linalg.inv(Rts)
        if i == 0:
            #images,_, mask = render_mesh(mesh_full, Ks, Rts, H, W)
            images, mask = get_mesh_render(mesh_full, Ks, Rts, H, W, near, far,device)
        else:
            #images,_, mask = render_mesh(mesh_masked, Ks, Rts, H, W)
            images, mask = get_mesh_render(mesh_masked, Ks, Rts, H, W, near, far, device)
        images = np.clip(images,0,1)
        Rts[:,:3,3] = 0.
        mask = mask.astype(np.float32)[:,:,:,None].repeat(3,3)
        #print(f"sdasdfafa: {images.max()} {images.min()}")
        cv2.imwrite("./sdads.png",(images[0] * 255).astype(np.uint8))
        merged_frame,_ = merge_panorama_image(2048, 1024, [images[i].astype(np.float32) for i in range(images.shape[0])], [np.ones(images[i].shape[:2], dtype=np.bool_) for i in range(images.shape[0])], [Rts[i] for i in range(Rts.shape[0])], [Ks[i] for i in range(Ks.shape[0])])
        merged_mask,_ = merge_panorama_image(2048, 1024, [mask[i] for i in range(mask.shape[0])], [np.ones(images[i].shape[:2], dtype=np.bool_) for i in range(images.shape[0])], [Rts[i] for i in range(Rts.shape[0])], [Ks[i] for i in range(Ks.shape[0])])
        merged_frame = cv2.resize(merged_frame, (672,384))
        merged_mask = cv2.resize(merged_mask, (672,384))
        merged_mask = merged_mask[:,:,0] > 0.8

        '''to_fill = (~merged_mask[1:-1,1:-1]) * merged_mask[:-2,1:-1]
        merged_frame[1:-1,1:-1][to_fill] = merged_frame[:-2,1:-1][to_fill]

        to_fill = (~merged_mask[1:-1,1:-1]) * merged_mask[2:,1:-1]
        merged_frame[1:-1,1:-1][to_fill] = merged_frame[2:,1:-1][to_fill]

        to_fill = (~merged_mask[1:-1,1:-1]) * merged_mask[1:-1,2:]
        merged_frame[1:-1,1:-1][to_fill] = merged_frame[1:-1,2:][to_fill]

        to_fill = (~merged_mask[1:-1,1:-1]) * merged_mask[1:-1,:-2]
        merged_frame[1:-1,1:-1][to_fill] = merged_frame[1:-1,:-2][to_fill]
        
        merged_mask[1:-1,1:-1] = merged_mask[1:-1,1:-1] + merged_mask[1:-1,:-2] + merged_mask[1:-1,2:] + merged_mask[:-2,1:-1] + merged_mask[2:,1:-1]'''

        cv2.imwrite("./sdads_pano.png",(merged_frame * 255).astype(np.uint8))
        all_merged_frames.append(merged_frame)
        all_merged_masks.append(merged_mask)
    return np.stack(all_merged_frames, axis=0), np.stack(all_merged_masks, axis=0)
if __name__=="__main__":

    print("try getting trimesh")
    base_dir = "/mnt/workspace/zhongqi.yang/VideoInpainting_new/output/seg_0402/case11_mesh/moge/11_out_superres"
    depth_path = os.path.join(base_dir, "depth.exr")
    rgb_path = os.path.join(base_dir, "img.png")
    mask_path = os.path.join(base_dir, "mask.png")
    process_dir = os.path.join(base_dir, "../../panorama_process")
    para_path = os.path.join(process_dir, "para.json")
    extrinsic_path = os.path.join(process_dir, "world_matrix_extended.npz")
    # extrinsic_path = os.path.join(process_dir, "world_matrix.npz")

    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    rgb = cv2.imread(rgb_path,cv2.IMREAD_UNCHANGED)[:,:,::-1].astype(np.float32)/255.
    mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)[:,:].astype(np.float32)
    depth[mask<0.5] = 20. 
    depth_edge_mask = (~depth_edge(depth, rtol = 0.05, kernel_size = 3, mask = None)).astype(np.float32)
    sz = (2048,4096)
    depth = cv2.resize(depth, sz, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.resize(rgb, sz, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(depth_edge_mask, sz, interpolation=cv2.INTER_NEAREST)>0.5
    mesh_trimesh = get_colored_mesh_from_pano(depth, rgb, mask)
    # mesh_trimesh.export("../../debug/Atest_pano.obj")
    index = 20
    K = load_intrinsic(para_path)
    _, Rts = read_blender_camera_as_opencv(extrinsic_path)
    Ks = K[None].repeat(20,0)
    Rts = Rts[index:index+20]
    H = 576
    W = 1024
    rgbs, depths,masks = render_mesh(mesh_trimesh, Ks, Rts, H, W)
    for i in range(rgbs.shape[0]):
        cv2.imwrite(f"../../debug/Atest_pano_render_{i}.png", rgbs[i][:,:,::-1])

