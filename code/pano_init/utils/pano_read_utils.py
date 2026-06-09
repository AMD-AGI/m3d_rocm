import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from packaging import version as pver
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

def unpack_mm_params(p):
    if isinstance(p, (tuple, list)):
        return p[0], p[1]
    elif isinstance(p, (int, float)):
        return p, p
    raise Exception(f'Unknown input parameter type.\nParameter: {p}.\nType: {type(p)}')


class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')
    
def euler_to_rotation_matrix(pitch, yaw, roll):
    """从欧拉角计算旋转矩阵"""
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ])

    Ry = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ])

    Rz = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    return R   

def euler_to_rotation_matrix_batch(pitch, yaw, roll):
    """从欧拉角计算旋转矩阵"""
    B = pitch.shape[0]
    Rx = torch.zeros((B,3,3))
    Ry = torch.zeros((B,3,3))
    Rz = torch.zeros((B,3,3))
    
    Rx[:,0,0] = 1.
    Ry[:,1,1] = 1.
    Rz[:,2,2] = 1.

    Rx[:,1,1] = torch.cos(roll)
    Rx[:,2,2] = torch.cos(roll)
    Rx[:,1,2] = -torch.sin(roll)
    Rx[:,2,1] = torch.sin(roll)
    
    Ry[:,0,0] = torch.cos(pitch)
    Ry[:,0,2] = torch.sin(pitch)
    Ry[:,2,0] = -torch.sin(pitch)
    Ry[:,2,2] = torch.cos(pitch)

    Rz[:,0,0] = torch.cos(yaw)
    Rz[:,1,1] = torch.cos(yaw)
    Rz[:,0,1] = -torch.sin(yaw)
    Rz[:,1,0] = torch.sin(yaw)

    R = Rz @ Ry @ Rx
    return R   

def compute_plucker(X, Y, Z, P, Yaw, R, H=480, W=720, device='cpu'):
    """计算 Plücker embedding"""
    # Convert float values to tensors
    P = torch.tensor(P, device=device)
    Yaw = torch.tensor(Yaw, device=device)
    R = torch.tensor(R, device=device)
    X = torch.tensor(X, device=device)
    Y = torch.tensor(Y, device=device)
    Z = torch.tensor(Z, device=device)

    # 计算 c2w 矩阵
    R_matrix = euler_to_rotation_matrix(torch.deg2rad(P), torch.deg2rad(Yaw), torch.deg2rad(R))
    T = torch.tensor([[X], [Y], [Z]], device=device)
    c2w = torch.eye(4, device=device)
    c2w[:3, :3] = R_matrix
    c2w[:3, 3] = T.squeeze()

    # Rest of the function remains the same
    j, i = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device),
        torch.linspace(0, W - 1, W, device=device),
        indexing='ij'
    )

    i = i.reshape(1, H * W) + 0.5
    j = j.reshape(1, H * W) + 0.5
    K = estimate_intrinsics()
    fx, fy, cx, cy = K[0], K[1], K[2], K[3]

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand_as(rays_d)

    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)

    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(H, W, 6)

    return plucker


def estimate_intrinsics(H=480, W=720):  #可能有问题
    fx = fy = W * 0.5  # 假设焦距为图像宽度的一半
    cx = W / 2
    cy = H / 2
    K = torch.tensor([fx, fy, cx, cy])
    return K

def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)         # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker
'''
from decord import VideoReader
from decord import cpu, gpu

vr = VideoReader('examples/flipping_a_pancake.mkv', ctx=cpu(0))
# a file like object works as well, for in-memory decoding
with open('examples/flipping_a_pancake.mkv', 'rb') as f:
  vr = VideoReader(f, ctx=cpu(0))
# 1. the simplest way is to directly access frames
for i in range(len(vr)):
    # the video reader will handle seeking and skipping in the most efficient manner
    frame = vr[i]

# To get multiple frames at once, use get_batch
# this is the efficient way to obtain a long list of frames
frames = vr.get_batch([1, 3, 5, 7, 9])
# (5, 240, 320, 3)
# duplicate frame indices will be accepted and handled internally to avoid duplicate decoding
frames2 = vr.get_batch([1, 2, 3, 2, 3, 4, 3, 4, 5]).asnumpy()
# (9, 240, 320, 3)

# 2. you can do cv2 style reading as well
# skip 100 frames
vr.skip_frames(100)
# seek to start
vr.seek(0)
batch = vr.next()

'''

class RealEstate10KPoseControlnetDataset(Dataset):
    def __init__(
            self,
            video_root_dir,
            file_name_txt,

            stride=(1, 1),
            sample_n_frames=49,
            relative_pose=True,
            zero_t_first_frame=True,
            image_size=[480, 720],
            rescale_fxy=True,
            shuffle_frames=False,
            hflip_p=0.0,
            first_start=False,
            use_text_prompt=False,
            use_xyz_norm=False,
            load_depth=False
    ):
        minimum_sample_stride, sample_stride = stride
        if hflip_p != 0.0:
            use_flip = True
        else:
            use_flip = False
        root_path = video_root_dir
        self.root_path = root_path
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.max_sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames
        self.first_start=first_start

        self.load_depth_ = load_depth
        # with open(os.path.join(root_path,file_name_txt), 'r') as f:
        #    self.dataset = f.readlines() 
        with open(os.path.join(root_path,file_name_txt), 'r') as f:
            self.dataset = f.readlines()

        #self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))
        # assert(len(self.dataset)==len(self.dataset_mask))
        self.length = len(self.dataset)
        sample_size = image_size
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        if use_flip:
            pixel_transforms = [transforms.Resize(sample_size),
                                RandomHorizontalFlipWithPose(hflip_p),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        else:
            pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip
        self.use_text_prompt=use_text_prompt
        self.use_xyz_norm=use_xyz_norm

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_video_reader(self, idx):
        video_id = self.dataset[idx]
        video_id=video_id.replace('\n','')
        video_path=video_id
        #print('video path:',video_path)
        if os.path.exists(video_path):
            #video_path = os.path.join(self.root_path, 'videos',f'{video_id}.mp4')
            video_reader = VideoReader(video_path)
            return  video_reader
        else:
            return None
    def load_mask_video_reader(self, idx):
        video_id = self.dataset_mask[idx]
        video_id=video_id.replace('\n','')
        video_path=video_id
        #print('video path:',video_path)
        if os.path.exists(video_path):
            #video_path = os.path.join(self.root_path, 'videos',f'{video_id}.mp4')
            video_reader = VideoReader(video_path)
            return  video_reader
        else:
            return None
        

    def load_cameras(self, idx, normalize_xyz=False):
        video_id = self.dataset[idx]
        video_id = video_id.replace('\n', '')

        # pose_file = video_id.replace('.mp4', '.csv')
        ls = ".".join(video_id.split(".")[:-1])
        fs = int(ls[-7:-4])
        pose_file = ls[:-4] + f".{fs-2}.csv"
        # pose_file = video_id.replace('.mp4', '.csv')
        if not os.path.exists(pose_file):
            pose_file=pose_file.replace('images','csv')  #for infer
        
        cam_params = []
        positions = []
        
        # First pass to collect all positions if normalization is needed
        with open(pose_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                frame_data = line.strip().split(',')
                if len(frame_data) != 3:
                    continue
                
                pos = frame_data[1].split()
                X = float(pos[0].split('=')[1])
                Y = float(pos[1].split('=')[1])
                Z = float(pos[2].split('=')[1])
                positions.append((X, Y, Z))
        
        # Calculate normalization parameters if needed
        if normalize_xyz and positions:
            positions = np.array(positions)
            min_vals = positions.min(axis=0)
            max_vals = positions.max(axis=0)
            scale = max_vals - min_vals
            scale[scale == 0] = 1  # Avoid division by zero
        
        # Second pass to process all parameters
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
                
                # Normalize XYZ if requested
                if normalize_xyz:
                    X = (X - min_vals[0]) / scale[0]
                    Y = (Y - min_vals[1]) / scale[1]
                    Z = (Z - min_vals[2]) / scale[2]
                
                # Parse rotation
                rot = frame_data[2].split()
                P = float(rot[0].split('=')[1])
                Yaw = float(rot[1].split('=')[1])
                R = float(rot[2].split('=')[1])
                
                cam_params.append((X, Y, Z, P, Yaw, R))
        
        return cam_params
    
    
    def load_prompt(self,idx):
        video_id= self.dataset[idx]
        video_id=video_id.replace('\n','')
        prompt_path = os.path.join(self.root_path, 'prompts',f'{video_id}.txt')
        with open(prompt_path, 'r') as f:
            prompt = f.readlines()
        prompt=prompt[0].strip()
        return prompt
    
    def load_image(self,idx):
        video_id= self.dataset[idx]
        video_id=video_id.replace('\n','')
        image_path=video_id.replace('.mp4','.png')

        image = Image.open(image_path)
        return image
    def load_depth(self, idx, batch_idx):
        video_id = self.dataset[idx]
        video_id=video_id.replace('\n','')
        video_name = os.path.basename(video_id)
        video_dir = os.path.dirname(video_id)
        video_base_name = ".".join(video_name.split(".")[:-1])
        depth_base_name = "Mono_" + video_base_name[:-4]
        all_depths = []
        for b in batch_idx:
            depth_path = os.path.join(video_dir, f'{depth_base_name}.{b+1:04d}_depth.exr')
            cur_depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
            # so be it
            # print(f"depth path: {depth_path}")
            all_depths.append(torch.from_numpy(cur_depth).float()[:,:,0])
        return torch.stack(all_depths)
    
    
    def get_batch(self, idx):
        video_reader = self.load_video_reader(idx)
        mask_video_reader = self.load_mask_video_reader(idx)
        cam_params = self.load_cameras(idx, normalize_xyz=self.use_xyz_norm)
        '''
        if self.use_text_prompt:
            video_caption=self.load_prompt(idx)
        else:
            video_caption=""
        '''
        video_caption = self.dataset_prompt[idx]
        assert len(cam_params) >= self.sample_n_frames  
        total_frames = len(cam_params)

        current_sample_stride = self.max_sample_stride  #3

        #长度小于最大的stride的时候，从1取到视频长度的最大stride
        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            maximum_sample_stride=max(1,maximum_sample_stride)
            if maximum_sample_stride<1:  #注意这里目前只能是1，不然有49<98帧的情况
                maximum_sample_stride=1
            current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

        #长度大于最大stride时候，从1-max stride取
        current_sample_stride = random.randint(self.minimum_sample_stride, self.max_sample_stride)
        cropped_length = self.sample_n_frames * current_sample_stride
        if self.first_start:
            start_frame_ind=0
        else:
            start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        if self.shuffle_frames:
            perm = np.random.permutation(self.sample_n_frames)
            frame_indices = frame_indices[perm]
        
        if video_reader is not None:
            pixel_values = torch.from_numpy(video_reader.get_batch(frame_indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.

            pixel_values_mask = torch.from_numpy(mask_video_reader.get_batch(frame_indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values_mask = pixel_values / 255.
        else:
            pixel_values=self.load_image(idx)
            pixel_values = transforms.ToTensor()(pixel_values)  # Convert PIL Image to tensor [3,H,W]
            pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension [1,3,H,W]
            pixel_values = pixel_values.repeat(self.sample_n_frames, 1, 1, 1)  # Repeat to match n_frames
            pixel_values = pixel_values.contiguous()
            pixel_values_mask = torch.ones_like(pixel_values)

        # Get camera parameters for selected frames
        selected_cam_params = [cam_params[i] for i in frame_indices]
        
        # Compute plucker embeddings for each frame
        plucker_embeddings = []
        for X, Y, Z, P, Yaw, R in selected_cam_params:
            plucker = compute_plucker(
                X, Y, Z, P, Yaw, R,
                H=self.sample_size[0],
                W=self.sample_size[1],
                device='cpu'
            )
            plucker_embeddings.append(plucker)
        
        # Stack all plucker embeddings
        plucker_embedding = torch.stack(plucker_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()
        #print('plucker embedding shape:',plucker_embedding.shape)
        
        # Create flip flag (unchanged)
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool, device='cpu')
        

        filter_mask = pixel_values_mask < 0.5
        cond_video = pixel_values.clone()
        cond_video[filter_mask] = 0.

        pixel_values_mask = pixel_values_mask[:,:1,:,:]
        return pixel_values, cond_video, pixel_values_mask, video_caption, plucker_embedding, flip_flag
    def load_video_only(self, idx):
        video_reader = self.load_video_reader(idx)
        cam_params = self.load_cameras(idx, normalize_xyz=self.use_xyz_norm)
        depths = None
        if video_reader is not None:
            batch_index = list(range(len(video_reader)))

            pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.

            selected_cam_params = [torch.tensor(cam_params[i]).float().to(pixel_values.device) for i in batch_index]
            selected_cam_params = torch.stack(selected_cam_params, dim=0)
            if self.load_depth_:
                depths = self.load_depth(idx, batch_index)
        else:
            pixel_values = None
            selected_cam_params = None
        return pixel_values, selected_cam_params, depths

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        
        # cond video: 0~1, [F, C, H, W]
        # prompt: str
        # 
        while True:
            #try:
                # pixel_values, cond_video, pixel_values_mask, video_caption, plucker_embedding, flip_flag
                video, cam, depth = self.load_video_only(idx)
                break

            # except Exception as e:
            #     print('Dataload Error:', e)
            #     idx = random.randint(0, self.length - 1)
        
        data = {
            'videos': video,
            'cams': cam, 
            'depths': depth
        }
        return data

