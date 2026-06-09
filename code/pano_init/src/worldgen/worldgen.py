import torch 
import numpy as np
import cv2
from PIL import Image,ImageOps
import open3d as o3d
from .pano_seg import build_segment_model, seg_pano_fg
from .pano_gen import build_pano_gen_model, gen_pano_image, build_pano_fill_model, gen_pano_fill_image
from .pano_inpaint import build_inpaint_model, inpaint_image
from .utils.splat_utils import convert_rgbd_to_gs, SplatFile, mask_splat, merge_splats
from .utils.general_utils import map_image_to_pano, resize_img, depth_match, convert_rgbd2mesh_panorama
from typing import Optional, Union


class WorldGen:
    def __init__(self, 
            mode: str = 't2s',
            inpaint_bg: bool = False,
            lora_path: str = None,
            resolution: int = 1600,
            device: torch.device = 'cuda',
        ):
        self.device = device

        self.mode = mode
        self.resolution = resolution

        if mode == 't2s':
            self.pano_gen_model = build_pano_gen_model(lora_path=lora_path, device=device)
        elif mode == 'i2s':
            self.pano_gen_model = build_pano_fill_model(lora_path=lora_path, device=device)
        else:
            raise ValueError(f"Invalid mode: {mode}, mode must be 'i2p' or 't2p'")

        self.inpaint_bg = inpaint_bg
        if inpaint_bg:
            self.seg_processor, self.seg_model = build_segment_model(device)
            self.inpaint_pipe = build_inpaint_model(device)

    def depth2gs(self, predictions) -> SplatFile:
        rgb = predictions["rgb"]
        distance = predictions["distance"]
        rays = predictions["rays"]
        splat = convert_rgbd_to_gs(rgb, distance, rays)
        return splat
    
    def depth2mesh(self, predictions) -> o3d.geometry.TriangleMesh:
        rgb = predictions["rgb"] / 255.0
        distance = predictions["distance"]
        rays = predictions["rays"]
        mesh = convert_rgbd2mesh_panorama(rgb, distance, rays)
        return mesh
    

    

    def i2p_init(self, image, FOV, Theta, Phi,height, width,h_Fov=None, debug=False):
        image = np.array(image)
        #得到有效区域
        wFov = FOV
        img_h, img_w = image.shape[:2]
        if h_Fov is not None:
            hFov = h_Fov
        else:
            hFov =float(img_h)/img_w*wFov
        w_len = np.tan(np.radians(wFov/2.0))
        h_len = np.tan(np.radians(hFov/2.0))
        #该模型x朝前，z朝上
        x,y = np.meshgrid(np.linspace(-180,180,width), np.linspace(90,-90,height))

        x_map = np.cos(np.radians(x))*np.cos(np.radians(y))
        y_map = np.sin(np.radians(x))*np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map, y_map, z_map),axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)

        R1,_ = cv2.Rodrigues(z_axis*np.radians(Theta))
        R2,_ = cv2.Rodrigues(np.dot(R1, y_axis)*np.radians(-Phi))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T
        xyz = xyz.reshape([height, width, 3])
        inverse_mask = np.where(xyz[:,:,0]>0,1,0)
        #归一化到x=1
        xyz[:,:]= xyz[:,:]/np.repeat(xyz[:,:,0][:,:,np.newaxis],3,axis=2)#xyz[:,:,0]为height width
        


        lon_map = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
            &(xyz[:,:,2]<h_len),(xyz[:,:,1]+w_len)/2/w_len*img_w,0)
        lat_map = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
            &(xyz[:,:,2]<h_len),(-xyz[:,:,2]+h_len)/2/h_len*img_h,0)

        mask = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
            &(xyz[:,:,2]<h_len),1,0)

        #将当前图像映射到全景图上
        pano = cv2.remap(image.astype(np.float32), 
                        lon_map.astype(np.float32), 
                        lat_map.astype(np.float32),
                        cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_WRAP)

        if np.max(pano)< 180:
            pano = np.clip(pano, 0,1)
        # persp = (persp - persp.min())/(persp.max()-persp.min())
        else:
            pano=np.clip(pano, 0, 255)
        mask = mask*inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        pano = pano*mask
        pano = Image.fromarray(pano.astype(np.uint8))#转会Image类型
        
        mask = Image.fromarray((mask*255).astype(np.uint8))

        if debug:
            pano.save("/ai-video-sh/haoyuan.li/AIGC/WorldGen/debug_img/pano_init.jpg")
            mask.save("/ai-video-sh/haoyuan.li/AIGC/WorldGen/debug_img/mask_init.jpg")
        return pano, mask

    def blend_pano_edges(self, image, blend_width=50):
        """混合全景图左右边缘"""
        img_array = np.array(image)
        left_edge = img_array[:, :blend_width]
        right_edge = img_array[:, -blend_width:]
        
        # 线性渐变混合
        for i in range(blend_width):
            alpha = i / blend_width
            img_array[:, i] = alpha * right_edge[:, -blend_width + i] + (1 - alpha) * left_edge[:, i]
            img_array[:, -blend_width + i] = img_array[:, i]
        
        return Image.fromarray(img_array)
    
    def generate(self, img_path, prompt, w_fov, h_fov,seed=42):
        image = Image.open(img_path)
        image, mask = self.i2p_init(image,w_fov,0,0,512,1024,h_fov,debug=False)
        mask = np.array(mask)
        kernel = np.ones((3,3),dtype=np.uint8)
        mask = cv2.erode(mask, kernel,iterations=3)
        mask = Image.fromarray(mask)
        mask = ImageOps.invert(mask)

        pano_image = gen_pano_fill_image(
        self.pano_gen_model, 
        image=image, 
        mask= mask,
        prompt=prompt, 
        height=self.resolution//2, 
        width=self.resolution,
        seed=seed)
        return pano_image

    def inpaint_img(self, prompt, image, mask,seed=42):
        #?使用自定义mask
        # image, mask = self.i2p_init(img,90,0,30,512,1024)
        # 
        mask = np.array(mask)
        kernel = np.ones((3,3),dtype=np.uint8)
        mask = cv2.erode(mask, kernel,iterations=3)
        # mask =np.where(mask>128,255,0).astype(np.uint8)
        mask = Image.fromarray(mask)
        # mask.save("/ai-video-sh/haoyuan.li/AIGC/WorldGen/output/mask/mask.jpg")

        
        mask = ImageOps.invert(mask)
        pano_image = gen_pano_fill_image(
                self.pano_gen_model, 
                image=image, 
                mask= mask,
                prompt=prompt, 
                height=self.resolution//2, 
                width=self.resolution,
                seed=seed
            )

        return pano_image
    
    
    @torch.inference_mode()
    def generate_world(
        self, 
        prompt: str = "", 
        image: Optional[Image.Image] = None, 
        return_mesh: bool = False
    ) -> Union[SplatFile, o3d.geometry.TriangleMesh]:
        pano_image = self.generate_pano(prompt, image)
        return pano_image#?修改代码逻辑，直接返回pano_img
        pano_image.save("/ai-video-sh/haoyuan.li/AIGC/WorldGen/output/output.jpg")
        scene = self._generate_world(pano_image, return_mesh)
        return scene