import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipelineNew, ModelConfig
from modelscope import dataset_snapshot_download
from peft import LoraConfig, inject_adapter_in_model
from diffsynth import load_state_dict
from diffsynth.trainers.utils import VideoDataset
import cv2
import numpy as np
import os
def write_video(frames, out_path, fps = 24):
    width, height = frames[0].shape[1], frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
def add_lora_to_model(model, target_modules, lora_rank, lora_alpha=None):
    if lora_alpha is None:
        lora_alpha = lora_rank
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
    model = inject_adapter_in_model(lora_config, model)
    return model
def mapping_lora_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A.weight" in key or "lora_B.weight" in key:
            new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
            new_state_dict[new_key] = value
    return new_state_dict
pipe = WanVideoPipelineNew.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)
#print(f"pipe check: {pipe.dit.device}")
model = add_lora_to_model(
    getattr(pipe, "dit"),
    "q,k,v,o,ffn.0,ffn.2".split(","),
    48,
    48.0
)
# die eisenfaust 
# 81 or 121,which is better trained?
# will need to regenerate data if wish to be trained on 121 though.
# 如果不塞第一帧的话，这个模型能做的事情就是按左右摇摆的玩意晃一晃镜头。
# the tracking video does seem to be injected; however not fully.
# 神人网络，气笑了
# 这种loss曲线怎么能work的
# this model does seem to work...
# but doesnt work well.
# so i would say the training may be correct.
# but why the loss keeps shaking?
# 挂一个批量的测试在这儿
#lora_checkpoint = "/datasets_3d/zhongqi.yang/matrix3d_inference/DiffSynth-Studio-tmp/models/train/Wan2.2-TI2V-5B_lora-test-0823-lowres/step-2800.safetensors"
#lora_checkpoint = "/datasets_3d/zhongqi.yang/matrix3d_inference/DiffSynth-Studio-tmp/models/train/Wan2.2-TI2V-5B_lora-test-0825-lora-h704-5k/step-6200.safetensors"
lora_checkpoint = "/datasets_3d/zhongqi.yang/matrix3d_inference/DiffSynth-Studio-tmp/models/train/Wan2.2-TI2V-5B_lora-test-0825-lora-h704-100k-mm/step-9000.safetensors"
output_dir = "/datasets_3d/zhongqi.yang/git_test/Matrix-3D/code/DiffSynth-Studio/output"
os.makedirs(output_dir,exist_ok=True)
#lora_checkpoint = "/datasets_3d/zhongqi.yang/matrix3d_inference/DiffSynth-Studio-tmp/models/train/Wan2.2-TI2V-5B_lora-test-0823-lowres-random-init/step-2600.safetensors"
state_dict = load_state_dict(lora_checkpoint)
print(state_dict['patch_embedding_extra_concate.weight'].max(),state_dict['patch_embedding_extra_concate.weight'].min())
#state_dict = mapping_lora_state_dict(state_dict)
print(state_dict.keys())
load_result = model.load_state_dict(state_dict, strict=False)
if len(load_result[1]) > 0:
    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
else:
    print("load fine.")
# pipe.dit.load_state_dict()
for n, p in pipe.dit.patch_embedding_extra_concate.named_parameters():
    print(f"name: {n} parameters: {p.max()} {p.min()}")

for n, p in pipe.dit.named_parameters():
    if "lora" in n:
        print(f"name lora: {n} parameters: {p.max()} {p.min()}")

for n, p in model.patch_embedding_extra_concate.named_parameters():
    print(f"name: {n} parameters: {p.max()} {p.min()}")
pipe.enable_vram_management()
# patchify 的路子看起来不太对？
# 不过本身嗷，合成数据这一块，das原版就没管过
# 
height = 704
width = height * 2
vid_dset = VideoDataset(
    #base_path="/", metadata_path="/datasets_3d/zhongqi.yang/matrix3d_inference/dataset/metadata_1k.csv",
    base_path="/", metadata_path="/datasets_3d/zhongqi.yang/matrix3d_inference/DiffSynth-Studio-tmp/data/examples/wan/tmp.csv",
    num_frames=81,
    time_division_factor=4, time_division_remainder=1,
    max_pixels=height*width, height=height, width=width,
    height_division_factor=16, width_division_factor=16,
    data_file_keys=("video","cond_video","cond_mask"),
    image_file_extension=("jpg", "jpeg", "png", "webp"),
    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
    repeat=1,
    args=None,
)
chunk = vid_dset[43]
#print((chunk["cond_video"]).shape)
tmp = [np.array(i) for i in chunk["cond_video"]]
print(tmp[0].dtype,tmp[0].max())
write_video(tmp,os.path.join(output_dir,"cond.mp4"),24)
# at least you see things work.
# get the training back online before off work.
# gonna take a long time to train.
# still, there is hope that something may go good due to wan2.2 things.
# after we switch the network structure to wan 2.2 the speed issue may go with the wind.
# with some lower lr this may be solved.
video = pipe(
    prompt=chunk['prompt'],
    negative_prompt="The video is not of a high quality, it has a low resolution. Distortion. strange artifacts. flickering. worst quality. low quality",
    seed=120, tiled=True,
    height=height, width=width,
    input_image=chunk["video"][0],
    num_frames=81,
    cond_video = (chunk["cond_video"]),
    cond_mask = (chunk["cond_mask"]),
)

save_video(video, os.path.join(output_dir,"gen.mp4"), fps=15, quality=5)


concate = [np.concatenate([np.array(video[i]),tmp[i]],axis=1)[:,:,::-1]for i in range(81)]
write_video(concate,os.path.join(output_dir,"gen_cond_concate.mp4"),24)
concate1 = [np.concatenate([np.array(video[i]),np.array(video[i])],axis=1)[:,:,::-1]for i in range(81)]
write_video(concate1,os.path.join(output_dir,"gen_gen_concate.mp4"),24)
