import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
import pandas as pd

model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    [
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
        ],
    "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
])
# ok...
model_manager.load_lora("models/lightning_logs/version_7/checkpoints/epoch=11-step=756.ckpt", lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

prompt = pd.read_csv('prompt.txt')
prompt =prompt['Prompt'].values
for i in range(len(prompt)):
    if i <0:
        continue   
    else:
        print(prompt[i])
        video = pipe(
            prompt=prompt[i],
            num_frames=81,
            num_inference_steps=50,
            seed=0, tiled=True,
            height=512,
        width=1024,
        )
        save_video(video, f"lora_rank_16_0514/video_citypark_{i}.mp4", fps=24, quality=5)


 # negative_prompt="...",