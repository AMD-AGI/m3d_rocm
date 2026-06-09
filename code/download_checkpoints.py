#Reference https://github.com/SkyworkAI/Matrix-3D/blob/main/code/download_checkpoints.py
import os
# os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download, snapshot_download


def download_ckpt(local_dir, repo_id, filename, repo_type="model"):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(filename))
    if not os.path.exists(local_path):
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            repo_type=repo_type,
        )
        print(f"File has been downloaded to: {file_path}")
    else:
        print(f"File exists already: {local_path}")


def download_snapshot(local_dir, repo_id):
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"Directory exists already: {local_dir}")
        return
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id, local_dir=local_dir)
    print(f"Snapshot downloaded to: {local_dir}")


# ── Matrix-3D / pipeline checkpoints ──────────────────────────────────────────
os.makedirs("./checkpoints", exist_ok=True)
repo_id_list   = ["Ruicheng/moge-vitl","Iceclear/StableSR","Iceclear/StableSR","Skywork/Matrix-3D","Skywork/Matrix-3D","Skywork/Matrix-3D","Skywork/Matrix-3D","Skywork/Matrix-3D"]
filename_list  = ["model.pt","stablesr_turbo.ckpt","vqgan_cfw_00011.ckpt","checkpoints/text2panoimage_lora.safetensors","checkpoints/pano_lrm_480p.pt","checkpoints/pano_video_gen_480p.ckpt","checkpoints/pano_video_gen_720p.bin","checkpoints/pano_video_gen_720p_5b.safetensors"]
local_dir_list = ["./checkpoints/moge","./checkpoints/StableSR","./checkpoints/StableSR","./checkpoints/flux_lora","./checkpoints/pano_lrm","./checkpoints/Wan-AI/wan_lora","./checkpoints/Wan-AI/wan_lora","./checkpoints/Wan-AI/wan_lora"]

for repo_id, filename, local_dir in zip(repo_id_list, filename_list, local_dir_list):
    print(f"\nDownloading {filename} from {repo_id} to local folder {local_dir}...\n")
    download_ckpt(local_dir, repo_id, filename)


# ── OSEDiff checkpoints ────────────────────────────────────────────────────────
OSEDIFF_MODELS = "./code/OSEDiff/preset/models"
os.makedirs(OSEDIFF_MODELS, exist_ok=True)

# Stable Diffusion 2.1 Base (~5GB)
sd_path = os.path.join(OSEDIFF_MODELS, "stable-diffusion-2-1-base")
print(f"\nDownloading SD 2.1 Base -> {sd_path}")
download_snapshot(sd_path, "Manojb/stable-diffusion-2-1-base")

# RAM Swin-Large (~5.3GB) — hosted on HuggingFace Spaces
print(f"\nDownloading RAM Swin-Large -> {OSEDIFF_MODELS}")
download_ckpt(OSEDIFF_MODELS, "xinyu1205/recognize-anything", "ram_swin_large_14m.pth",
              repo_type="space")

# osediff.pkl and DAPE.pth ship with the repo clone (preset/models/) — no download needed
