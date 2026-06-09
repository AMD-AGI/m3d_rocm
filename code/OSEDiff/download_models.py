import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download pretrained models for OSEDiff")
    parser.add_argument("--model_dir", type=str, default="preset/models", help="Directory to save models")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    from huggingface_hub import snapshot_download, hf_hub_download

    # SD 2.1 Base
    sd_path = os.path.join(args.model_dir, "stable-diffusion-2-1-base")
    if os.path.isdir(sd_path) and os.path.isdir(os.path.join(sd_path, "unet")):
        print(f"[skip] SD 2.1 Base already exists at {sd_path}")
    else:
        print(f"[downloading] SD 2.1 Base -> {sd_path} (~5GB)")
        snapshot_download("Manojb/stable-diffusion-2-1-base", local_dir=sd_path)
        print(f"[done] SD 2.1 Base")

    # RAM Swin-Large
    ram_path = os.path.join(args.model_dir, "ram_swin_large_14m.pth")
    if os.path.isfile(ram_path):
        print(f"[skip] RAM already exists at {ram_path}")
    else:
        print(f"[downloading] RAM Swin-Large -> {ram_path} (~5.3GB)")
        hf_hub_download("xinyu1205/recognize-anything", "ram_swin_large_14m.pth",
                        repo_type="space", local_dir=args.model_dir)
        print(f"[done] RAM Swin-Large")

    # DAPE
    dape_path = os.path.join(args.model_dir, "DAPE.pth")
    if os.path.isfile(dape_path):
        print(f"[skip] DAPE already exists at {dape_path}")
    else:
        print(f"[warning] DAPE not found at {dape_path}")
        print(f"  Download manually from: https://drive.google.com/file/d/1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm/view")

    # OSEDiff weights
    for name in ["osediff.pkl", "osediff_face.pkl"]:
        path = os.path.join(args.model_dir, name)
        if os.path.isfile(path):
            print(f"[skip] {name} already exists at {path}")
        else:
            print(f"[warning] {name} not found at {path}")

    print("\nAll models ready. You can now run inference:")
    print(f"  python test_osediff.py \\")
    print(f"    -i preset/datasets/test_dataset/input \\")
    print(f"    -o preset/datasets/test_dataset/output \\")
    print(f"    --osediff_path {os.path.join(args.model_dir, 'osediff.pkl')} \\")
    print(f"    --pretrained_model_name_or_path {sd_path} \\")
    print(f"    --ram_ft_path {dape_path} \\")
    print(f"    --ram_path {ram_path}")


if __name__ == "__main__":
    main()
