# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Enhance images in mv_rgb_orig using OSEDiff and save to mv_rgb.

Usage:
    python code/enhance_with_osediff.py --inout_dir output/case1
    python code/enhance_with_osediff.py --inout_dir output/case1 --input_dir custom/input --output_dir custom/output
"""

import os
import sys
import glob
import argparse
import time
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OSEDiff'))

from osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix
from ram.models.ram_lora import ram
from ram import inference_ram as inference


OSEDIFF_DIR = os.path.join(os.path.dirname(__file__), 'OSEDiff')

DEFAULT_SD_PATH    = os.path.join(OSEDIFF_DIR, 'preset/models/stable-diffusion-2-1-base')
DEFAULT_OSEDIFF    = os.path.join(OSEDIFF_DIR, 'preset/models/osediff.pkl')
DEFAULT_RAM_PATH   = os.path.join(OSEDIFF_DIR, 'preset/models/ram_swin_large_14m.pth')
DEFAULT_RAM_FT     = os.path.join(OSEDIFF_DIR, 'preset/models/DAPE.pth')


def build_args(sd_path, osediff_path, ram_path, ram_ft_path):
    import types
    return types.SimpleNamespace(
        pretrained_model_name_or_path=sd_path,
        osediff_path=osediff_path,
        ram_path=ram_path,
        ram_ft_path=ram_ft_path,
        process_size=512,
        upscale=1,
        align_method='adain',
        mixed_precision='fp16',
        merge_and_unload_lora=False,
        vae_decoder_tiled_size=224,
        vae_encoder_tiled_size=1024,
        latent_tiled_size=96,
        latent_tiled_overlap=32,
        prompt='',
        seed=42,
    )


def load_models(args):
    print('Loading OSEDiff...')
    t0 = time.time()
    model = OSEDiff_test(args)
    print(f'  done ({time.time()-t0:.1f}s)')

    print('Loading RAM/DAPE...')
    t0 = time.time()
    DAPE = ram(pretrained=args.ram_path, pretrained_condition=args.ram_ft_path,
               image_size=384, vit='swin_l')
    DAPE.eval()
    DAPE.to('cuda')
    DAPE = DAPE.to(dtype=torch.float16)
    print(f'  done ({time.time()-t0:.1f}s)')

    return model, DAPE


def warmup(model, weight_dtype):
    print('Warming up GPU kernels...')
    dummy = torch.zeros(1, 3, 512, 512, device='cuda', dtype=weight_dtype)
    with torch.no_grad():
        model(dummy * 2 - 1, prompt='warmup')
    torch.cuda.synchronize()
    print('  done')


def enhance_images(input_dir, output_dir, model, DAPE, weight_dtype, output_size=1024):
    tensor_transforms = transforms.Compose([transforms.ToTensor()])
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')) +
                         glob.glob(os.path.join(input_dir, '*.jpg')))
    if not image_paths:
        print(f'No images found in {input_dir}')
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f'Enhancing {len(image_paths)} images: {input_dir} -> {output_dir} (output {output_size}px)')

    for i, img_path in enumerate(image_paths):
        t0 = time.time()
        bname = os.path.basename(img_path)

        # Load and resize to 512px (fed directly to diffusion model, no upscaling)
        input_image = Image.open(img_path).convert('RGB').resize((512, 512), Image.LANCZOS)

        # Generate caption via RAM
        lq = tensor_transforms(input_image).unsqueeze(0).to('cuda')
        lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
        captions = inference(lq_ram, DAPE)
        prompt = f'{captions[0]}, ,'

        # Run OSEDiff
        with torch.no_grad():
            output_image = model(lq * 2 - 1, prompt=prompt)

        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = adain_color_fix(target=output_pil, source=input_image)
        output_pil = output_pil.resize((output_size, output_size), Image.LANCZOS)

        out_path = os.path.join(output_dir, bname)
        output_pil.save(out_path)
        print(f'  [{i+1}/{len(image_paths)}] {bname} ({time.time()-t0:.2f}s) | {prompt[:60]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inout_dir', type=str, default=None,
                        help='Pipeline output dir (sets input/output to geom_optim/data/mv_rgb_orig and mv_rgb)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Override: explicit input directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override: explicit output directory')
    parser.add_argument('--sd_path', type=str, default=DEFAULT_SD_PATH)
    parser.add_argument('--osediff_path', type=str, default=DEFAULT_OSEDIFF)
    parser.add_argument('--ram_path', type=str, default=DEFAULT_RAM_PATH)
    parser.add_argument('--ram_ft_path', type=str, default=DEFAULT_RAM_FT)
    parser.add_argument('--output_size', type=int, default=1024,
                        help='Output image size in pixels (default: 1024)')
    args = parser.parse_args()

    if args.inout_dir:
        input_dir  = os.path.join(args.inout_dir, 'geom_optim/data/mv_rgb_orig')
        output_dir = os.path.join(args.inout_dir, 'geom_optim/data/mv_rgb')
    elif args.input_dir and args.output_dir:
        input_dir  = args.input_dir
        output_dir = args.output_dir
    else:
        parser.error('Provide either --inout_dir or both --input_dir and --output_dir')

    model_args = build_args(args.sd_path, args.osediff_path, args.ram_path, args.ram_ft_path)

    model, DAPE = load_models(model_args)
    warmup(model, model_args.mixed_precision == 'fp16' and torch.float16 or torch.float32)
    enhance_images(input_dir, output_dir, model, DAPE,
                   torch.float16 if model_args.mixed_precision == 'fp16' else torch.float32,
                   output_size=args.output_size)

    print('Done.')


if __name__ == '__main__':
    main()
