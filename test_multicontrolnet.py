from diffusers import AutoencoderKL, UniPCMultistepScheduler
import torch
from PIL import Image 
import os 
import json 
import random 
import cv2 
import argparse
import numpy as np 

import sys 
sys.path.append("../")
from src import StableDiffusionControlNetPipeline, ControlNetModel


####################################################################################################################
# Global Variables 
####################################################################################################################
SD_ROOT = r"/path/to/pretrained_models/StableDiffusion"


####################################################################################################################
# Util Functions
####################################################################################################################
def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_image(path, to_size=None):
    # load image 
    image = Image.open(path).convert("RGB")
    # resize 
    if to_size is not None:
        image = image.resize((to_size, to_size))

    return image 


####################################################################################################################
# Test Functions 
####################################################################################################################
### composable test  
def multicontrol_compose_image_test(
        prompt, 
        mask_img_path, 
        bbox_img_path, 
        save_root, 
        mask_controlnet_path, 
        bbox_controlnet_path, 
        mask_deteriorate_ratio=None, 
        bbox_deteriorate_ratio=None, 
        mask_epoch_percentage=None, 
        bbox_epoch_percentage=None, 
    ):
    # save root 
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    # get test images
    mask_image = Image.open(mask_img_path).resize((512, 512), resample=Image.NEAREST)
    bbox_image = Image.open(bbox_img_path).resize((512, 512), resample=Image.NEAREST)
    images = [mask_image, bbox_image]
     
    # 2.5 load model 
    print("==> controlnet branch 1 (default mask): {}".format(mask_controlnet_path))
    print("==> controlnet branch 2 (default bbox): {}".format(bbox_controlnet_path))
    controlnets = [
        ControlNetModel.from_pretrained(
            mask_controlnet_path, torch_dtype=torch.float16, low_cpu_mem_usage=False, # use_safetensors=True
        ),
        ControlNetModel.from_pretrained(
            bbox_controlnet_path, torch_dtype=torch.float16, low_cpu_mem_usage=False, # use_safetensors=True
        ),
    ]
    
    stable_diffusion_path = os.path.join(SD_ROOT, r"stable-diffusion-v1-5")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
            stable_diffusion_path, controlnet=controlnets, torch_dtype=torch.float16, use_safetensors=True
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None 
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(42)  

    # run inference 
    scale_mask = 1.0
    scale_bbox = 1.0 
    print("---- scale_mask={:.02}, scale_bbox={:.02} ----".format(scale_mask, scale_bbox))
    
    if mask_deteriorate_ratio is None:
        mask_deteriorate_ratio = 0.0
    if bbox_deteriorate_ratio is None:
        bbox_deteriorate_ratio = 0.0

    deteriorate_ratio = [
            torch.tensor(mask_deteriorate_ratio, dtype=torch.float32).reshape(-1, 1), 
            torch.tensor(bbox_deteriorate_ratio, dtype=torch.float32).reshape(-1, 1), 
        ]

    # generation for shape-aware controlnet  
    output_image = pipe(
        prompt,
        image=images,
        num_inference_steps=50,
        generator=generator,
        negative_prompt="blurred, distorted",
        num_images_per_prompt=8,  # 100
        controlnet_conditioning_scale=[scale_mask, scale_bbox],
        do_ratio_condition=[True, True], 
        do_predict=[True, True], 
        deteriorate_ratio=deteriorate_ratio,  
        epoch_percentage=[mask_epoch_percentage, bbox_epoch_percentage], 
    ).images
                
    for idx, aimg in enumerate(output_image):
        save_path = os.path.join(save_root, "demo_id_{}.png".format(idx))
        aimg.save(save_path)


def arg_parser():
    # arg parser 
    parser = argparse.ArgumentParser(description='Inference with shape-aware controlnet.')  
    # input settings 
    # prompt 
    parser.add_argument('--prompt', type=str, default="", 
                        help="the prompt or caption for the input image.")
    # img path 
    parser.add_argument('--mask_img_path', type=str, default=None, 
                        help="the explicit mask for inference.")
    parser.add_argument('--bbox_img_path', type=str, default=None, 
                        help="the inexplicit mask like bbox for inference.")
    # checkpoint path 
    parser.add_argument('--mask_controlnet_path', type=str, default=None, 
                        help="the checkpoint for the mask image branch.")
    parser.add_argument('--bbox_controlnet_path', type=str, default=None, 
                        help="the checkpoint for the bbox image branch.")
    # deteriorate ratio 
    parser.add_argument('--mask_deteriorate_ratio', type=float, default=None, 
                        help="set the deteriorat_ratio (0 to 1) for the input mask.")
    parser.add_argument('--bbox_deteriorate_ratio', type=float, default=None, 
                        help="set the deteriorat_ratio (0 to 1) for the input bbox.")
    # output path 
    parser.add_argument('--output_path', type=str, default=None, 
                        help="the output path to save the result.")

    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    ### get args 
    args = arg_parser()

    # preprocess
    mask_epoch_percentage = 1.0 if args.mask_deteriorate_ratio is None else None
    bbox_epoch_percentage = 1.0 if args.bbox_deteriorate_ratio is None else None

    # main function 
    multicontrol_compose_image_test(
        prompt=args.prompt, 
        mask_img_path=args.mask_img_path, 
        bbox_img_path=args.bbox_img_path, 
        save_root=args.output_path, 
        mask_controlnet_path=args.mask_controlnet_path, 
        bbox_controlnet_path=args.bbox_controlnet_path, 
        mask_deteriorate_ratio=args.mask_deteriorate_ratio, 
        bbox_deteriorate_ratio=args.bbox_deteriorate_ratio, 
        mask_epoch_percentage=mask_epoch_percentage, 
        bbox_epoch_percentage=bbox_epoch_percentage, 
    )
    
    print("==> Congratulations!")



