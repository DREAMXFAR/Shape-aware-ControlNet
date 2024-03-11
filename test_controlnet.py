from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import os
import numpy as np
from PIL import Image 
import json 
import cv2 
import argparse

from src import StableDiffusionControlNetPipeline, ControlNetModel


##################################################################################################
# Global Variables 
##################################################################################################
SD_ROOT = r"/path/to/pretrained_models/StableDiffusion/stable-diffusion-v1-5"


##################################################################################################
# Utils Functions  
##################################################################################################
def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def dilate(image, r:int=5):
    if r == 0:
        return image
    else:
        # 5x5 elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r)) 
        image = np.array(image * 255, dtype=np.uint8)
        dst = cv2.dilate(image, kernel=kernel) 
        return (dst / 255.0).astype(np.float32)


##################################################################################################
# Inference Functions  
##################################################################################################
def controlnet_infer(
        base_model_path, controlnet_path, control_image_path, prompt, 
        output_root, save_name, seed=555, 
        control_guidance_start=0.0,   
        control_guidance_end=1.0, 
        do_ratio_condition=False, 
        do_predict=False, 
        deteriorate_ratio=0, 
        epoch_percentage=None, 
    ):
    print("==> model_path: {}".format(controlnet_path))

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)  # , low_cpu_mem_usage=False)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.safety_checker = None

    orig_control_image = load_image(control_image_path)
    image_shape = orig_control_image.size 
    control_image = orig_control_image.resize((512, 512), Image.BILINEAR)
    
    if deteriorate_ratio is None and epoch_percentage==1:
        deteriorate_ratio = 0.0
    deteriorate_ratio = torch.tensor(deteriorate_ratio, dtype=torch.float32).reshape(-1, 1)
    
    print("==> prompt: {}".format(prompt))

    # generate image
    print("==> seed: {}".format(seed))
    generator = torch.manual_seed(seed)

    output_image = pipe(
            prompt, 
            num_inference_steps=50, 
            generator=generator, 
            image=control_image, 
            num_images_per_prompt=4, 
            control_guidance_start=control_guidance_start, 
            control_guidance_end=control_guidance_end, 
            do_ratio_condition=do_ratio_condition, 
            do_predict=do_predict, 
            deteriorate_ratio=deteriorate_ratio,
            epoch_percentage=epoch_percentage, 
        ).images
    
    if not os.path.exists(output_root):
        os.mkdir(output_root)

    output_image = [aimg.resize(image_shape, Image.BILINEAR) for aimg in output_image]
    output_image.insert(0, orig_control_image)

    mix_image = image_grid(output_image, rows=1, cols=5)
    save_path = os.path.join(output_root, save_name)
    print("==> output is saved in: {}".format(save_path))
    mix_image.save(save_path)    


def inference_single(
        control_image_path, 
        prompt, 
        controlnet_path, 
        output_path, 
        deteriorate_ratio=None, 
        epoch_percentage=1.0,
        seed=555, 
    ):
    # preprocess 
    test_filename = os.path.basename(control_image_path)
    save_name = "{}_pred_d{}.png".format(test_filename.split(".")[0], dilate_ratio)

    # run inference
    controlnet_infer(
            SD_ROOT, 
            controlnet_path,
            control_image_path,
            prompt,
            output_root=output_path, 
            save_name=save_name,
            seed=seed, 
            do_predict=True, 
            do_ratio_condition=True, 
            deteriorate_ratio=deteriorate_ratio, 
            epoch_percentage=epoch_percentage, 
        )


def arg_parser():
    # arg parser 
    parser = argparse.ArgumentParser(description='Inference with shape-aware controlnet.')  
    # input settings 
    parser.add_argument('--img_path', type=str, default=None, 
                        help="the segmentation map for inference.")
    parser.add_argument('--prompt', type=str, default="", 
                        help="the prompt or caption for the input image.")
    parser.add_argument('--controlnet_path', type=str, default=None, 
                        help="the checkpoint path of the model.")
    parser.add_argument('--deteriorate_ratio', type=float, default=None, 
                        help="set the deteriorate_ratio (0 to 1) of the input mask, if None, use the ratio predicted by the ratio predictor network. Note that the deteriorate_ratio does not equals dilate_radius.")
    parser.add_argument('--output_path', type=str, default=None, 
                        help="the output path to save the result.")

    args = parser.parse_args()
    return args 


if __name__ == "__main__":

    ### arg parser 
    args = arg_parser()
    
    # preprocess 
    if args.deteriorate_ratio is None:
        # use the dilated ratio predicted by the model.
        epoch_percentage = 1.0 
    else:
        # use the user-provided dilate_ratio for inference.
        epoch_percentage = None 
    
    # set model path
    os.makedirs(args.output_path, exist_ok=True)
    
    ### inference with shape-aware controlnet 
    inference_single(
        control_image_path=args.img_path, 
        prompt=args.prompt, 
        deteriorate_ratio=args.deteriorate_ratio, 
        controlnet_path=args.controlnet_path, 
        output_path=args.output_path, 
        epoch_percentage=epoch_percentage, 
    )
        
    print(":) Congratulations!")
