import json
import cv2
import numpy as np
# from torch.utils.data import Dataset
import os
import datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import matplotlib.pyplot as plt
from PIL import Image
import pycocotools.mask as mask_utils

from rdp import rdp
import jsonlines
from tqdm import tqdm

import sys
sys.path.append(r"/dat03/xuanwenjie/code/controlnet/custom_datasets")
from lvis_coco import LVIS_Dataset

import ipdb 


#####################################################################################################################
# Utils  
#####################################################################################################################
def write_jsonl(save_path, item):
    with jsonlines.open(save_path, mode = 'a') as json_writer:
        json_writer.write(item)


def mask2bbox(anns, height, width):
    bbox_mask = np.zeros((height, width))
    for cur_ann in anns:
        cur_bbox = cur_ann["bbox"]
        cur_cat = cur_ann["category_id"]
        x = int(cur_bbox[0]) 
        y = int(cur_bbox[1])
        w = int(cur_bbox[2])
        h = int(cur_bbox[3])
        cv2.fillPoly(bbox_mask, [np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])], color=1)
    
    return bbox_mask.astype(np.float32)


def dilate(image, r=5):
    if r == 0:
        return image
    else:
        # 5x5 elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r)) 
        image = np.array(image * 255, dtype=np.uint8)
        dst = cv2.dilate(image, kernel=kernel) 
        return (dst / 255.0).astype(np.float32)


def add_caption2img(image, caption):
    """
    image: ndarray, RGB 
    caption: str
    """
    font_scale = 0.35

    image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    background = image.copy()
    width_per_word = 6
    rec_width = len(caption) * width_per_word
    rec_height = 15
    
    image_height = image.shape[0]
    image_width = image.shape[1]

    if rec_width > image_width:
        max_words = int((image_width / width_per_word) - 2)
        n_lines = int(len(caption) / max_words)
        cv2.rectangle(background, (0, 0), (image_width, (n_lines+1)*rec_height), (0, 0, 0), -1)
        mixed_image = cv2.addWeighted(image, 0.3, background, 0.7, 0, dtype = cv2.CV_32F)
        for n in range(n_lines):
            cv2.putText(mixed_image, caption[n*max_words:(n+1)*max_words], 
                        (5, (n+1)*rec_height-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        # write the last sentences
        cv2.putText(mixed_image, caption[n_lines*max_words:], 
                        (5, (n_lines+1)*rec_height-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    else:
        cv2.rectangle(background, (0, 0), (rec_width, rec_height), (0, 0, 0), -1)
        mixed_image = cv2.addWeighted(image, 0.3, background, 0.7, 0, dtype = cv2.CV_32F)
        cv2.putText(mixed_image, caption, (5, rec_height-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return mixed_image


#####################################################################################################################
# Functions   
#####################################################################################################################
def save_sketch_images(args, dataset, save_anno_img=False, debug=False):
    """
    additional saving for anno-images (raw&mixed) compared with func(save_train_sketch_images)
    """
    epsilon = args.rdp_epsilon 
    
    SAVE_ROOT = "/dat03/xuanwenjie/code/controlnet/output_dir/vis_sketch_image"
    save_sketch_root = os.path.join(SAVE_ROOT, r"epsilon_{}".format(epsilon))
    
    save_image_root = os.path.join(SAVE_ROOT, r"image")
    save_mixed_root = os.path.join(SAVE_ROOT, r"mixed_epsilon_{}".format(epsilon))

    save_prompt_path = os.path.join(SAVE_ROOT, r"captions.jsonl")
    if os.path.exists(save_prompt_path):
        os.remove(save_prompt_path)
    
    ### save sketch 
    if not os.path.exists(save_sketch_root):
        os.mkdir(save_sketch_root)
    ### save raw image and mixed ones 
    for apath in [save_image_root, save_mixed_root]:
        if save_anno_img and not os.path.exists(apath):
            os.mkdir(apath)
    
    dataset_len = len(dataset)

    pbar = tqdm(range(dataset_len))
    for idx in pbar:
        aitem = dataset[idx]
        filename = aitem["file_name"]
        image = aitem["pixel_values"]
        condition = aitem["conditioning_pixel_values"]
        caption = aitem["input_ids"][0]
        
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # condition = np.array(condition) * 255
        condition = np.array(condition)
        cur_save_path = os.path.join(save_sketch_root, filename.replace("jpg", "png"))
        condition = cv2.cvtColor(condition, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_save_path, condition)
        # log 
        pbar.set_description(cur_save_path)

        if save_anno_img:
            font_scale = 0.35
            ### save original image 
            cur_save_path = os.path.join(save_image_root, filename) 
            if os.path.exists(cur_save_path):
                pass
            else:
                cv2.imwrite(cur_save_path, image)   
            ### save mixed image 
            cur_mixed_path = os.path.join(save_mixed_root, filename)
            alpha = 0.5
            mixed_image = cv2.addWeighted(image, alpha, condition, 1-alpha, 0, dtype = cv2.CV_32F)
            
            background = mixed_image.copy()
            width_per_word = 6
            rec_width = len(caption) * width_per_word
            rec_height = 15
            
            image_height = image.shape[0]
            image_width = image.shape[1]

            if rec_width > image_width:
                max_words = int((image_width / width_per_word) - 2)
                n_lines = int(len(caption) / max_words)
                cv2.rectangle(background, (0, 0), (image_width, (n_lines+1)*rec_height), (0, 0, 0), -1)
                mixed_image = cv2.addWeighted(mixed_image, 0.3, background, 0.7, 0, dtype = cv2.CV_32F)
                for n in range(n_lines):
                    cv2.putText(mixed_image, caption[n*max_words:(n+1)*max_words], 
                                (5, (n+1)*rec_height-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                # write the last sentences
                cv2.putText(mixed_image, caption[n_lines*max_words:], 
                                (5, (n_lines+1)*rec_height-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            else:
                cv2.rectangle(background, (0, 0), (rec_width, rec_height), (0, 0, 0), -1)
                mixed_image = cv2.addWeighted(mixed_image, 0.3, background, 0.7, 0, dtype = cv2.CV_32F)
                cv2.putText(mixed_image, caption, (5, rec_height-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            # save image
            cv2.imwrite(cur_mixed_path, mixed_image)

        cur_line = {
            "filename": filename, 
            "caption": caption, 
        }
        write_jsonl(save_prompt_path, cur_line)

        if debug and idx >= 10:
            break


def save_train_sketch_images(args, dataset, debug=False):
    print("==> images are saved in .png by default. ")
    epsilon = args.rdp_epsilon 
    
    SAVE_ROOT = r"/dat03/xuanwenjie/datasets/LVIS_COCO_triplet/condition_train"
    save_sketch_root = os.path.join(SAVE_ROOT, r"epsilon_{}".format(epsilon))
    if not os.path.exists(save_sketch_root):
        os.mkdir(save_sketch_root)
    
    dataset_len = len(dataset)

    pbar = tqdm(range(dataset_len))
    for idx in pbar:
        aitem = dataset[idx]
        filename = aitem["file_name"]
        image = aitem["pixel_values"]
        condition = aitem["conditioning_pixel_values"]
        caption = aitem["input_ids"][0]
        
        # condition = np.array(condition) * 255
        condition = np.array(condition)
        cur_save_path = os.path.join(save_sketch_root, filename.replace("jpg", "png"))
        condition = cv2.cvtColor(condition, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_save_path, condition)

        pbar.set_description(cur_save_path)
        if debug and idx >= 10:
            break


def save_sketch_object_name_json(args, dataset):
    epsilon = args.rdp_epsilon 
    dataset_len = len(dataset)

    validation_json = {}
    test_name_list = ["000000286994.jpg", "000000403385.jpg", "000000153299.jpg"]

    pbar = tqdm(range(dataset_len))
    for idx in pbar:
        aitem = dataset[idx]
        filename = aitem["file_name"]
        image = aitem["pixel_values"]
        condition = aitem["conditioning_pixel_values"]
        caption = aitem["input_ids"][0]
        object_name = aitem["object_name"]
        center_pos = aitem["center_pos"]

        if not (filename in test_name_list):
            continue 

        ### pre-process
        w, h = image.size
        center_pos[:, 0] = center_pos[:, 0] / w 
        center_pos[:, 1] = center_pos[:, 1] / h 
        
        max_objs = 10
        if len(object_name) > 30:
            object_name = object_name[:max_objs]
            center_pos = center_pos[:max_objs, :]

        _center_pos = []
        for i in range(len(object_name)):
            _center_pos.append([center_pos[i, 0], center_pos[i, 1]])
        center_pos = _center_pos

        validation_json[filename] = { 
            "caption": caption,
            "object_name": object_name, 
            "center_pos": center_pos, 
        }

        # write_jsonl(save_prompt_path, cur_line)

        # if idx >= 10:
        #     break

    save_prompt_path = r"/mnt/data/xuanwenjie/project/diffusers/examples/controlnet/output_dir/vis_sketch_image/captions_with_10_objname_pos.json"
    with open(save_prompt_path, "w") as f:
        json.dump(validation_json, f, indent=4)


def save_dilate_sketch_images(args, dataset, mode=None, debug=False):
    """
    save dilate sketch images with 20,40... radius 
    """
    dilate_radius = dataset.dilate_radius 
    
    SAVE_ROOT = r"/dat03/xuanwenjie/datasets/LVIS_COCO_triplet"

    if mode == "val":
        save_root = os.path.join(SAVE_ROOT, r"condition_val")
    elif mode == "train": 
        save_root = os.path.join(SAVE_ROOT, r"condition_train")
    else:
        # save_root = r"/mnt/data/xuanwenjie/project/diffusers/examples/controlnet/output_dir/debug/"
        raise NotImplementedError

    save_sketch_root = os.path.join(save_root, "dilate_{}_within_bbox".format(dilate_radius))
    if not os.path.exists(save_sketch_root):
        os.mkdir(save_sketch_root)
    print("==> result save in: {}".format(save_sketch_root))
    
    record_dict = {}
    ratio_list = []

    dataset_len = len(dataset)

    pbar = tqdm(range(dataset_len))
    for idx in pbar:
        aitem = dataset[idx]
        filename = aitem["file_name"]
        image = aitem["pixel_values"]
        condition = aitem["conditioning_pixel_values"]
        caption = aitem["input_ids"][0]
        area_list = aitem["area_list"]
        
        ### get area and ratio
        mask_area = float(area_list[0])
        bbox_area = float(area_list[2])
        dilate_mask_area = float(area_list[1])
        mask_bbox_ratio = mask_area / bbox_area
        dilate_bbox_ratio = dilate_mask_area / bbox_area

        record_dict[filename] = {
                "mask_area": mask_area,
                "bbox_area": bbox_area, 
                "dilate_mask_area": dilate_mask_area,
                "mask_bbox_ratio": mask_bbox_ratio, 
                "dilatemask_bbox_ratio": dilate_bbox_ratio, 
            }

        ratio_list.append(dilate_bbox_ratio)
        
        # condition = np.array(condition) * 255
        condition = np.array(condition)
        cur_save_path = os.path.join(save_sketch_root, filename.replace("jpg", "png"))
        condition = cv2.cvtColor(condition, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_save_path, condition)

        if debug and idx >= 10:
            break
    
    ratio_list = np.array(ratio_list)
    print("\t min: {}, mean: {}, max: {}".format(np.min(ratio_list), np.mean(ratio_list), np.max(ratio_list)))

    save_json_path = os.path.join(save_root, "dilate_{}_area_info.json".format(dilate_radius))
    print("==> json save in: {}".format(save_json_path))
    with open(save_json_path, 'w') as f:
        json.dump(record_dict, f, indent=4)
    
    print(":) Congratulations!")


def save_instance_bbox_contour_json(args, dataset, mode="mask"):
    """
    23-11-05, save instance bbox or contours images 
    """
    DEBUG = False 
    # DEBUG = True 

    # save_root = "/mnt/data/xuanwenjie/project/output_dir/debug/lvis_bbox_contour_thick2"
    # save_root = "/dat03/xuanwenjie/datasets/LVIS_COCO_triplet/condition_val/lvisbbox"
    save_root = "/dat03/xuanwenjie/datasets/LVIS_COCO_triplet/condition_train/lvisbbox"
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    dataset_len = len(dataset)

    pbar = tqdm(range(dataset_len))
    for idx in pbar:
        aitem = dataset[idx]
        filename = aitem["file_name"]
        image = aitem["pixel_values"]
        condition = aitem["conditioning_pixel_values"]
        caption = aitem["input_ids"][0]
        width, height = image.size  
        pbar.set_description(filename)

        # draw_image = np.array(image)  
        draw_image = np.zeros((height, width)).astype(np.float32)

        anns = aitem["anns"]
        for cur_ann in anns: 
            if mode == "bbox": 
                x, y, w, h = cur_ann["bbox"]
                # cv2.rectangle(draw_image, (int(x), int(y)), (int(x+w), int(y+h)), (255,255,255), thickness=2)
                cv2.rectangle(draw_image, (int(x), int(y)), (int(x+w), int(y+h)), (255,255,255), thickness=-1)
            
            elif mode == "mask":
                cur_segmentation = cur_ann["segmentation"]
                for ainst_seg in cur_segmentation:
                    rles = mask_utils.frPyObjects([ainst_seg], height, width)
                    rle = mask_utils.merge(rles)
                    mask = mask_utils.decode(rle)
                    # draw_image = (draw_image + mask.astype(np.float32)) 
                    # get contour 
                    mask_contour = cv2.Canny(mask*255, threshold1=50, threshold2=100)
                    # dilate 
                    r = 2
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r)) 
                    mask_contour = cv2.dilate(mask_contour, kernel=kernel) 
                    # draw annotation on the image 
                    draw_image = (draw_image + mask_contour.astype(np.float32)) 
            else: 
                raise Exception("Mode {} not exist!".format(mode))
        
        # save image 
        save_path = os.path.join(save_root, filename.replace("jpg", "png"))
        draw_image_pil = Image.fromarray((draw_image > 0).astype(np.uint8) * 255)
        draw_image_pil.save(save_path)
        
        if DEBUG and idx >= 0:
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo')     
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--proportion_empty_prompts', type=float, default=0.0)
    parser.add_argument('--rdp_epsilon', type=float, default=0.0)
    parser.add_argument('--dilate_radius', type=int, default=None)
 
    args = parser.parse_args()

    args.rdp_epsilon = 0
    print("==> Epsilon = {}".format(args.rdp_epsilon))

    ### 1. save validation images 
    train_dataset = LVIS_Dataset(args, split='val', return_pil=True, return_objname_pos=False)
    save_sketch_images(args, train_dataset)
    
    ### 2. save train images 
    train_dataset = LVIS_Dataset(args, split='train', return_pil=True, return_objname_pos=False)
    save_train_sketch_images(args, train_dataset)

    ### 3. save dilate mask 
    for dilate_radius in [5, 10, 20, 40, 80, 100, ]:
        print("#"*50, " Radiu={} ".format(dilate_radius), "#"*50)
        args.dilate_radius = dilate_radius
        train_dataset = LVIS_Dataset(args, split='val', return_pil=True, return_objname_pos=False, )
        save_dilate_sketch_images(args, train_dataset, mode="val")

    for dilate_radius in [5, 10, 20, 40, 80, 100, ]:
        print("#"*50, " Radius={} ".format(dilate_radius), "#"*50)
        args.dilate_radius = dilate_radius
        train_dataset = LVIS_Dataset(args, split='train', return_pil=True, return_objname_pos=False, )
        save_dilate_sketch_images(args, train_dataset, mode="train")

    ### 4. save object instance bbox 
    ## save val 
    train_dataset = LVIS_Dataset(args, split='val', return_pil=True, return_objname_pos=False)
    save_instance_bbox_contour_json(args, train_dataset, mode="bbox")
    ## save train 
    train_dataset = LVIS_Dataset(args, split='train', return_pil=True, return_objname_pos=False)
    save_instance_bbox_contour_json(args, train_dataset, mode="bbox")