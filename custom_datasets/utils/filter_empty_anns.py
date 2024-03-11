import json
import jsonlines
import cv2
import numpy as np
# from torch.utils.data import Dataset
import os
import datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import matplotlib.pyplot as plt
import pycocotools.mask as mask_utils

from rdp import rdp


def write_jsonl(save_path, item):
    with jsonlines.open(save_path, mode = 'a') as json_writer:
        json_writer.write(item)


def poly2mask(anns, height, width):
    segm = []
    for ainst in anns:
        for seg_part in ainst["segmentation"]:
            # polygon -- a single object might consist of multiple parts, refer to lvis.py
            segm.append(seg_part) 
    # we merge all parts into one mask rle code
    rles = mask_utils.frPyObjects(segm, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask

def gray_to_rgb(image_gray):
    # create a zero array with 3 channels
    height, width = image_gray.shape
    image_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    # copy values
    image_rgb[:, :, 0] = image_gray
    image_rgb[:, :, 1] = image_gray
    image_rgb[:, :, 2] = image_gray
    return image_rgb


if __name__ == "__main__":
    from tqdm import tqdm
    
    LVIS_ROOT = r"/dat03/xuanwenjie/datasets/LVIS"
    LVIS_COCO_ROOT = r"/dat03/xuanwenjie/datasets/LVIS_COCO_triplet"

    # subset = "val_subtrain"  # val img_anns_cap_val_subtrain.jsonl 
    subset = "val" 
    # subset = "train"  

    ### basic info
    if subset == "val_subtrain":
        image_root = os.path.join(LVIS_ROOT, r"train2017")
    else:
        image_root = os.path.join(LVIS_ROOT, r"{}2017".format(subset))

    ### load json
    json_path = os.path.join(LVIS_COCO_ROOT, r"img_anns_cap_{}.jsonl".format(subset))
    json_data = []
    with open(json_path, 'r') as f:
        json_data_lines = f.readlines()
        for line in json_data_lines:
            json_data.append(json.loads(line))

    ### recorder
    empty_count = 0
    empty_save_path = os.path.join(LVIS_COCO_ROOT, r"img_anns_cap_{}_empty.jsonl".format(subset))
    if os.path.exists(empty_save_path):
        os.remove(empty_save_path)

    ### save filtered data 
    save_path = os.path.join(LVIS_COCO_ROOT, r"img_anns_cap_{}_filtered.jsonl".format(subset))
    if os.path.exists(save_path):
        os.remove(save_path)

    ### loop
    pbar = tqdm(json_data)
    for item in pbar:
        pbar.set_description("[{}]".format(subset))

        filename = item["filename"]
        image_id = item["image_id"]
        height = item["height"] 
        width = item["width"] 
        anns = item["anns"] 
        captions = item["captions"] 
        not_exhaustive_category_ids = item["not_exhaustive_category_ids"]
        coco_url = item["coco_url"]

        ### get image
        image = cv2.imread(os.path.join(image_root, filename))

        ### get image caption as prompt
        prompt = [acap["caption"] for acap in captions]

        # Do not forget that OpenCV read images in BGR order.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize images to [0, 1].
        image = image.astype(np.float32) / 255.0

        ### get condition image
        if anns == []:
            write_jsonl(empty_save_path, item)
            empty_count += 1
            continue
        else:
            condition_image = poly2mask(anns, height, width)
            condition_image = gray_to_rgb(condition_image)
            write_jsonl(save_path, item)
        

    print("==> num of empty item = {}".format(empty_count))




