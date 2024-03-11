import os
import numpy as np
import matplotlib.pyplot as plt

import re
import tempfile
from collections import Counter
import json as JSON
import jsonlines
import orjson as json
import cv2
import random 
from tqdm import tqdm

import skimage.io as io
from pycocotools.coco import COCO
import pycocotools as pycoco
from lvis import LVIS
from lvis import LVISVis

from PIL import Image 


###############################################################################################
# Global Variables
###############################################################################################
COCO_ROOT = r"/dat03/xuanwenjie/datasets/COCO"
COCO_PATHES = {
    'root': os.path.join(COCO_ROOT, "annotations"),
    'train2017': os.path.join(COCO_ROOT, "train2017"), 
    'val2017': os.path.join(COCO_ROOT, "val2017"),
    'train_seg': os.path.join(COCO_ROOT, r"annotations/instances_train2017.json"),
    'val_seg': os.path.join(COCO_ROOT, r"annotations/instances_val2017.json"),   
    'train_caption': os.path.join(COCO_ROOT, r"annotations/captions_train2017.json"), 
    'val_caption': os.path.join(COCO_ROOT, r"annotations/captions_val2017.json"), 
}

LVIS_ROOT = "/dat03/xuanwenjie/datasets/LVIS"
LVIS_PATHES = {
    'root': LVIS_ROOT,
    'train2017': os.path.join(LVIS_ROOT, "train2017"),
    'val2017': os.path.join(LVIS_ROOT, "val2017"),
    'train_seg': os.path.join(LVIS_ROOT, "lvis_v1_train.json"),
    'val_seg': os.path.join(LVIS_ROOT, "lvis_v1_val.json"),   
}

###############################################################################################
# Functions
###############################################################################################
def write_jsonl(save_path, item):
    # 建立data.jsonl文件，以追加的方式写入数据
    with jsonlines.open(save_path, mode = 'a') as json_writer:
        json_writer.write(item)


def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def draw_hist(name_list, count_list, dataset_name, save_fig=False, num_labeling=True):
    """
    draw hist map from Counter()
    """
    ### draw hist map
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/4., 1.01*height, '%s' % int(height), size=10)
    
    label = name_list
    y = count_list
    x = range(len(y))

    ### Show image
    fig = plt.figure()  # figsize=(8, 3)
    ax1 = fig.add_subplot(111)
    cm = ax1.bar(x, y,tick_label=label, width=0.3)
    # set axies params
    plt.xticks(rotation=0)
    plt.tick_params(labelsize=4) 
    # label data
    if num_labeling:
        autolabel(cm)
    
    plt.title(dataset_name, fontdict={'family': 'serif', 'size': 10, 'color': 'red', 'weight': 'bold'})
    plt.xlabel("Task Category", fontdict={'family': 'serif', 'size': 10, 'color': 'red', 'weight': 'bold'})
    plt.ylabel("# Sample Num", fontdict={'family': 'serif', 'size': 10, 'color': 'red', 'weight': 'bold'})

    if save_fig:
        # save fig
        save_path = r"/dat03/xuanwenjie/code/controlnet/output_dir/data_info/{}.png".format(dataset_name)
        plt.savefig(save_path, dpi=300)
        # plt.close()
        print("==> the task distribution saved in: {}".format(save_path))

    # plt.show()
    plt.close()
    

###############################################################################################
# Main
###############################################################################################
def check_img_anns_cap_json(select="train"): 
    ### 6. check all images, captions, segmentation pairs
    LVIS_COCO_ROOT = r"/dat03/xuanwenjie/datasets/LVIS_COCO_triplet"
    train_json_path = os.path.join(LVIS_COCO_ROOT, r"img_anns_cap_train.jsonl")
    val_json_path = os.path.join(LVIS_COCO_ROOT, r"img_anns_cap_val.jsonl")
    val_subtrain_json_path = os.path.join(LVIS_COCO_ROOT, r"img_anns_cap_val_subtrain.jsonl")

    if select == "train":
        json_path = train_json_path
    elif select == "val":
        json_path = val_json_path
    elif select == "subtrain":
        json_path = val_subtrain_json_path
    else:
        raise Exception(r"Wrong checking mode! Only support [train, val, subtrain]. ")

    with open(json_path, 'r') as f:
        json_data = []
        for aline in f.readlines():
            json_data.append(json.loads(aline))
    
    num_data = len(json_data)

    random.seed(666)
    sampled_img_ids = random.sample(range(num_data), 10)
    # sampled_img_ids = list(range(num_data))

    # Counter
    cap_per_img_counter = Counter()
    ann_per_img_counter = Counter()
    img_counter = 0

    pbar = tqdm(range(num_data))

    for aid in pbar:
        pbar.set_description("[{}-info-check]".format(select))

        cur_data = json_data[aid]
        cur_filename = cur_data["filename"]
        cur_anns = cur_data["anns"]
        cur_captions = cur_data["captions"]

        img_counter = img_counter + 1
        cap_per_img_counter.update([len(cur_captions)])
        ann_per_img_counter.update([len(cur_anns)])
    
    ### print info
    cur_counter = cap_per_img_counter
    name_list = [i for i in cur_counter.keys()]
    count_list = [i for i in cur_counter.values()]
    draw_hist(name_list=name_list, count_list=count_list, dataset_name="captions-per-image_{}".format(select), save_fig=True) 

    cur_counter = ann_per_img_counter
    name_list = [i for i in cur_counter.keys()]
    count_list = [i for i in cur_counter.values()]
    sorted_ids = np.argsort(-np.array(count_list))  # in reversed order
    sorted_count_list = [count_list[id] for id in sorted_ids]    
    sorted_name_list = [name_list[id] for id in sorted_ids]  
    draw_hist(name_list=sorted_name_list, count_list=sorted_count_list, dataset_name="anns-per-mage_{}".format(select), save_fig=True) 

    print("==> Sum = {}".format(img_counter))
    
    ### check zero annotation info
    if "0" in ann_per_img_counter.keys():
        print("Warning! {} images have zero annotations!".format(ann_per_img_counter["0"]))
    if "0" in cap_per_img_counter.keys():
        print("Warning! {} images have zero caption!".format(cap_per_img_counter["0"]))



# ---------------------------------------------------------------------------------------------
def generate_cocolvis_val_jsonl(debug=False):
    ### get raw data
    coco_train_caption = COCO(COCO_PATHES['train_caption'])
    lvis_train = LVIS(LVIS_PATHES['train_seg'])
    dataset = lvis_train

    lvis_val = LVIS(LVIS_PATHES['val_seg'])
    dataset = lvis_val
    coco_val_caption = COCO(COCO_PATHES['val_caption'])
    
    ### basic info
    dataset_len = len(dataset.imgs)
    count = 0

    # set save path
    SAVE_ROOT = r"/dat03/xuanwenjie/datasets/LVIS_COCO_triplet"
    save_val_path = os.path.join(SAVE_ROOT, r"img_anns_cap_val.jsonl")
    save_train_path = os.path.join(SAVE_ROOT, r"img_anns_cap_val_subtrain.jsonl")
    for afile_path in [save_val_path, save_train_path]:
        if os.path.exists(afile_path):
            os.remove(afile_path)

    # for loop
    pbar = tqdm(dataset.imgs.items())
    for idx, cur_img in pbar:
        count += 1
        pbar.set_description("[val-set]".format(count, dataset_len))

        ### get basic image information
        cur_img_id = cur_img["id"]
        cur_width = cur_img["width"]
        cur_height = cur_img["height"]
        cur_url = cur_img["coco_url"]

        cur_filename = os.path.basename(cur_img["coco_url"])
        cur_subset = cur_img["coco_url"].split(r"/")[-2]
        cur_noexhaustive_cats_id = cur_img["not_exhaustive_category_ids"]

        if cur_subset == "train2017":
            ref_dataset = coco_train_caption
            cur_save_path = save_train_path
        elif cur_subset == "val2017": 
            ref_dataset = coco_val_caption
            cur_save_path = save_val_path
        else:
            print("==> Not existing a subset called: {}".format(cur_subset))

        ### get annotation info
        # instance-seg
        cur_anns_ids = dataset.get_ann_ids(img_ids=[cur_img_id])
        cur_anns = dataset.load_anns(cur_anns_ids)

        # caption
        cur_caption_ids = ref_dataset.getAnnIds(imgIds=[cur_img_id])
        cur_captions = ref_dataset.loadAnns(cur_caption_ids)

        assert cur_captions[0]["image_id"] == cur_img_id
        
        ### collect triplet pair
        cur_line = {
            "filename": cur_filename, 
            "image_id": cur_img_id, 
            "height": cur_height,
            "width": cur_width, 
            "anns": cur_anns,
            "captions": cur_captions, 
            "not_exhaustive_category_ids": cur_noexhaustive_cats_id, 
            "coco_url": cur_url, 
        }
        write_jsonl(cur_save_path, cur_line)

        ### check 
        # img path
        coco_img_path = os.path.join(LVIS_PATHES["root"], cur_subset, cur_filename)
        try:
            I = io.imread(coco_img_path)
        except:
            print("!!! wrong image: {}".format(coco_img_path))

        if debug and count > 2:
            break
    
    print("==> Finish!")


def generate_cocolvis_train_jsonl(debug=False):
    ### get raw data
    coco_train_caption = COCO(COCO_PATHES['train_caption'])
    lvis_train = LVIS(LVIS_PATHES['train_seg'])
    dataset = lvis_train
    
    ### basic info
    dataset_len = len(dataset.imgs)
    count = 0

    # set save path
    SAVE_ROOT = r"/dat03/xuanwenjie/datasets/LVIS_COCO_triplet"
    save_train_path = os.path.join(SAVE_ROOT, "img_anns_cap_train.jsonl")
    if os.path.exists(save_train_path):
        os.remove(save_train_path)

    # for loop
    pbar = tqdm(dataset.imgs.items())
    for idx, cur_img in pbar:
        count += 1
        # print("[{:6}|{}]".format(count, dataset_len))
        pbar.set_description("[train-set]")

        ### get basic image information
        cur_img_id = cur_img["id"]
        cur_width = cur_img["width"]
        cur_height = cur_img["height"]
        cur_url = cur_img["coco_url"]

        cur_filename = os.path.basename(cur_img["coco_url"])
        cur_subset = cur_img["coco_url"].split(r"/")[-2]
        cur_noexhaustive_cats_id = cur_img["not_exhaustive_category_ids"]

        if cur_subset == "train2017":
            ref_dataset = coco_train_caption
            cur_save_path = save_train_path
        else:
            print("==> Not existing a subset called: {}".format(cur_subset))

        ### get annotation info
        # instance-seg
        cur_anns_ids = dataset.get_ann_ids(img_ids=[cur_img_id])
        cur_anns = dataset.load_anns(cur_anns_ids)

        # caption
        cur_caption_ids = ref_dataset.getAnnIds(imgIds=[cur_img_id])
        cur_captions = ref_dataset.loadAnns(cur_caption_ids)

        assert cur_captions[0]["image_id"] == cur_img_id
        
        ### collect triplet pair
        cur_line = {
            "filename": cur_filename, 
            "image_id": cur_img_id, 
            "height": cur_height,
            "width": cur_width, 
            "anns": cur_anns,
            "captions": cur_captions, 
            "not_exhaustive_category_ids": cur_noexhaustive_cats_id, 
            "coco_url": cur_url, 
        }
        write_jsonl(cur_save_path, cur_line)

        ### check 
        # img path
        coco_img_path = os.path.join(LVIS_PATHES["root"], cur_subset, cur_filename)
        try:
            I = io.imread(coco_img_path)
        except:
            print("!!! wrong image: {}".format(coco_img_path))

        if debug and count > 2:
            break
    
    print("==> Finish!")


def generate_cocolvis_category(debug=False):
    ### get raw data
    lvis_train = LVIS(LVIS_PATHES['train_seg']) 
    lvis_val = LVIS(LVIS_PATHES['val_seg'])
    
    ### basic info
    dataset_len = len(lvis_train.cats)
    count = 0

    # set save path
    LVIS_COCO_ROOT = r"/dat03/xuanwenjie/datasets/LVIS_COCO_triplet"
    save_cat_path = os.path.join(LVIS_COCO_ROOT, "catgories.jsonl")
    # if os.path.exists(save_cat_path):
    #     os.remove(save_cat_path)

    # for loop
    rewrite_json ={}
    
    pbar = tqdm(lvis_val.cats.keys())
    for idx in pbar:
        count += 1
        pbar.set_description("[category]")

        cur_train_cat = lvis_train.cats[idx]
        cur_val_cat = lvis_val.cats[idx]

        ### get train 
        cur_train_id = cur_train_cat["id"]
        cur_train_name = cur_train_cat["name"]
        cur_train_def = cur_train_cat["def"]
        cur_train_synonyms = cur_train_cat["synonyms"]
        cur_train_synset = cur_train_cat["synset"]
        cur_train_frequency = cur_train_cat["frequency"]
        cur_train_instance_count = cur_train_cat["instance_count"]
        cur_train_image_count = cur_train_cat["image_count"]

        ### get val
        cur_val_id = cur_val_cat["id"]
        assert cur_val_id == cur_train_id
        cur_val_name = cur_val_cat["name"]
        assert cur_val_name == cur_train_name
        cur_val_def = cur_val_cat["def"]
        assert cur_val_def == cur_train_def
        cur_val_synonyms = cur_val_cat["synonyms"]
        assert cur_val_synonyms == cur_train_synonyms
        cur_val_synset = cur_val_cat["synset"]
        assert cur_val_synset == cur_train_synset
        cur_val_frequency = cur_val_cat["frequency"]
        assert cur_val_frequency == cur_train_frequency  # same for train and val
        cur_val_instance_count = cur_val_cat["instance_count"]
        cur_val_image_count = cur_val_cat["image_count"]

        ### not same 
        # assert cur_val_image_count == cur_train_image_count 
        # assert cur_val_instance_count == cur_train_instance_count

        ### collect triplet pair
        rewrite_json[idx] = {
            "id": cur_train_id, 
            "name": cur_train_name, 
            "def": cur_train_def,
            "synonyms": cur_train_synonyms, 
            "synset": cur_train_synset,
            "frequency": cur_train_frequency, 
            "instance_count_train": cur_train_instance_count, 
            "instance_count_val": cur_val_instance_count, 
            "image_count_train": cur_train_image_count, 
            "image_count_val": cur_val_image_count, 
        }

        if debug and count > 2:
            break
    
    with open(save_cat_path, 'w') as f:
        JSON.dump(rewrite_json, f, indent=4)

    print("==> Finish!")


if __name__ == "__main__":

    ### get validataion, short for check 
    # 1. generate val jsonl 
    generate_cocolvis_val_jsonl()
    # 2. check hist 
    check_img_anns_cap_json(select='val')
    check_img_anns_cap_json(select='subtrain')

    ### 2. get train jsonl
    generate_cocolvis_train_jsonl()
    check_img_anns_cap_json(select="train")

    ### 3. get category file 
    generate_cocolvis_category()





