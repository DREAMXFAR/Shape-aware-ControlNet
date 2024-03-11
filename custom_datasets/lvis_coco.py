"""
https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md
https://github.com/HighCWu/ControlLoRA/blob/main/process/diffusiondb_canny.py
"""

import json
import cv2
import numpy as np
# from torch.utils.data import Dataset
import os
import datasets
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random

from rdp import rdp
import pycocotools.mask as mask_utils

import ipdb 


######################################################################################################################################
# Information
######################################################################################################################################
DATA_ROOT = r"/dat03/xuanwenjie/datasets"

### get category json 
category_json_path = os.path.join(DATA_ROOT, r"LVIS_COCO_triplet/catgories.jsonl")
with open(category_json_path, 'r') as f:
    category_json = json.load(f)

### set random seed 
random.seed(42)

######################################################################################################################################
# Class
######################################################################################################################################
class LVIS_Dataset():
    def __init__(self, args=None, tokenizer=None, text_encoder=None, split='val', return_pil=False, return_objname_pos=False, ):
        self.json_path = os.path.join(DATA_ROOT, r"LVIS_COCO_triplet/{}.jsonl".format(split))
        self.image_root = os.path.join(DATA_ROOT, r'LVIS/{}2017'.format(split))
        self.epsilon = args.rdp_epsilon        
        
        if split == 'train':
            self.is_train = True
        elif split == 'val':
            self.is_train = False
        else:
            raise Exception("The split {} is not defined.".format(split))

        self.args = args
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.column_names = []
        self.return_pil = return_pil

        ### dilate mask 
        self.dilate_radius = args.dilate_radius

        self.return_objname_pos = return_objname_pos

        self.data = []
        with open(self.json_path, 'r') as f:
            json_data = f.readlines()
            for line in json_data:
                self.data.append(json.loads(line))


    def __len__(self):
        return len(self.data)


    def corrupt_mask(self, anns):
        for inst_id in range(len(anns)):
            segmentation_list = anns[inst_id]["segmentation"]
            corrupt_segmentation_list = []
            for seg_pts in segmentation_list: 
                seg_pts = np.reshape(seg_pts, (-1, 2))
                seg_pts = rdp(seg_pts, epsilon=self.epsilon)
                seg_pts = seg_pts.reshape(1, -1).tolist()[0]
                if len(seg_pts) <= 4:  # less than two points, debug by xwj 0823
                    continue
                else:
                    corrupt_segmentation_list.append(seg_pts)

            anns[inst_id]["segmentation"] = corrupt_segmentation_list


    def dilate(self, image, r:int=5):
        if r == 0:
            return image
        else:
            # 5x5 elements
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r)) 
            image = np.array(image * 255, dtype=np.uint8)
            dst = cv2.dilate(image, kernel=kernel) 
            return (dst / 255.0).astype(np.float32)

    
    def poly2mask(self, anns, height, width):
        segm = []
        for ainst in anns:
            for seg_part in ainst["segmentation"]:
                # polygon -- a single object might consist of multiple parts, refer to lvis.py
                segm.append(seg_part) 
        # we merge all parts into one mask rle code
        if segm == []:
            mask = np.zeros((height, width))  # debug 0823 by xwj
        else:
            rles = mask_utils.frPyObjects(segm, height, width)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
        
        return mask
    

    def gray_to_rgb(self, image_gray, fg_color=(255, 0, 0), bg_color=(255, 255, 255)):
        # create a zero array with 3 channels
        height, width = image_gray.shape
        image_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        # copy values
        # fg_color, bg_color all in RGB
        image_rgb[:, :, 0][image_gray > 0] = fg_color[0]
        image_rgb[:, :, 1][image_gray > 0] = fg_color[1]
        image_rgb[:, :, 2][image_gray > 0] = fg_color[2]

        image_rgb[:, :, 0][image_gray == 0] = bg_color[0]
        image_rgb[:, :, 1][image_gray == 0] = bg_color[1]
        image_rgb[:, :, 2][image_gray == 0] = bg_color[2]

        return image_rgb


    def tokenize_captions(self, caption):
        # sometimes use non-prompt, noted by xwj
        if random.random() < self.args.proportion_empty_prompts: 
            caption = ""
        elif isinstance(caption, str):
            caption = caption
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            caption = random.choice(caption) if self.is_train else caption[0]
        else:
            raise ValueError(
                f"Caption column `{caption}` should contain either strings or lists of strings."
            )
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids[0]


    def preprocess(self, image, text, conditioning_image, object_name, center_pos, sample_object_pos_mode="default"):
        # get basic information 
        orig_w, orig_h = image.size 

        # preprocess image
        image_transforms = transforms.Compose(
            [
                transforms.Resize(self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # NOTE 
            ]
        )
        image = image_transforms(image)

        # preprocess cond images
        conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.args.resolution),
                transforms.ToTensor(),
            ]
        )
        conditioning_image = conditioning_image_transforms(conditioning_image)
        
        # preprocess text
        if self.tokenizer is None:
            text = text
        else: 
            text = self.tokenize_captions(text)

        # process object name and center pos 
        grounding_tokens = {}
        if self.return_objname_pos:
            ### filter out of range boxes and objects  
            scale_ratio = self.args.resolution * 1.0 / min(orig_w, orig_h)
            scaled_w = int(orig_w * scale_ratio)
            scaled_h = int(orig_h * scale_ratio)
            
            crop_x_min = max(0, int((scaled_w - self.args.resolution)/2))
            crop_x_max = scaled_w - crop_x_min
            crop_y_min = max(0, int((scaled_h - self.args.resolution)/2))
            crop_y_max = scaled_h - crop_y_min
            
            cropped_object_name = []
            cropped_center_pos = []
            for i in range(len(object_name)):
                cur_x = center_pos[i][0] * scale_ratio 
                cur_y = center_pos[i][1] * scale_ratio

                flag = (cur_x>crop_x_min) and (cur_x<crop_x_max) and (cur_y>crop_y_min) and (cur_y<crop_y_max)
                if flag: 
                    new_x = (min(cur_x - crop_x_min, self.args.resolution)) / self.args.resolution
                    new_y = (min(cur_y - crop_y_min, self.args.resolution)) / self.args.resolution
                    cropped_object_name.append(object_name[i])
                    cropped_center_pos.append([new_x, new_y])
            
            # Processing GLIGEN inputs 
            gligen_phrases = cropped_object_name
            gligen_boxes = cropped_center_pos

            # truncate object 
            max_objs = 30 

            if len(gligen_phrases) == 0:
                # text_embeddings = torch.zeros((max_objs, self.text_encoder.config.projection_dim))
                tokenizer_input_ids = torch.zeros(max_objs, self.tokenizer.model_max_length)
                tokenizer_attention_mask = torch.zeros(max_objs, self.tokenizer.model_max_length)
                bboxes = torch.zeros(max_objs, 2, )
                masks = torch.zeros(max_objs, )
            else: 
                if len(gligen_phrases) > 30:
                    if sample_object_pos_mode == "random": 
                        # random sample 30 
                        sample_ids = random.sample(range(len(gligen_phrases)), max_objs)
                        gligen_phrases = [gligen_phrases[j] for j in sample_ids]
                        gligen_boxes = [gligen_boxes[j] for j in sample_ids]
                        gligen_boxes = np.array(gligen_boxes)
                    elif sample_object_pos_mode == "default": 
                        # select the first 30 
                        gligen_phrases = gligen_phrases[:30]
                        gligen_boxes = gligen_boxes[:30, :]
            
                n_objs = len(gligen_phrases)

                # tokenize phrases, {"attention_mask": (30, 77), "inputs_ids": (30, 77)}
                tokenizer_inputs = self.tokenizer(gligen_phrases, max_length=self.tokenizer.model_max_length, padding="max_length", return_tensors="pt")
                # tokenizer_inputs.to(self.text_encoder.device)
                
                _tokenizer_input_ids = tokenizer_inputs["input_ids"]
                tokenizer_input_ids = torch.zeros(max_objs, self.tokenizer.model_max_length)
                tokenizer_input_ids[:n_objs] = _tokenizer_input_ids
                
                _tokenizer_attention_mask = tokenizer_inputs["attention_mask"]
                tokenizer_attention_mask = torch.zeros(max_objs, self.tokenizer.model_max_length)
                tokenizer_attention_mask[:n_objs] = _tokenizer_attention_mask

                ### this implementation cause: CUDA error initialization error 
                # _text_embeddings = self.text_encoder(**tokenizer_inputs).pooler_output
                # text_embeddings = torch.zeros(max_objs, self.text_encoder.config.projection_dim) # unet.config.cross_attention_dim
                # text_embeddings[:n_objs] = _text_embeddings  # (30, 768)  
                
                # process boxes 
                bboxes = torch.zeros(max_objs, 2, )
                bboxes[:n_objs] = torch.tensor(gligen_boxes)
                # Generate a mask for each object that is entity described by phrases
                masks = torch.zeros(max_objs, )
                masks[:n_objs] = 1 
                
            # grounding_tokens["object_name_embedding"] = text_embeddings  # gligen_phrases
            grounding_tokens["object_name_input_ids"] = tokenizer_input_ids
            # grounding_tokens["object_name_input_ids"] = gligen_phrases  # NOTE
            grounding_tokens["object_name_attention_mask"] = tokenizer_attention_mask
            grounding_tokens["center_pos"] = bboxes
            grounding_tokens["masks"] = masks 

        return image, text, conditioning_image, grounding_tokens
  

    def get_object_name_from_anns(self, anns):
        object_name = []
        for cur_ann in anns:
            cur_category_id = cur_ann["category_id"]
            cur_category_name = category_json["{}".format(cur_category_id)]["name"]
            object_name.append(cur_category_name)

        return object_name


    def get_center_pos_from_anns(self, anns, ):
        center_pos = []
        for cur_ann in anns:
            cur_x, cur_y, cur_w, cur_h = cur_ann["bbox"] 
            cur_center_x = cur_x + cur_w/2.0
            cur_center_y = cur_y + cur_h/2.0

            center_pos.append([cur_center_x, cur_center_y])

        return np.array(center_pos)   


    def mask2bbox(self, anns, height, width):
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


    def __getitem__(self, idx):
        item = self.data[idx]
        ### get basic info 
        filename = item["filename"]
        image_id = item["image_id"]
        height = item["height"] 
        width = item["width"] 
        anns = item["anns"] 
        captions = item["captions"] 
        not_exhaustive_category_ids = item["not_exhaustive_category_ids"]
        coco_url = item["coco_url"]

        ### get image
        image = Image.open(os.path.join(self.image_root, filename)).convert('RGB')

        ### get condition image
        if not (self.epsilon is None or self.epsilon == 0):
            condition_image = self.corrupt_mask(anns)
        # convert to mask
        conditioning_image = self.poly2mask(anns, height, width)

        ### dilate 
        area_list = []
        deteriorate_ratio = 0
        deteriorate_radius = 0 
        if self.dilate_radius is not None:
            # ipdb.set_trace()
            # original mask area 
            area_mask = np.sum(conditioning_image)
            # bbox area 
            conditioning_bbox = self.mask2bbox(anns, height, width)
            area_bbox = np.sum(conditioning_bbox)
            # dilated mask area             
            if self.dilate_radius == "random": 
                random_r = np.random.randint(low=0, high=80)  # before 231220, r=130, iou=98.8
                # random_r = 30 + 30 * np.random.randn()
            else:
                random_r = int(self.dilate_radius)
            
            if self.dilate_radius == "bbox": 
                conditioning_image = conditioning_bbox  
            else: 
                conditioning_image = self.dilate(conditioning_image, r=random_r)
            conditioning_image = conditioning_bbox * conditioning_image
            
            # compute area 
            area_dilate_mask_within_bbox = np.sum(conditioning_image)
            # collect info 
            area_list = [area_mask, area_dilate_mask_within_bbox, area_bbox]
            ### two choice: area ratio or radius 
            deteriorate_ratio = (area_dilate_mask_within_bbox - area_mask) / (area_bbox - area_mask + 1e-5)
            deteriorate_radius = random_r

        # convert to 3-channel
        # white foreground and black background
        conditioning_image = self.gray_to_rgb(conditioning_image, fg_color=(255, 255, 255), bg_color=(0, 0, 0)) 
        # red foreground and white background
        # conditioning_image = self.gray_to_rgb(conditioning_image, fg_color=(255, 0, 0), bg_color=(255, 255, 255))

        ### vis image for check
        # vis_img = Image.fromarray(np.uint8(conditioning_image * 255.0))
        # vis_img.save(r"/mnt/data/xuanwenjie/project/output_dir/lvis_cond.png")

        conditioning_image = Image.fromarray(conditioning_image)

        ### get image caption as prompt
        text = [acap["caption"] for acap in captions]

        ### get object_name and position, 230923 
        object_name = None 
        center_pos = None 
        if self.return_objname_pos:
            object_name = self.get_object_name_from_anns(anns)
            center_pos = self.get_center_pos_from_anns(anns)
        
        # the default parser settings: 
        # --image_column image
        # --conditioning_image_column conditioning_image
        # --caption_column text
        if self.return_pil:
            return dict(
                    file_name=filename, 
                    pixel_values=image, 
                    input_ids=text, 
                    conditioning_pixel_values=conditioning_image, 
                    anns=anns, 
                    object_name=object_name, 
                    center_pos=center_pos, 
                    area_list=area_list, 
                    deter_ratio=deteriorate_ratio, 
                    deteriorate_radius=deteriorate_radius, 
                )
        else:
            image, text, conditioning_image, grounding_tokens = self.preprocess(
                                                                    image=image, 
                                                                    text=text, 
                                                                    conditioning_image=conditioning_image, 
                                                                    object_name=object_name, 
                                                                    center_pos=center_pos, 
                                                                    sample_object_pos_mode="random", 
                                                                )
            return dict(file_name=filename, 
                        pixel_values=image, 
                        input_ids=text, 
                        conditioning_pixel_values=conditioning_image,
                        deteriorate_ratio=deteriorate_ratio, 
                        deteriorate_radius=deteriorate_radius, 
                        **grounding_tokens)


######################################################################################################################################
# Funtions
######################################################################################################################################
def LVIS_demo():
    DATA_PATH = os.path.join(DATA_ROOT, r"img_anns_cap_val.jsonl")

    def map_fn(kwargs):
        def _map_fn(filename, image_id, height, width, anns, captions, not_exhaustive_category_ids, coco_url):
            image = os.path.join(DATA_ROOT, r'LVIS/val2017', filename)
            condition = os.path.join(DATA_ROOT, r"LVIS/val2017", filename)
            text = captions
            return dict(image=image, condition=condition, text=text)
        return _map_fn(**kwargs)

    dataset = Dataset.from_json(DATA_PATH) 
    dataset = dataset.map(map_fn)
    # fill50k_dataset = fill50k_dataset.remove_columns(['source', 'target', 'prompt'])
    dataset = dataset.cast_column("image", datasets.Image(decode=True))
    dataset = dataset.cast_column("condition", datasets.Image(decode=True))
    dataset = DatasetDict(train=dataset)

    plt.figure()
    plt.imshow(dataset['train']['image'][0])
    # plt.show()
    plt.close()
    
    print("Finish")


def erode(image, r=5):
    if r == 0:
        return image
    else:
        # 5x5 elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r)) 
        image = np.array(image * 255, dtype=np.uint8)
        dst = cv2.erode(image, kernel=kernel) 
        return (dst / 255.0).astype(np.float32)


def save_sketch_images(epsilon=None):
    if epsilon is None or epsilon == 0:
        save_sketch_root = r"/mnt/data/tmp_users/xuanwenjie/diffusers/examples/controlnet/output_dir/skech_demo/raw"      
    else:
        save_sketch_root = r"/mnt/data/tmp_users/xuanwenjie/diffusers/examples/controlnet/output_dir/skech_demo/epsilon_{}".format(epsilon)
    save_image_root = r"/mnt/data/tmp_users/xuanwenjie/diffusers/examples/controlnet/output_dir/skech_demo/image"
    
    for apath in [save_image_root, save_sketch_root]:
        if not os.path.exists(apath):
            os.mkdir(apath)

    dataset = LVIS_Dataset(epsilon=epsilon)
    for idx, aitem in enumerate(dataset):
        filename = aitem["filename"]
        image = aitem["image"]
        condition = aitem["condition"]
        
        image = image * 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cur_save_path = os.path.join(save_image_root, filename)
        if os.path.exists(cur_save_path):
            pass
        else:
            cv2.imwrite(cur_save_path, image)   

        condition = condition * 255
        cur_save_path = os.path.join(save_sketch_root, filename)
        cv2.imwrite(cur_save_path, condition)

        if idx >= 99:
            break


if __name__ == "__main__":
    ### save sketches
    # save_sketch_images(epsilon=5)
    from transformers import AutoTokenizer, PretrainedConfig
    from datasets import load_dataset
    import argparse

    def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

            return RobertaSeriesModelWithTransformation
        else:
            raise ValueError(f"{model_class} is not supported.")


    parser = argparse.ArgumentParser(description='Demo')     
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--proportion_empty_prompts', type=float, default=0.0)
    parser.add_argument('--rdp_epsilon', type=float, default=0.0)
    parser.add_argument('--dilate_radius', type=str, default=10)
     
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(
        r"/mnt/data/users/xuanwenjie/pretrained_param/StableDiffusion/stable-diffusion-v1-5",
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    ) 

    # text_encoder_cls = import_model_class_from_model_name_or_path(
    #     r"/mnt/data/xuanwenjie/pretrained_param/StableDiffusion/stable-diffusion-v1-5", 
    #     False, 
    # )
    # text_encoder = text_encoder_cls.from_pretrained(
    #     r"/mnt/data/xuanwenjie/pretrained_param/StableDiffusion/stable-diffusion-v1-5", 
    #     subfolder="text_encoder", 
    #     revision=False, 
    # )
    text_encoder = None 
    
    train_dataset = LVIS_Dataset(
            args,
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            return_objname_pos=True)


    def collate_fn(examples):
        print("==> example keys: {}".format(examples[0].keys()))

        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.stack([example["input_ids"] for example in examples])

        return_dict = {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "input_ids": input_ids,
        }

        ### add additional key and values 
        # collect for mask, object_name and center pose 
        if "object_name_input_ids" in examples[0].keys():
            # return_dict["object_name_embedding"] = torch.stack([example["object_name_embedding"] for example in examples])
            return_dict["object_name_input_ids"] = [example["object_name_input_ids"] for example in examples]

        if "center_pos" in examples[0].keys():    
            return_dict["center_pos"] = torch.stack([example["center_pos"] for example in examples])

        if "masks" in examples[0].keys():
            return_dict["masks"] = torch.stack([example["masks"] for example in examples])
        
        return return_dict


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn, 
        batch_size=1,
        num_workers=0,
    )

    # data_loader = load_dataset(dataset, cache_dir=r"/mnt/data/xuanwenjie/hf_cache")
    for aitem in train_dataloader:
        print(type(aitem))
        
        image = aitem["pixel_values"]
        conditioning_image = aitem["conditioning_pixel_values"]
        
        # conditioning_image in [0, 1]
        print("==> cond_tensor: {}".format(conditioning_image.shape))
        print("==> min={}, max={}".format(conditioning_image.min(), conditioning_image.max()))
        
        cond_pil = conditioning_image.squeeze().numpy().transpose([1, 2, 0])
        # cond_pil = image.squeeze().numpy().transpose([1, 2, 0])
        cond_pil = np.ascontiguousarray(cond_pil * 255, dtype=np.uint8)
        w, h, c = cond_pil.shape 
        
        if train_dataset.return_objname_pos: 
            for i in range(30):
                amask = aitem["masks"][0, i]

                if amask == 1:
                    aobj_name = aitem["object_name_input_ids"][0][i]
                    acenter_pos = aitem["center_pos"][0, i, :]

                    acenter_pos = acenter_pos * w 
                    ax = int(acenter_pos[0])
                    ay = int(acenter_pos[1])
                    # cv2.putText(cond_pil, aobj_name + "({}, {})".format(ax, ay), (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
       
        cond_pil = Image.fromarray(np.uint8(cond_pil))       
        # print("cond image: {}".format(cond_pil))
        print("==> cond shape: {}".format(cond_pil.size))
        cond_pil.save(r"/mnt/data/tmp_users/xuanwenjie/codes/controlnet/output_dir/debug/check_lvis_coco.png")
        
        break

