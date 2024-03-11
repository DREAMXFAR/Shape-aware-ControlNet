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


######################################################################################################################################
# Information
######################################################################################################################################
### dataset root
DATASET_ROOT = r"/path/to/datasets/LVIS_COCO_triplet"

### get category json 
category_json_path = os.path.join(DATASET_ROOT, r"catgories.jsonl")
with open(category_json_path, 'r') as f:
    category_json = json.load(f)

### set random seed 
random.seed(42)

######################################################################################################################################
# Class
######################################################################################################################################
class CocoOfflineDataset():
    """
    The condition images are offline generated.  
    """
    def __init__(self, args=None, tokenizer=None, text_encoder=None, split='val', return_objname_pos=False):
        ### set path
        self.json_path = os.path.join(DATASET_ROOT, r"{}.jsonl".format(split))
        self.image_root = os.path.join(DATASET_ROOT, r'{}2017'.format(split))

        self.conditioning_mode = args.conditioning_mode
        # mode in: canny, hed, fakescribble, midas, mlsd
        self.conditioning_root = os.path.join(DATASET_ROOT, r'condition_{}/{}'.format(split, self.conditioning_mode))

        # 230922 by xwj, adapt to GLIGEN grounding token 
        self.return_objname_pos = return_objname_pos

        # 231001 by xwj, add mixed conditioning data for training
        self.mixed_condition = args.mixed_condition
        self.mixed_rate = 0.5
        
        if split == 'train':
            self.is_train = True
        elif split == 'val':
            self.is_train = False
        else:
            raise Exception("The split {} is not defined.".format(split))

        self.args = args
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder  # not used 
        self.column_names = []

        self.data = []
        with open(self.json_path, 'r') as f:
            json_data = f.readlines()
            for line in json_data:
                self.data.append(json.loads(line))


    def __len__(self):
        return len(self.data)


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
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
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
                ### original implementation 
                transforms.Resize(self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.args.resolution), 
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]), 
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
            tokenizer_inputs.to(self.text_encoder.device)
            _text_embeddings = self.text_encoder(**tokenizer_inputs).pooler_output
            text_embeddings = torch.zeros(max_objs, self.text_encoder.config.projection_dim) # unet.config.cross_attention_dim
            text_embeddings[:n_objs] = _text_embeddings  # (30, 768)  
            
            # process boxes 
            bboxes = torch.zeros(max_objs, 2, )
            bboxes[:n_objs] = torch.tensor(gligen_boxes)
            # Generate a mask for each object that is entity described by phrases
            masks = torch.zeros(max_objs, )
            masks[:n_objs] = 1 
            
            grounding_tokens["object_name_embedding"] = text_embeddings  # gligen_phrases
            # grounding_tokens["object_name_embedding"] = gligen_phrases
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
        image_path = os.path.join(self.image_root, filename)
        image = Image.open(image_path).convert('RGB')

        ### get condition image
        if self.conditioning_mode in ['cocostuff', 'classtag', 'dilate_20_within_bbox', 'dilate_40_within_bbox', 'dilate_80_within_bbox', 'dilate_100_within_bbox', 'dilate_120_within_bbox', "dilate_5_within_bbox", "dilate_10_within_bbox"]:
            conditioning_image = Image.open(os.path.join(self.conditioning_root, filename).replace('jpg', 'png')).convert('RGB')
        else:
            conditioning_image_path = os.path.join(self.conditioning_root, filename)
            if self.mixed_condition is not None:
                if random.random() > self.mixed_rate:
                    conditioning_image_path = conditioning_image_path.replace(self.conditioning_mode, self.mixed_condition)
            
            # print("==> conditioning_image_path = {}".format(conditioning_image_path))  # debug 
            conditioning_image = Image.open(conditioning_image_path).convert('RGB')

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
        if self.args is None:
            return dict(
                    filename=filename, pixel_values=image, input_ids=text, conditioning_pixel_values=conditioning_image, 
                )
        else:
            # NOTE: sample_object_pos_mode = "default": the first 30 objects may have the same category, "random": random sample 30
            image, text, conditioning_image, grounding_tokens = self.preprocess(
                                                                            image=image, 
                                                                            text=text, 
                                                                            conditioning_image=conditioning_image, 
                                                                            object_name=object_name, 
                                                                            center_pos=center_pos,
                                                                            sample_object_pos_mode="random", 
                                                                        )
            return dict(
                    file_name=filename, pixel_values=image, input_ids=text, conditioning_pixel_values=conditioning_image,
                    **grounding_tokens, 
                )


######################################################################################################################################
# __main__
######################################################################################################################################
if __name__ == "__main__":
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
    parser.add_argument(
        "--conditioning_mode", type=str, default=None,
        help=( "Set the condition mode for custom offline dataset by xwj."),
    )
    parser.add_argument(
        "--mixed_condition", type=str, default=None,
    )    

    args = parser.parse_args()
    args.conditioning_mode = 'lvis_blackbg'
    args.mixed_condition = 'lvisbbox'

    tokenizer = AutoTokenizer.from_pretrained(
        r"/mnt/data/xuanwenjie/pretrained_param/StableDiffusion/stable-diffusion-v1-5",
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    ) 

    text_encoder_cls = import_model_class_from_model_name_or_path(
        r"/mnt/data/xuanwenjie/pretrained_param/StableDiffusion/stable-diffusion-v1-5", 
        False, 
    )
    text_encoder = text_encoder_cls.from_pretrained(
        r"/mnt/data/xuanwenjie/pretrained_param/StableDiffusion/stable-diffusion-v1-5", 
        subfolder="text_encoder", 
        revision=False, 
    )
    
    # canny  fakescribble  hed  midas  mlsd
    train_dataset = CocoOfflineDataset(args, split='train', tokenizer=tokenizer, return_objname_pos=False)
    # train_dataset = CocoOfflineDataset(args, split='val', tokenizer=tokenizer, return_objname_pos=False)


    def collate_fn(examples):
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
        if "object_name_embedding" in examples[0].keys():
            # return_dict["object_name_embedding"] = torch.stack([example["object_name_embedding"] for example in examples])
            return_dict["object_name_embedding"] = [example["object_name_embedding"] for example in examples]

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

    ### detokenize
    detokenize_dict = tokenizer.decoder
    cnt = 0

    # data_loader = load_dataset(dataset, cache_dir=r"/mnt/data/xuanwenjie/hf_cache")
    for aitem in train_dataloader:
        print(type(aitem))
        # conditioning_image in [0, 1]
        
        image = aitem["pixel_values"]
        conditioning_image = aitem["conditioning_pixel_values"]
        
        print("==> cond_tensor: {}".format(conditioning_image.shape))
        print("==> min={}, max={}".format(conditioning_image.min(), conditioning_image.max()))
        
        # cond_pil = conditioning_image.squeeze().numpy().transpose([1, 2, 0])
        cond_pil = image.squeeze().numpy().transpose([1, 2, 0])
        cond_pil = np.ascontiguousarray(cond_pil * 255, dtype=np.uint8)
        w, h, c = cond_pil.shape 
        
        if train_dataset.return_objname_pos: 
            for i in range(30):
                amask = aitem["masks"][0, i]

                if amask == 1:
                    aobj_name = aitem["object_name_embedding"][0][i]
                    acenter_pos = aitem["center_pos"][0, i, :]

                    acenter_pos = acenter_pos * w 
                    ax = int(acenter_pos[0])
                    ay = int(acenter_pos[1])
                    cv2.putText(cond_pil, aobj_name, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
       
        cond_pil = Image.fromarray(np.uint8(cond_pil))
        # print("cond image: {}".format(cond_pil))
        # print("==> cond shape: {}".format(cond_pil.size))
        # cond_pil.save(r"/mnt/data/xuanwenjie/project/output_dir/debug/check_offline.png")
        
        cnt = cnt + 1
        if cnt > 100:
            break


"""debug
image = Image.fromarray(np.uint8(aitem["conditioning_pixel_values"].squeeze().numpy().transpose([1, 2, 0]))*255)
"""
