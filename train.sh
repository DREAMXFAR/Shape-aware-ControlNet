#!/bin/bash
export HF_HOME="/dat03/xuanwenjie/cache_dir/hf_cache" 
export MODEL_DIR="/dat03/xuanwenjie/pretrained_models/StableDiffusion/stable-diffusion-v1-5"
export OUTPUT_DIR="./output_dir/debug"
export CUDA_VISIBLE_DEVICES=3

DATASET_NAME="lvis"
CONDITIONING_MODE="epsilon_0_blackbg"
TRACKER_PROJECT_NAME="controlnet-lvis_debug"
VALIDATION_STEPS=20

VALIDATION_IMAGE_1='./condition_val/epsilon_0_blackbg/000000286994.png'
VALIDATION_PROMPT_1='some elephants and one is by some water'
VALIDATION_IMAGE_4='./condition_val/lvisbbox/000000403385.png'
VALIDATION_RPOMPT_4='A bathroom that has a broken wall in the shower.'
VALIDATION_IMAGE_2='./condition_val/epsilon_0_blackbg/000000286994.png'
VALIDATION_PROMPT_2='some elephants and one is by some water'
VALIDATION_IMAGE_3='./condition_val/lvisbbox/000000403385.png'
VALIDATION_RPOMPT_3='A bathroom that has a broken wall in the shower.'

accelerate config default 

# accelerate launch train_controlnet.py \
python train_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
  --controlnet_model_name_or_path=$CONTROLNET_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --conditioning_mode=$CONDITIONING_MODE \
  --dilate_radius="random" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --dataloader_num_workers=0 \
  --checkpointing_steps=40 \
  --validation_steps=$VALIDATION_STEPS \
	--validation_image $VALIDATION_IMAGE_1 $VALIDATION_IMAGE_3 $VALIDATION_IMAGE_2 \
  --validation_prompt "${VALIDATION_PROMPT_1}" "${VALIDATION_RPOMPT_3}" "${VALIDATION_PROMPT_2}" \
  --train_batch_size=4 \
	--tracker_project_name=$TRACKER_PROJECT_NAME \
	--num_train_epochs=1 \
  --max_train_steps=100 \
  --proportion_empty_prompts=0.5 \
  --do_ratio_condition \
  --do_predict \
  --detach_feature \
  --low_cpu_mem_usage \
  --predictor_lr=1e-4

# Notations 
# --low_cpu_mem_usage: enable to load unstrictly matched param_dict 
# --controlnet_model_name_or_path: set pretrained controlnet parameter path 
# --report_to=wandb: enable log on wandb, remember to login with your account
# --batch_size: 4 for about 20G, and 24 for about 70G GPU Memory 