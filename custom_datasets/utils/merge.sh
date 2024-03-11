#! /bin/bash

ROOT=/dat03/xuanwenjie/datasets/LVIS_COCO_triplet
MERGE_FILE_1=${ROOT}/cleaned_files/img_anns_cap_train_filtered.jsonl
MERGE_FILE_2=${ROOT}/cleaned_files/img_anns_cap_val_subtrain_filtered.jsonl

SAVE_PATH=${ROOT}/train.jsonl

python merge_jsonls.py --files $MERGE_FILE_1 $MERGE_FILE_2 --save_path $SAVE_PATH
