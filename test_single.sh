#/bin/bash

python test_controlnet.py \
    --img_path ./demo/single_img/000000355905_d20.png \
    --prompt "A white dog is on a sandy beach while the sea foam washes ashore behind it." \
    --controlnet_path ./controlnet_checkpoints/docond_addpred_detach_checkpoint-46000/controlnet \
    --output_path ./output_dir/inference_demo
