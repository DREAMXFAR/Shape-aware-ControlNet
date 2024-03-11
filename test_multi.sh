#/bin/bash
python test_multicontrolnet.py \
    --prompt "a dog is sitting on a skateboard" \
    --mask_img_path ./demo/multi_img/dog_mask.png \
    --mask_controlnet_path ./controlnet_checkpoints/docond_addpred_detach_checkpoint-46000/controlnet \
    --bbox_img_path ./demo/multi_img/skateboard_bbox.png \
    --bbox_controlnet_path ./controlnet_checkpoints/docond_addpred_detach_checkpoint-46000/controlnet \
    --output_path ./output_dir/multictrl_test 

### use the following argments to set deteriorate_ratio manually
# --mask_deteriorate_ratio 0.0 \
# --bbox_deteriorate_ratio 1.5 \