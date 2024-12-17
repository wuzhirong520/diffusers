CUDA_VISIBLE_DEVICES=0 \
    accelerate launch \
        --main_process_port 42351 \
        --config_file /home/user/wuzhirong/Projects/diffusers/wzr_fft/accelerate_config_machine_single.yaml \
    /home/user/wuzhirong/Projects/diffusers/wzr_fft/sft_inject_key_fbf_fps1_more-text.py \
        --mixed_precision=fp16 \
        --pretrained_model_name_or_path /data/wuzhirong/hf-models/CogVideoX-2b \
        --instance_data_root /data/wangxd/nuscenes/ \
        --dataset_name nuscenes \
        --validation_prompt "turn left. Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings. :::wait. Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings. :::turn right. Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings.:::go straight. Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings.:::slow down. Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings." \
        --validation_images /data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885320012463.jpg \
        --validation_prompt_separator ::: \
        --num_validation_videos 1 \
        --validation_epochs 3 \
        --seed 42 \
        --tracker_name cogvideo-A4-clean-image-inject-sft-f13-fps1-1205-fbf-fft01 \
        --output_dir /data/wuzhirong/ckpts/cogvideox-A4-clean-image-inject-f13-fps1-1205-fbf-inherit1022-fft01 \
        --height 480 --width 720 --fps 8 --max_num_frames 13 \
        --train_batch_size 1 \
        --num_train_epochs 500 \
        --checkpointing_steps 1000 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --lr_scheduler cosine_with_restarts \
        --lr_warmup_steps 100 \
        --lr_num_cycles 1 \
        --enable_slicing --enable_tiling \
        --optimizer Adam --adam_beta1 0.9 --adam_beta2 0.95 \
        --max_grad_norm 1.0 --allow_tf32 --report wandb \
        --gradient_checkpointing