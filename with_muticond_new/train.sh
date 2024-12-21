CUDA_VISIBLE_DEVICES=3 \
    accelerate launch \
        --main_process_port 42352 \
        --config_file /home/user/wuzhirong/Projects/diffusers/with_muticond_new/accelerate_config_machine_single.yaml \
    /home/user/wuzhirong/Projects/diffusers/with_muticond_new/sft_inject_key_fbf_fps1_more-text_traj.py \
        --mixed_precision=fp16 \
        --pretrained_model_name_or_path /data/wuzhirong/hf-models/CogVideoX-2b \
        --instance_data_root /data/wuzhirong/datasets/Nuscenes \
        --dataset_name nuscenes \
        --validation_prompt "Turn right, then go straight and speed up fast.. Cloudy. Daytime. An intersection with multiple lanes, traffic lights, and various vehicles. Traffic lights, vehicles (including a large truck), and the road markings. The scene then changes: Urban street with multiple lanes, buildings, and traffic lights. Traffic lights, vehicles (including a large truck), buildings. The scene then changes: Urban, with buildings on both sides of the road, a bicycle lane marked on the street, and a fire hydrant on the sidewalk. Buildings, road markings, fire hydrant." \
        --validation_images /data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151182512404.jpg \
        --validation_prompt_separator ::: \
        --num_validation_videos 1 \
        --validation_epochs 3 \
        --seed 42 \
        --tracker_name cogvideo-A4-clean-image-inject-sft-f13-fps1-more-text-traj \
        --output_dir /data/wuzhirong/ckpts/cogvideo-A4-clean-image-inject-sft-f13-fps1-more-text-traj \
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