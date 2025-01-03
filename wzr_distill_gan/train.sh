# /home/user/wuzhirong/Projects/diffusers/wzr_distill_gan/sft_inject_key_fbf_fps10_distill_gan_fps1.py \
export CUDA_VISIBLE_DEVICES=3
WANDB__SERVICE_WAIT=300 accelerate launch \
    --main_process_port 25338 \
    --config_file /home/user/wuzhirong/Projects/diffusers/wzr_distill_gan/accelerate_config_machine_single.yaml \
    /home/user/wuzhirong/Projects/diffusers/wzr_distill_gan/sft_inject_key_fbf_fps10_distill_gan_fps1.py \
        --mixed_precision=bf16 --pretrained_model_name_or_path /data/wuzhirong/hf-models/CogVideoX-2b \
        --instance_data_root /data/wuzhirong/datasets/Nuscenes/ \
        --dataset_name nuscenes \
        --validation_prompt "go straight. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. ::: wait. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. ::: slow down. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. " \
        --validation_images /data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151671612404.jpg \
        --validation_prompt_separator ::: --num_validation_videos 1 \
        --validation_epochs 70 \
        --seed 42 \
        --tracker_name cogvideox-A4-clean-image-distill-gan \
        --output_dir /data/wuzhirong/ckpts/cogvideox-A4-clean-image-distill-gan-0101-new-4 \
        --height 480 --width 720 --fps 8 \
        --max_num_frames 145 \
        --train_batch_size 1 \
        --num_train_epochs 500 \
        --checkpointing_steps 70 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-6 --lr_scheduler cosine_with_restarts --lr_warmup_steps 10 --lr_num_cycles 1 \
        --optimizer Adam --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 --allow_tf32 --report_to wandb \
        --gradient_checkpointing \
        --denoised_image \
        --generator_update_ratio 5 \
        --distill_loss_weight 1.0 \
        --gen_cls_loss_weight 0.1 \
        --diffusion_loss_weight 1.0 \
        --gan_loss_weight 0.01
        # --enable_slicing --enable_tiling \