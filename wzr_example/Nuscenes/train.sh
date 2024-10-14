export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME="/root/autodl-fs/huggingface/"
export HUGGINGFACE_HUB_CACHE="/root/autodl-fs/huggingface/hub"
export TOKENIZERS_PARALLELISM=false

export MODEL_PATH="THUDM/CogVideoX-2b"
export DATASET_PATH="/root/autodl-fs/Nuscenes-v1.0-trainval-CAM_FRONT"
export OUTPUT_PATH="/root/autodl-tmp/cogvideox-lora-single-node_test"
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
  train.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --validation_prompt "The ego car is moving forward slowly, approaching the barrier gate." \
  --validation_images "/root/PKU/diffusers/wzr_example/Nuscenes/val/scene-0003/0.jpg" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 2 \
  --seed 42 \
  --rank 128 \
  --lora_alpha 64 \
  --mixed_precision fp16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 9 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --num_train_epochs 4 \
  --checkpointing_steps 10 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb