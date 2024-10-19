export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME="/root/autodl-fs/huggingface/"
export HUGGINGFACE_HUB_CACHE="/root/autodl-fs/huggingface/hub"
export TOKENIZERS_PARALLELISM=false

export MODEL_PATH="THUDM/CogVideoX-2b"
export DATASET_PATH="/root/autodl-fs/Nuscenes-v1.0-trainval-CAM_FRONT"
export OUTPUT_PATH="/root/autodl-fs/cogvideox-lora-single-node_test_full_withembedtrain_test"
export CUDA_VISIBLE_DEVICES=0,1

  # --resume_from_checkpoint "/root/autodl-tmp/cogvideox-lora-single-node_test_from2000/checkpoint-2000" \

  # --validation_prompt "The ego car speeds up, then slows down as it approaches a red traffic light. It stops at the traffic light, waits for it to turn green, and then continues driving forward." \
  # --validation_images "/root/PKU/diffusers/wzr_example/Nuscenes/val/scene-0001/0.jpg" \

#--multi_gpu
accelerate launch --config_file accelerate_config_machine_single.yaml  \
  train.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --validation_prompt "The ego car moves forward at a steady pace, occasionally shifting slightly to the left and right as it navigates the curve of the road." \
  --validation_images "/root/PKU/diffusers/wzr_example/Nuscenes/val/scene-0012/0.jpg" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 1 \
  --seed 42 \
  --rank 128 \
  --lora_alpha 64 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 33 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --num_train_epochs 80 \
  --checkpointing_steps 350 \
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
  --report_to wandb \
  --mixed_precision fp16 \
  # --resume_from_checkpoint "/root/autodl-tmp/cogvideox-lora-single-node_test_full_withembedtrain/checkpoint-10"