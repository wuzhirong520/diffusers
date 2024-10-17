# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE']="/root/autodl-fs/huggingface/hub"
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

import sys
sys.path.append("/root/PKU/diffusers/src")
import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
import numpy as np
from diffusers.image_processor import VaeImageProcessor

from load_cogvidx_2b_i2v_fintuned import LoadCogvidx2BI2VFintuned
import PIL.Image
import decord
decord.bridge.set_bridge("torch")

ckpt_folder = "/root/autodl-tmp/cogvideox-lora-single-node_test_full_withembedtrain_fiximg/checkpoint-18900"
val_folder_root = "/root/autodl-fs/nuscene_val"
output_folder = "/root/PKU/diffusers/wzr_example/Nuscenes/val/val_videos"

pipe = LoadCogvidx2BI2VFintuned(ckpt_folder)

for scene in sorted(os.listdir(val_folder_root)):

    val_path = os.path.join(val_folder_root, scene)

    val_img = load_image(os.path.join(val_path, "image0.jpg"))
    val_video = decord.VideoReader(os.path.join(val_path, "video.mp4"), width=720, height=480).get_batch(list(range(0,33)))
    val_prompt = open(os.path.join(val_path, "prompt.txt")).read()

    pipeline_args = {
        "image": val_img,
        "prompt": val_prompt,
        "guidance_scale": 6,
        "use_dynamic_cfg": False,
        "height": 480,
        "width": 720,
        "num_frames": 33
    }

    with torch.no_grad():
        gen_video = pipe(**pipeline_args, output_type="pt").frames[0].to(val_video.device)
        
        pt_images = torch.cat([val_video,gen_video],dim=2)
        pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])
        image_np = VaeImageProcessor.pt_to_numpy(pt_images)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)

    filename = os.path.join(output_folder, f"{scene}.mp4")

    export_to_video(image_pil, filename, fps=2)