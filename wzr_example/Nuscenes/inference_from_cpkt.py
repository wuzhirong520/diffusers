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

from nuscenes_dataset_for_cogvidx import NuscenesDatasetForCogvidx

# from safetensors.torch import load_file
# p = load_file("/root/autodl-tmp/cogvideox-lora-single-node_test_full_withembedtrain/checkpoint-10/pytorch_lora_weights.safetensors")
# print(p.keys())
# exit(0)

ckpt_folder_root = "/root/autodl-tmp/cogvideox-lora-single-node_test_full_withembedtrain"
# validation_prompt = "The ego car is moving forward slowly, approaching the barrier gate."
# validation_image = "/root/PKU/diffusers/wzr_example/Nuscenes/val/scene-0003/0.jpg"
validation_prompt = "The ego car moves forward at a steady pace, occasionally shifting slightly to the left and right as it navigates the curve of the road."
validation_image = "/root/PKU/diffusers/wzr_example/Nuscenes/val/scene-0012/0.jpg"

for ckpt_path in sorted(os.listdir(ckpt_folder_root),reverse=True):

    ckpt_folder = os.path.join(ckpt_folder_root, ckpt_path)
    if not os.path.isdir(ckpt_folder) :
        continue
    print(ckpt_folder)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "/root/autodl-fs/CogVidx-2b-I2V-base-transfomer",
        torch_dtype=torch.float16,
    )

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16,
        transformer = transformer
    )
    # ).to("cuda")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config,)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Load LoRA weights
    lora_alpha = 64
    rank = 128
    lora_scaling = lora_alpha / rank
    pipe.load_lora_weights(ckpt_folder, adapter_name="cogvideox-i2v-lora")
    pipe.set_adapters(["cogvideox-i2v-lora"], [lora_scaling])

    from safetensors.torch import load_file
    transformer_patch_embed_proj = load_file(os.path.join(ckpt_folder,"transformer_patch_embed_proj.safetensors"))
    transformer.patch_embed.proj.weight.data = transformer_patch_embed_proj['transformer.patch_embed.proj.weight'].to(torch.float16)
    transformer.patch_embed.proj.bias.data = transformer_patch_embed_proj['transformer.patch_embed.proj.bias'].to(torch.float16)

    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()

    pipeline_args = {
        "image": load_image(validation_image),
        "prompt": validation_prompt,
        "guidance_scale": 6,
        "use_dynamic_cfg": False,
        "height": 480,
        "width": 720,
        "num_frames": 33
    }

    seed=42

    generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

    with torch.no_grad():
        pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[0]
    pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])

    image_np = VaeImageProcessor.pt_to_numpy(pt_images)
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)

    filename = os.path.join(ckpt_folder, "validation.mp4")

    export_to_video(image_pil, filename, fps=2)