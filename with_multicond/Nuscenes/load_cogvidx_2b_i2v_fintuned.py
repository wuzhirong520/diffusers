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

import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE']="/root/autodl-fs/huggingface/hub"
import sys
sys.path.append("/root/PKU/diffusers/src")
import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
import torch

# from safetensors.torch import load_file
# p = load_file("/root/autodl-tmp/cogvideox-lora-single-node_test_full_withembedtrain/checkpoint-10/pytorch_lora_weights.safetensors")
# print(p.keys())
# exit(0)

def LoadCogvidx2BI2VFintuned(lora_weight_path):

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-2b",
        subfolder="transformer",
        torch_dtype=torch.float16,
        in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16,
        transformer=transformer
    )
    # ).to("cuda")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config,)

    # Load LoRA weights
    lora_alpha = 64
    rank = 128
    lora_scaling = lora_alpha / rank
    pipe.load_lora_weights(lora_weight_path, adapter_name="cogvideox-i2v-lora")
    pipe.set_adapters(["cogvideox-i2v-lora"], [lora_scaling])

    from safetensors.torch import load_file
    transformer_patch_embed_proj = load_file(os.path.join(lora_weight_path,"transformer_patch_embed_proj.safetensors"))
    transformer.patch_embed.proj.weight.data = transformer_patch_embed_proj['transformer.patch_embed.proj.weight'].to(torch.float16)
    transformer.patch_embed.proj.bias.data = transformer_patch_embed_proj['transformer.patch_embed.proj.bias'].to(torch.float16)

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    return pipe