import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE']="/root/autodl-fs/huggingface/hub"
import sys
sys.path.append("/root/PKU/diffusers/src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file

"""
load pipe for 2B finetuned
"""

pretrained_model_name_or_path = "THUDM/CogVideoX-2b"

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=None, 
        torch_dtype=torch.float16,
    )

text_encoder = T5EncoderModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
    torch_dtype=torch.float16,
)

transformer = CogVideoXTransformer3DModel.from_pretrained(
        "/root/autodl-fs/cogvideox-D4-clean-image-sft/1022",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        # in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )

scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",)

components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

pipe = CogVideoXImageToVideoPipeline(**components)

pipe.to("cuda")
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

print('Pipeline loaded!')


image = load_image("/root/PKU/campus/9.jpg")
prompt = "go straight"

pipeline_args = {
    "image": image,
    "prompt": prompt,
    "guidance_scale": 6,
    "use_dynamic_cfg": True,
    "height": 480,
    "width": 720,
    "num_frames": 33
}
frames = pipe(**pipeline_args).frames[0]

export_to_video(frames,"infer.mp4" , fps=8)