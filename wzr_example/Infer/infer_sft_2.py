import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
# os.environ['HUGGINGFACE_HUB_CACHE']="/root/autodl-fs/huggingface/hub"
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
        # "/root/autodl-fs/cogvideox-D4-clean-image-sft/1022",
        "/root/PKU/ckpt",
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

"""
load 5B-I2V baseline pipe
"""
# pipe = CogVideoXImageToVideoPipeline.from_pretrained(
#     "THUDM/CogVideoX-5b-I2V",
#     torch_dtype=torch.bfloat16
# )
# from diffusers import CogVideoXPipeline
# pipe = CogVideoXPipeline.from_pretrained(
#     "THUDM/CogVideoX-2b",
#     torch_dtype=torch.bfloat16
# )

pipe.to("cuda")
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

print('Pipeline loaded!')


import decord
import PIL.Image
decord.bridge.set_bridge("torch")

# video_path = "/root/PKU/diffusers/wzr_example/Infer/val_campus/go straight-9.mp4"
# image = decord.VideoReader(video_path, width=720, height=480).get_batch([-1]).squeeze(0)
# image = PIL.Image.fromarray(image.numpy())

image = load_image("/root/PKU/campus/9.jpg")
prompt = "go straight"

videos = []
for i in range(5):
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
    image = frames[-1]
    if i==0: 
        videos += frames
    else:
        videos += frames[1:]

export_to_video(videos,"go straight-9-more.mp4" , fps=8)