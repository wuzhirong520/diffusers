import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import sys
sys.path.append("../src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    # CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_cond2 import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file

# # pretrained_model_name_or_path = "THUDM/CogVideoX-2b"
pretrained_model_name_or_path = "/data/wuzhirong/hf-models/CogVideoX-2b"

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
        "/data/wuzhirong/ckpts/cogvideox-D4-clean-image-sft-1022/transformer",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
    )

from safetensors import safe_open
tensors = {}
with safe_open(os.path.join("/data/wuzhirong/hf-models/CogVideoX-2b/transformer", "diffusion_pytorch_model.safetensors"), framework="pt", device='cpu') as f:
    for k in f.keys():
        if "patch_embed.proj" in k:
            nk = k.replace("proj", "origin_proj")
            tensors[nk] = f.get_tensor(k)
            print(k)

transformer.load_state_dict(tensors, strict=False)

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
# from load_pipeline import LoadPipelineLora
# pipe = LoadPipelineLora("")
# print('Pipeline loaded!')


image = load_image("/data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984233512470.jpg")
prompt = "sharp right turn"
import numpy as np
traj = np.array([
            -2.2737367544323206e-13,
            2.2737367544323206e-13,
            0.3695570238021446,
            2.00443961224164,
            1.0896928038487204,
            3.949010497127574,
            1.9147848224356494,
            5.358738002764312,
            3.104858568277905,
            6.974434312025778
        ]).reshape(-1,2)

pipeline_args = {
    "image": image,
    "prompt": prompt,
    "trajectory": traj,
    "guidance_scale": 6,
    "use_dynamic_cfg": True,
    "height": 480,
    "width": 720,
    "num_frames": 25,
    "num_inference_steps": 2,
}
frames = pipe(**pipeline_args).frames[0]

export_to_video(frames,"infer.mp4" , fps=10)