import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import sys
sys.path.append("../src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    # CogVideoXImageToVideoPipeline,
    # CogVideoXTransformer3DModel,
)

from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_cond2 import CogVideoXImageToVideoPipeline
from diffusers.models.transformers.cogvideox_transformer_3d_traj_2 import CogVideoXTransformer3DModel

from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
import numpy as np
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
        # "/data/wuzhirong/ckpts/cogvideox-D4-clean-image-sft-1022/transformer",
        # "/data/wuzhirong/ckpts/cogvideox-25f-traj/checkpoint-7800/transformer",
        "/data/wuzhirong/ckpts/cogvideox-25f-traj_2/checkpoint-1200/transformer",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
    )

# from safetensors import safe_open
# tensors = {}
# with safe_open(os.path.join("/data/wuzhirong/hf-models/CogVideoX-2b/transformer", "diffusion_pytorch_model.safetensors"), framework="pt", device='cpu') as f:
#     for k in f.keys():
#         if "patch_embed.proj" in k:
#             nk = k.replace("proj", "origin_proj")
#             tensors[nk] = f.get_tensor(k)
#             print(k)

# transformer.load_state_dict(tensors, strict=False)

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
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()
# from load_pipeline import LoadPipelineLora
# pipe = LoadPipelineLora("")
print('Pipeline loaded!')

from nuscenes_dataset import NuscenesDatasetForCogvidx
val_dataset = NuscenesDatasetForCogvidx("/data/wuzhirong/datasets/Nuscenes",split="val")
from PIL import Image,ImageDraw,ImageFont

def visualizeCondition(image:Image, cond_dict:dict, index:int)->Image:
    image_new = image.copy()
    image_draw = ImageDraw.Draw(image_new)
    w,h = 720,480
    if "trajectory" in cond_dict.keys():
        color = (255,0,0)
        radius = 8
        scale = 15
        image_draw.circle((w/2,h-radius),radius,color)
        for i in range(4):
            x = w/2 + cond_dict["trajectory"][i*2]*scale
            y = h - cond_dict["trajectory"][i*2+1]*scale
            image_draw.circle((x,y),radius,color)
    if "goal" in cond_dict.keys():
        x = cond_dict["goal"][0]*w
        y = cond_dict["goal"][1]*h
        image_draw.circle((x,y),12,(0,255,0))
    if "speed" in cond_dict.keys():
        speed = cond_dict["speed"][index//6 if index<24 else 3] 
        image_draw.text((20,20),f"speed : {speed}",(0,0,255),ImageFont.load_default(30))
    if "angle" in cond_dict.keys():
        angle = cond_dict["angle"][index//6 if index<24 else 3]
        image_draw.text((20,70),f"angle : {angle:.3f}",(0,0,255),ImageFont.load_default(30))
    return image_new

seleced = [206,683,1041,0,]

output_dir = "./infer_2_results_1200"
os.makedirs(output_dir,exist_ok=True)

for index in seleced:
    scene = val_dataset.anno[index]

    video = [Image.open(os.path.join(val_dataset.data_root, path)).resize([720,480]) for path in scene["frames"]]
    
    if scene["cmd"]==0:
        prompt = "turn right"
    elif scene["cmd"]==1:
        prompt = "turn left"
    elif scene["cmd"]==2:
        prompt = "wait"
    elif scene["cmd"]==3:
        prompt = "go straight"

    traj = np.array(scene["traj"][2:]).reshape(-1,2)

    print(index, prompt, traj)

    pipeline_args = {
        "image": video[0],
        "prompt": prompt,
        "trajectory": traj,
        "guidance_scale": 6,
        "use_dynamic_cfg": True,
        "height": 480,
        "width": 720,
        "num_frames": 25,
        "num_inference_steps": 50,
    }
    frames = pipe(**pipeline_args).frames[0]

    cond_dict={"trajectory": scene["traj"][2:]}

    gen_video = [visualizeCondition(v,cond_dict,i) for i,v in enumerate(frames)]
    raw_video = [visualizeCondition(v,cond_dict,i) for i,v in enumerate(video)]
    video_tensor = torch.stack([torch.tensor(np.array(v)) for v in raw_video])
    gen_video_tensor = torch.stack([torch.tensor(np.array(p)) for p in gen_video])
    cat_video_tensor = torch.cat([gen_video_tensor,video_tensor],dim=2)
    cat_video = [Image.fromarray(cat_video_tensor[idx].numpy()) for idx in range(cat_video_tensor.shape[0])]

    export_to_video(cat_video, os.path.join(output_dir, f"{index}.mp4") , fps=10)

    if scene["cmd"]==0:
        prompt = "turn left" # inv
    elif scene["cmd"]==1:
        prompt = "turn right" # inv
    elif scene["cmd"]==2:
        continue
    elif scene["cmd"]==3:
        prompt = "go straight"

    traj[:,0] = -traj[:,0]

    print(index, prompt, traj)

    pipeline_args = {
        "image": video[0],
        "prompt": prompt,
        "trajectory": traj,
        "guidance_scale": 6,
        "use_dynamic_cfg": True,
        "height": 480,
        "width": 720,
        "num_frames": 25,
        "num_inference_steps": 50,
    }
    frames = pipe(**pipeline_args).frames[0]

    cond_dict={"trajectory": traj.reshape(-1)}

    gen_video = [visualizeCondition(v,cond_dict,i) for i,v in enumerate(frames)]
    raw_video = [visualizeCondition(v,cond_dict,i) for i,v in enumerate(video)]
    video_tensor = torch.stack([torch.tensor(np.array(v)) for v in raw_video])
    gen_video_tensor = torch.stack([torch.tensor(np.array(p)) for p in gen_video])
    cat_video_tensor = torch.cat([gen_video_tensor,video_tensor],dim=2)
    cat_video = [Image.fromarray(cat_video_tensor[idx].numpy()) for idx in range(cat_video_tensor.shape[0])]

    export_to_video(cat_video, os.path.join(output_dir, f"{index}_inv.mp4") , fps=10)