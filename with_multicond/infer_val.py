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

import PIL.ImageFont
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
# os.environ['HUGGINGFACE_HUB_CACHE']="/root/autodl-fs/huggingface/hub"
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
# from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

import sys
sys.path.append("../src")
import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    # CogVideoXImageToVideoPipeline,
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
import torchvision

import PIL.Image
import PIL.ImageDraw
import decord
decord.bridge.set_bridge("torch")

output_folder = "/data/wuzhirong/val/cond_sft_test-1/checkpoint-7600-cfg"
os.makedirs(output_folder,exist_ok=True)

from load_pipeline import LoadPipeline
pipe = LoadPipeline("/data/wuzhirong/ckpts/cogvideox-sft-test-1/checkpoint-7600/transformer").to("cuda")
print('Pipeline loaded!')

from nuscenes_dataset import NuscenesDatasetForCogvidx
val_dataset = NuscenesDatasetForCogvidx("/data/wuzhirong/datasets/Nuscenes",split="val")

# for i in tqdm(range(0, len(val_dataset))):
#     val_dataset.anno[i]['index']=i
# import json
# with open("/data/wuzhirong/val/val_gt/val.json","w") as f:
#     json.dump(val_dataset.anno, f, indent=4, separators=(",", ": "), ensure_ascii=False)
# exit(0)

def visualizeCondition(image:PIL.Image, cond_dict:dict, index:int)->PIL.Image:
    image_new = image.copy()
    image_draw = PIL.ImageDraw.Draw(image_new)
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
        image_draw.text((20,20),f"speed : {speed}",(0,0,255),PIL.ImageFont.load_default(30))
    if "angle" in cond_dict.keys():
        angle = cond_dict["angle"][index//6 if index<24 else 3]
        image_draw.text((20,70),f"angle : {angle:.3f}",(0,0,255),PIL.ImageFont.load_default(30))
    return image_new

selected = [206,683,1041,0]

for i in tqdm(selected):

    scene = val_dataset.anno[i]
    
    video = [PIL.Image.open(os.path.join(val_dataset.data_root, path)).resize([720,480]) for path in scene["frames"]]
    
    # with open(os.path.join(output_folder,f"prompt_{i:05}.txt"),"w") as f:
    #     f.write(prompt+"\n")

    for action_mod in range(5):
        cond_dict = {}
        cond_name = ""
        prompt = ""

        if scene["cmd"]==0:
            prompt = "sharp right turn"
        elif scene["cmd"]==1:
            prompt = "sharp left turn"
        elif scene["cmd"]==2:
            prompt = "wait"
        elif scene["cmd"]==3:
            prompt = "go straight"
        else:
            raise ValueError
        
        if action_mod == 0:
            cond_name+="trajectory"
            cond_dict["trajectory"] = scene["traj"][2:]
        elif action_mod == 1:
            if scene["speed"]:
                cond_name+="speed"
                cond_dict["speed"] = scene["speed"][1:]
            if scene["angle"]:
                cond_name+="angle"
                cond_dict["angle"] = list(np.array(scene["angle"][1:]) / 780)
        elif action_mod == 2:
            if scene["z"] > 0 and 0 < scene["goal"][0] < 1600 and 0 < scene["goal"][1] < 900:
                cond_name+="goal"
                cond_dict["goal"] = [
                    scene["goal"][0] / 1600,
                    scene["goal"][1] / 900
                ]
        elif action_mod == 3:
            # if scene["cmd"]==0:
            #     prompt = "sharp right turn"
            # elif scene["cmd"]==1:
            #     prompt = "sharp left turn"
            # elif scene["cmd"]==2:
            #     prompt = "wait"
            # elif scene["cmd"]==3:
            #     prompt = "go straight"
            # else:
            #     raise ValueError
            cond_name="prompt"
        elif action_mod == 4:
            prompt=""
            cond_name="none"

        if cond_name=="":
            continue
    
        
        # video_new = [visualizeCondition(v,cond_dict,i) for i,v in enumerate(video)]
        # export_to_video(video_new, os.path.join(output_folder, f"test_video_{i:05}_{cond_name}.mp4"), fps=10)
        # continue
                

        # with open(os.path.join(output_folder,f"prompt_{i:05}.txt"),"a") as f:
        #     f.write(cond+"\n")
        # continue

        cond = cond_dict.__str__().replace("'","\"")
        pipeline_args = {
            "image": video[0],
            "prompt": prompt,
            "cond":cond,
            "guidance_scale": 0,
            "use_dynamic_cfg": False,
            "height": 480,
            "width": 720,
            "num_frames": 25
        }
        with torch.no_grad():
            gen_video = pipe(**pipeline_args).frames[0]
            gen_video = [visualizeCondition(v,cond_dict,i) for i,v in enumerate(gen_video)]
            raw_video = [visualizeCondition(v,cond_dict,i) for i,v in enumerate(video)]
            video_tensor = torch.stack([torch.tensor(np.array(v)) for v in raw_video])

            gen_video_tensor = torch.stack([torch.tensor(np.array(p)) for p in gen_video])
            cat_video_tensor = torch.cat([gen_video_tensor,video_tensor],dim=2)
            cat_video = [PIL.Image.fromarray(cat_video_tensor[idx].numpy()) for idx in range(cat_video_tensor.shape[0])]

            export_to_video(cat_video, os.path.join(output_folder, f"video_{i:05}_{cond_name}.mp4"), fps=10)

        if action_mod<4:
            if action_mod==0:
                cond_dict["trajectory"][0]*=-1
                cond_dict["trajectory"][2]*=-1
                cond_dict["trajectory"][4]*=-1
                cond_dict["trajectory"][6]*=-1
            elif action_mod==1:
                cond_dict["angle"][0]*=-1
                cond_dict["angle"][1]*=-1
                cond_dict["angle"][2]*=-1
                cond_dict["angle"][3]*=-1
            elif action_mod==2:
                cond_dict["goal"][0]=1-cond_dict["goal"][0]
            elif action_mod==3:
                if prompt=="sharp left turn":
                    prompt="sharp right turn"
                elif prompt=="sharp right turn":
                    prompt="sharp left turn"
                else:
                    continue
            cond = cond_dict.__str__().replace("'","\"")
            pipeline_args = {
                "image": video[0],
                "prompt": prompt,
                "cond":cond,
                "guidance_scale": 0,
                "use_dynamic_cfg": False,
                "height": 480,
                "width": 720,
                "num_frames": 25
            }
            with torch.no_grad():
                gen_video = pipe(**pipeline_args).frames[0]
                gen_video = [visualizeCondition(v,cond_dict,i) for i,v in enumerate(gen_video)]
                raw_video = [visualizeCondition(v,cond_dict,i) for i,v in enumerate(video)]
                video_tensor = torch.stack([torch.tensor(np.array(v)) for v in raw_video])

                gen_video_tensor = torch.stack([torch.tensor(np.array(p)) for p in gen_video])
                cat_video_tensor = torch.cat([gen_video_tensor,video_tensor],dim=2)
                cat_video = [PIL.Image.fromarray(cat_video_tensor[idx].numpy()) for idx in range(cat_video_tensor.shape[0])]

                export_to_video(cat_video, os.path.join(output_folder, f"video_{i:05}_{cond_name}_inv.mp4"), fps=10)