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

# pretrained_model_name_or_path = "THUDM/CogVideoX-2b"

# tokenizer = AutoTokenizer.from_pretrained(
#         pretrained_model_name_or_path, subfolder="tokenizer", revision=None, 
#         torch_dtype=torch.float16,
#     )

# text_encoder = T5EncoderModel.from_pretrained(
#     pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
#     torch_dtype=torch.float16,
# )

# transformer = CogVideoXTransformer3DModel.from_pretrained(
#         "/root/autodl-fs/cogvideox-D4-clean-image-sft/1022",
#         torch_dtype=torch.float16,
#         revision=None,
#         variant=None,
#         # in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
#     )

# vae = AutoencoderKLCogVideoX.from_pretrained(
#         pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
#         torch_dtype=torch.float16,
#     )

# scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",)

# components = {
#             "transformer": transformer,
#             "vae": vae,
#             "scheduler": scheduler,
#             "text_encoder": text_encoder,
#             "tokenizer": tokenizer,
#         }

# pipe = CogVideoXImageToVideoPipeline(**components)

"""
load 5B-I2V baseline pipe
"""
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16
)

# pipe.to("cuda")
pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

print('Pipeline loaded!')

# prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
# image = load_image("/root/PKU/diffusers/wzr_example/astronaut.jpg")
# video = pipe(image, prompt, guidance_scale=6, use_dynamic_cfg=True, num_frames=49)
# export_to_video(video.frames[0], "output.mp4", fps=8)

# while True:
#     image_path = input("image_path: ")
#     validation_prompt = input("prompt: ")
#     guidance_scale = input("cfg: ") # 6
#     pipeline_args = {
#         "image": load_image(image_path),
#         "prompt": validation_prompt,
#         "guidance_scale": int(guidance_scale),
#         "use_dynamic_cfg": True,
#         "height": 480,
#         "width": 720,
#         "num_frames": 33
#     }
#     name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
#     frames = pipe(**pipeline_args).frames[0]
#     export_to_video(frames, f"{name_prefix}_cfg_{guidance_scale}.mp4", fps=8)


'''
infer vista demos
'''
import decord
import PIL.Image
decord.bridge.set_bridge("torch")

# vista_actions = ["right","left","stop","forward", "right", "left"]
# actions = ["turn right","turn left","wait","go straight", "sharp right turn","sharp left turn"]
# vista_actions = ["left", "left"]
# actions = ["turn left", "sharp left turn"]
vista_actions = ["left","left","left","right","right","right","forward","forward","forward","forward","forward","forward","stop"]
actions = ["shift slightly to the left",
           "sharp left turn",
           "follow the road left",

           "shift slightly to the right",
           "sharp right turn",
           "follow the road right",

           "go straight",
           "maintain speed",
           "speed up",
           "slow down",
           "drive fast",
           "drive slowly",
           "wait"
           ]
demos_per_action = 10
vista_demo_path = "/root/autodl-fs/vista_demos"
infer_path = "./val_5B_I2V"

os.makedirs(infer_path, exist_ok=True)

for video_id in range(1, demos_per_action+1):
    if video_id!=3:
        continue
    for action_id in range(len(actions)) :
        save_path = os.path.join(infer_path, f"{actions[action_id]}-{video_id}.mp4")
        if os.path.exists(save_path):
            continue
        vista_video_path = os.path.join(vista_demo_path, f"{vista_actions[action_id]}-{video_id}.mp4")
        image = decord.VideoReader(vista_video_path, width=720, height=480).get_batch([0]).squeeze(0)
        image = PIL.Image.fromarray(image.numpy())
        prompt = actions[action_id]
        pipeline_args = {
            "image": image,
            "prompt": prompt,
            "guidance_scale": 6,
            "use_dynamic_cfg": True,
            "height": 480,
            "width": 720,
            "num_frames": 49
        }
        frames = pipe(**pipeline_args).frames[0]
        export_to_video(frames,save_path , fps=8)


'''
infer nuscenes
'''
from PIL import Image
import numpy as np
def image2pil(filename):
    return Image.open(filename)

def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)

def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in pil])
    else:
        arr = np.array(pil)
    return arr       
import sys
sys.path.append("../Nuscenes")
from nuscenes_dataset_for_cogvidx import NuscenesDatasetForCogvidx

val_dataset = NuscenesDatasetForCogvidx(
        data_root="/root/autodl-fs/Nuscenes-v1.0-trainval-CAM_FRONT",
        split="val",
        preload_all_data=False)

infer_path = "./val_5B_I2V"

os.makedirs(infer_path, exist_ok=True)

import json
with open("/root/PKU/diffusers/wzr_example/Infer/nusc_action_val.json", 'r') as f:
    action_annotations = json.load(f)

num_frames = 33
width = 720
height = 480

for i in range(len(val_dataset)):
    scene_name = val_dataset.scenes[i]
    if scene_name!='scene-0016':
        continue
    print(scene_name)
    frames = [image2pil(path) for path in val_dataset.frames_group[scene_name][:num_frames]]
    frames = [img.resize((width, height)) for img in frames]

    image = frames[0]
    
    scene_annotation = val_dataset.annotations[scene_name]
    annotation = scene_annotation["0"]
    caption = ""
    selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
    for attr in selected_attr:
        anno = annotation.get(attr, "")
        if anno == "":
            continue
        if anno[-1] == ".":
            anno = anno[:-1] 
        caption = caption + anno + ". "
    
    prompt = action_annotations[scene_name]["0"] + ". " + caption
    
    # actions = ["turn right","turn left","wait","go straight", "sharp right turn","sharp left turn"]
    actions = ["shift slightly to the left",
           "sharp left turn",
           "follow the road left",

           "shift slightly to the right",
           "sharp right turn",
           "follow the road right",

           "go straight",
           "maintain speed",
           "speed up",
           "slow down",
           "drive fast",
           "drive slowly",
           "wait"
           ]
    
    for prompt in actions:

        print(scene_name, prompt)

        name_prefix = prompt.replace(" ", "_").strip()[:40]
        save_path = os.path.join(infer_path, f"{scene_name}_{name_prefix}.mp4")
        if os.path.exists(save_path):
            continue

        pipeline_args = {
            "image": image,
            "prompt": prompt,
            "guidance_scale": 6,
            "use_dynamic_cfg": True,
            "height": 480,
            "width": 720,
            "num_frames": 49
        }
        frames = pipe(**pipeline_args).frames[0]

        export_to_video(frames, save_path , fps=8)
    