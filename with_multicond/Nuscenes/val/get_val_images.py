
from torch.utils.data import Dataset
import json
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode, resize
import os
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

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
sys.path.append("/root/PKU/diffusers/src")
from diffusers.utils import export_to_video
sys.path.append("../")
from nuscenes_dataset_for_cogvidx import NuscenesDatasetForCogvidx

if __name__ == "__main__":
    val_dataset = NuscenesDatasetForCogvidx(
        data_root="/root/autodl-fs/Nuscenes-v1.0-trainval-CAM_FRONT",
        split="val",
        preload_all_data=False)
    
    num_frames = 33
    fps = 2
    width = 720
    height = 480
    root_path = "/root/autodl-fs/nuscene_val"

    for i in tqdm(range(len(val_dataset))):
        scene_name = val_dataset.scenes[i]
        frames = [image2pil(path) for path in val_dataset.frames_group[scene_name][:num_frames]]
        frames = [img.resize((width, height)) for img in frames]
        path  = os.path.join(root_path, scene_name)
        os.makedirs(path, exist_ok=True)
        frames[0].save(os.path.join(path, "image0.jpg"))
        export_to_video(frames, os.path.join(path,"video.mp4"), fps=fps)
        scene_annotation = val_dataset.annotations[scene_name]
        annotation = scene_annotation["0"]
        driving_prompt = annotation.get("Driving action", "").strip()
        prompt_file = open(os.path.join(path, "prompt.txt"), "w")
        prompt_file.write(driving_prompt)
        prompt_file.close()