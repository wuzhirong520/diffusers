from PIL import Image,ImageDraw,ImageFont
import numpy as np
import cv2

def visualizeCondition(image:Image, trajectory_points)->Image:
    image_new = image.copy()
    image_draw = ImageDraw.Draw(image_new)
    w,h = 720,480
    color = (255,0,0)
    radius = 5
    scale = 10
    image_draw.circle((w/2,h-radius),radius,color)
    for i in range(len(trajectory_points)):
        x = w/2 + trajectory_points[i][0]*scale
        y = h - trajectory_points[i][1]*scale
        image_draw.circle((x,y),radius,color)
    return image_new

import os
from tqdm import tqdm
import sys
sys.path.append("../../src")
from diffusers.utils import export_to_video
from nuscenes_dataset import NuscenesDatasetForCogvidx
val_dataset = NuscenesDatasetForCogvidx("/data/wuzhirong/datasets/Nuscenes",split="val")

import torch
from diffusers import AutoencoderKLCogVideoX
from diffusers.video_processor import VideoProcessor
video_processor = VideoProcessor(vae_scale_factor=8)
import torchvision

pretrained_model_name_or_path = "/data/wuzhirong/hf-models/CogVideoX-2b"

# device = "cpu"
device = "cuda"

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )
vae.to(device)
# vae.enable_slicing()
# vae.enable_tiling()
print('Pipeline loaded!')

selected = [206,197,1041]

for i in tqdm(selected):
    scene = val_dataset.anno[i]
    video = [Image.open(os.path.join(val_dataset.data_root, path)).resize([720,480]) for path in scene["frames"][::6]]
    print(len(video))
    trajectory_points_x = scene["traj"][2::2]
    trajectory_points_y = scene["traj"][3::2]
    # for j in range(8):
    #     scene = val_dataset.anno[i+j+1]
    #     video.append(Image.open(os.path.join(val_dataset.data_root, scene["frames"][-1])).resize([720,480]))
    #     trajectory_points_x.append(scene["traj"][-2] - scene["traj"][-4] + trajectory_points_x[-1])
    #     trajectory_points_y.append(scene["traj"][-1] - scene["traj"][-3] + trajectory_points_y[-1])
    # print(len(video))

    trajectory_points = [[0,0]]
    for j in range(len(trajectory_points_x)):
        trajectory_points.append([trajectory_points_x[j],trajectory_points_y[j]])
    print(trajectory_points)
    raw_video = [visualizeCondition(v, trajectory_points) for i,v in enumerate(video)]
    # export_to_video(raw_video, os.path.join("./", f"raw_video_{i:05}.mp4"), fps=8)

    imgs = video_processor.preprocess(video, height=480, width=720).to(device=device,dtype=torch.float16) # [F,C,H,W]
    imgs = imgs.unsqueeze(0).permute(0, 2, 1 ,3,4) # [B, C, F, H, W]

    with torch.no_grad():

        latents = []
        for ii in tqdm(range(len(video))):
            latent = vae.encode(imgs[:,:,ii:ii+1]).latent_dist.sample()
            latents.append(latent)
        latents = torch.cat(latents, dim=2)

        from traj_utils_torch import get_trajectory_latent
        trajectory_points = torch.Tensor(trajectory_points).to(device=device)
        trajectory_points = torch.cat([trajectory_points[:,0:1], torch.zeros_like(trajectory_points[:,0:1]) , trajectory_points[:,1:]], dim=1)
        print(trajectory_points.shape)
        new_latents = latents[0,:,0].permute(1,2,0)
        new_latents, new_latents_mask = get_trajectory_latent(new_latents, trajectory_points)
        new_latents = new_latents.permute(3,0,1,2).unsqueeze(0)
        # latents += new_latents
        # latents[:,:,new_latents_mask] /= 2
        latents = new_latents

        rec_imgs = []
        for ii in tqdm(range(len(video))):
            rec_img = vae.decode(latents[:,:,ii:ii+1]).sample
            rec_imgs.append(rec_img)
        rec_imgs = torch.cat(rec_imgs,dim=2)
        print(rec_imgs.shape)
    
    new_video = video_processor.postprocess_video(video=rec_imgs, output_type="pil")[0]

    raw_video_tensor = torch.stack([torchvision.transforms.ToTensor()(v) for v in raw_video])
    new_video_tensor = torch.stack([torchvision.transforms.ToTensor()(v) for v in new_video])
    cat_video_tensor = torch.cat([raw_video_tensor,new_video_tensor],dim=3)
    cat_video = [torchvision.transforms.ToPILImage()(cat_video_tensor[index]) for index in range(cat_video_tensor.shape[0])]
    export_to_video(cat_video, os.path.join("./", f"cat_video_{i:05}_no_fill.mp4"), fps=8)