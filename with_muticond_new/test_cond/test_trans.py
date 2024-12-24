from PIL import Image,ImageDraw,ImageFont
import numpy as np
import cv2
import torch

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

selected = [197, 206, 1041]

for i in tqdm(selected):
    scene = val_dataset.anno[i]
    video = [Image.open(os.path.join(val_dataset.data_root, path)).resize([720,480]) for path in scene["frames"]]
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
    
    from traj_utils_torch import get_trajectory_image_pil, interpolate_trajectory
    trajectory_points = np.array(trajectory_points)
    print(trajectory_points.shape)
    new_trajectory_points = interpolate_trajectory(trajectory_points, 25)
    print(new_trajectory_points.shape)
    trajectory_points = new_trajectory_points
    trajectory_points = np.concatenate([trajectory_points[:,0:1], np.zeros_like(trajectory_points[:,0:1]) , trajectory_points[:,1:]], axis=1)
    print(trajectory_points.shape)
    new_video = get_trajectory_image_pil(video[0], torch.Tensor(trajectory_points))
    # export_to_video(new_video, os.path.join("./", f"new_video_{i:05}.mp4"), fps=8)

    import torchvision
    import torch
    raw_video_tensor = torch.stack([torchvision.transforms.ToTensor()(v) for v in raw_video])
    new_video_tensor = torch.stack([torchvision.transforms.ToTensor()(v) for v in new_video])
    cat_video_tensor = torch.cat([raw_video_tensor,new_video_tensor],dim=3)
    cat_video = [torchvision.transforms.ToPILImage()(cat_video_tensor[index]) for index in range(cat_video_tensor.shape[0])]
    export_to_video(cat_video, os.path.join("./", f"cat_video_{i:05}_image_interpolate.mp4"), fps=10)