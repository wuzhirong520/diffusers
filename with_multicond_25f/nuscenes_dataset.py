
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
from traj_utils_torch import get_trajectory_image_pil, interpolate_trajectory

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

class NuscenesDatasetForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/root/autodl-fs/Nuscenes-v1.0-trainval-CAM_FRONT",
        height: int = 480,
        width: int = 720,
        split: str = "train",
        encode_video = None,
        encode_prompt = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.split = split
        self.encode_video=encode_video
        self.encode_prompt=encode_prompt

        anno_path = os.path.join(data_root, f"vista_anno/nuScenes_{self.split}.json")
        
        with open(anno_path, 'r') as f:
            self.anno = json.load(f)
    
    def __len__(self):
        return len(self.anno)
    
    def load_sample(self, index):
        scene = self.anno[index]

        frames = torch.Tensor(np.stack([image2arr(os.path.join(self.data_root, path)) for path in scene["frames"]]))
        frames = (frames - 127.5) / 127.5
        frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)

        if scene["cmd"]==0:
            prompt = "turn right"
        elif scene["cmd"]==1:
            prompt = "turn left"
        elif scene["cmd"]==2:
            prompt = "wait"
        elif scene["cmd"]==3:
            prompt = "go straight"
        else:
            raise ValueError
        
        trajectory = np.array(scene["traj"][2:])
        trajectory = trajectory.reshape(-1,2)
        trajectory = interpolate_trajectory(trajectory, 25)
        trajectory = np.concatenate([trajectory[:,0:1], np.zeros_like(trajectory[:,0:1]) , trajectory[:,1:]], axis=1)

        first_frame = Image.open(os.path.join(self.data_root, scene["frames"][0])).resize([self.width, self.height])
        traj_frames = get_trajectory_image_pil(first_frame, torch.Tensor(trajectory))
        traj_frames = torch.Tensor(np.stack([pil2arr(t) for t in traj_frames]))
        traj_frames = (traj_frames - 127.5) / 127.5
        traj_frames = traj_frames.permute(0, 3, 1, 2) # [F, C, H, W]
        

        return prompt, frames, trajectory, traj_frames

    def encode(self, prompt, video, traj_video):
        if self.encode_prompt is not None:
            prompt = self.encode_prompt(prompt).to("cpu")
        if self.encode_video is not None:
            v1,v2,v3 = self.encode_video(video, traj_video)
            video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
            traj_video = v3.sample().to("cpu")
        return prompt, video, traj_video
        
    def __getitem__(self, index):
        prompt, video, trajectory, traj_video = self.load_sample(index)
        prompt, video, traj_video = self.encode(prompt, video, traj_video)
        return {
            "instance_prompt": prompt,
            "instance_video": video,
            "instance_traj": trajectory,
            "instance_traj_video": traj_video
        }

if __name__ == "__main__":
    train_dataset = NuscenesDatasetForCogvidx(split="val")
    import sys
    sys.path.append("/root/PKU/diffusers/src")
    from diffusers.utils import export_to_video

    for i in tqdm(range(len(train_dataset))):
        sample = train_dataset[i]
        # print(sample)
        print(sample["instance_prompt"])
        # np.savetxt("prompt.txt", sample["instance_prompt"].numpy()[0,0])
        print(sample["instance_prompt"].shape)
        # break
        print(train_dataset.anno[i],file=open(f"val_gt/{i}.txt","w"))
        export_to_video([image2pil("/root/autodl-fs/Nuscenes-v1.0-trainval-CAM_FRONT/"+path) for path in train_dataset.anno[i]["frames"]],f"val_gt/{i}.mp4")

        if i>20:
            break