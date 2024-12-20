
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
from get_condition import Conditioner

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

        if self.split == "train":
            anno_path = os.path.join(data_root, "vista_anno/nuScenes.json")
        elif self.split == "val":
            anno_path = os.path.join(data_root, "vista_anno/nuScenes_val.json")
        else:
            raise ValueError
        
        with open(anno_path, 'r') as f:
            self.anno_old = json.load(f)
        self.anno = []
        for anno in self.anno_old:
            if len(anno['traj'])!=10:
                continue
            if len(anno['speed'])!=5:
                continue
            if len(anno['angle'])!=5:
                continue
            self.anno.append(anno)
        
        self.action_mod = 0
        self.encode_condition = Conditioner()

        self.drop = 0.1
    
    def __len__(self):
        return len(self.anno)
    
    def load_sample(self, index):
        scene = self.anno[index]

        frames = torch.Tensor(np.stack([image2arr(os.path.join(self.data_root, path)) for path in scene["frames"]]))

        frames = (frames - 127.5) / 127.5
        frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)

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
        
        if random.random()<self.drop:
            prompt = ""
        
        cond_dict = {}
        if random.random()>self.drop:
            cond_dict["trajectory"] = torch.tensor(scene["traj"][2:])
        if scene["speed"] and random.random()>self.drop:
                cond_dict["speed"] = torch.tensor(scene["speed"][1:])
        if scene["angle"] and random.random()>self.drop:
            cond_dict["angle"] = torch.tensor(scene["angle"][1:]) / 780
        if random.random()>self.drop and scene["z"] > 0 and 0 < scene["goal"][0] < 1600 and 0 < scene["goal"][1] < 900:
                cond_dict["goal"] = torch.tensor([
                    scene["goal"][0] / 1600,
                    scene["goal"][1] / 900
                ])
        
        for k in cond_dict.keys():
            cond_dict[k] = cond_dict[k].unsqueeze(0)

        # print(cond_dict)

        return prompt, cond_dict, frames

    def encode(self, prompt, cond, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        cond = self.encode_condition(cond).to("cpu")
        prompt = torch.cat([prompt,cond],dim=1)
        if self.encode_video is not None:
            v1,v2 = self.encode_video(video)
            video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
        return prompt, video
        
    # def __getitem__(self, index):
    #     prompt, cond, video = self.load_sample(index)
    #     if self.split=="train":
    #         prompt, video = self.encode(prompt, cond, video)

    #     self.action_mod = (self.action_mod + 1) % 3
    #     return {
    #         "instance_prompt": prompt,
    #         "instance_video": video,
    #     }
    def __getitem__(self, index):
        return {
            "traj":torch.Tensor(self.anno[index]["traj"])[2:],
            "speed":torch.Tensor(self.anno[index]["speed"]),
            "angle":torch.Tensor(self.anno[index]["angle"])
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