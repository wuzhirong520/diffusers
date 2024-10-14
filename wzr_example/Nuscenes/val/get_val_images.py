
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

class NuscenesDatasetForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/home/wuzhirong/PKU_new/Nuscenes-v1.0-trainval-CAM_FRONT",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 9, # must be (4k+1)
        split: str = "train",
        encode_prompt = None,
        encode_video = None
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video

        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

        self.splits = create_splits_scenes()

        # training samples
        self.samples_groups = self.group_sample_by_scene(split)
        
        self.scenes = list(self.samples_groups.keys())
        
        self.frames_group = {} # (scene, image_paths)
        
        for my_scene in self.scenes:
            self.frames_group[my_scene] = self.get_paths_from_scene(my_scene)

        print('Total samples: %d' % len(self.scenes))

        # search annotations
        json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.scenes)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def __getitem__(self, index):
        try:
            # index: scene idx
            my_scene = self.scenes[index]
            all_frames = self.frames_group[my_scene]
            
            seek_start = random.randint(0, len(all_frames) - self.max_num_frames)
            seek_path = all_frames[seek_start: seek_start + self.max_num_frames]            

            scene_annotation = self.annotations[my_scene]

            seek_id = seek_start//4*4
            search_id = f"{seek_id}"
            timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

            annotation = scene_annotation[timestep]
            
            caption = ""
            selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
            for attr in selected_attr:
                anno = annotation.get(attr, "")
                if anno == "":
                    continue
                if anno[-1] == ".":
                    anno = anno[:-1] 
                caption = caption + anno + ". "
            
            driving_prompt = annotation.get("Driving action", "").strip()
            # print(driving_prompt)

            frames = torch.Tensor(np.stack([image2arr(path) for path in seek_path]))

            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0

            # Training transforms
            frames = (frames - 127.5) / 127.5
            frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
            frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)

            prompt = self.encode_prompt(driving_prompt)
            video = self.encode_video(frames)

            return {
                "instance_prompt": prompt,
                "instance_video": video,
            }
            
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

if __name__ == "__main__":
    train_dataset = NuscenesDatasetForCogvidx(data_root="/root/autodl-fs/Nuscenes-v1.0-trainval-CAM_FRONT",split="train")
    scene = train_dataset.scenes[0]
    print(scene)
    os.makedirs(scene,exist_ok=True)
    frames = train_dataset.get_paths_from_scene(scene)
    for i,path in enumerate(frames):
        os.system(f"cp {path} ./{scene}/{i}.jpg")