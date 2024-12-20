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
import transformers

import math

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

class NuscenesDatasetFPS1OneByOneForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 13, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples_groups = self.group_sample_by_scene(split)
            
            self.scenes = list(self.samples_groups.keys())

            if len(self.scenes) > max_samples:
                self.scenes = self.scenes[:max_samples]
            
            self.frames_group = {} # (scene, image_paths)
            
            for my_scene in self.scenes:
                f = self.get_paths_from_scene(my_scene)
                if len(f) >= max_num_frames :
                    self.frames_group[my_scene] = self.get_paths_from_scene(my_scene)
            self.scenes = [k for k,v in self.frames_group.items()]

            print('Total samples: %d' % len(self.scenes))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            command_path = f'{data_root}/nusc_action_26keyframe_all_{split}.json'
            with open(command_path, 'r') as f:
                self.command_dict = json.load(f)
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.RandomResizedCrop((height, width), scale=(0.5, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            
            
            if preload_all_data:
                self.preload()

    
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
    
    def load_sample(self, index):
        # my_scene = self.samples[index]
        # my_scene_name = my_scene['scene']
        # my_sample_list = my_scene["video"]
        # inner_frame_len = len(my_sample_list)
        
        my_scene_name = self.scenes[index]
        all_frames = self.frames_group[my_scene_name]
        
        # suppose 13*2
        seek_start = random.randint(0, len(all_frames) - self.max_num_frames*2)
        seek_path = all_frames[seek_start: seek_start + self.max_num_frames*2]
        
        seek_path = seek_path[::2] # 2 fps -> 1 fps

        scene_annotation = self.annotations[my_scene_name]
        
        seek_key_frame_idx = seek_start

        seek_id = seek_key_frame_idx//4*4
                
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
        
        # NOTE add more detailed captions
        addtional_timestep = [10, 20]
        for add_timestep in addtional_timestep:
            a_timestep = seek_key_frame_idx//4*4 + add_timestep
            a_timestep = f"{a_timestep}"
            a_timestep = a_timestep if a_timestep in scene_annotation else str((int(a_timestep)//4-1)*4)
            annotation = scene_annotation[a_timestep]
            selected_attr = ["Road environment", "Critical objects"]
            caption = caption + "The scene then changes: "
            for attr in selected_attr:
                anno = annotation.get(attr, "")
                if anno == "":
                    continue
                if anno[-1] == ".":
                    anno = anno[:-1] 
                caption = caption + anno + ". "
        
        command_idx = str(seek_key_frame_idx) # maybe 0-31, each denote 8 keyframes
        
        # import pdb; pdb.set_trace()
        
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
            "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
            "scene-0171", "scene-0172", "scene-0173", "scene-0174",
            "scene-0175", "scene-0176", "scene-0419"
            ]
        
        if my_scene_name in command_miss_scenes:
            driving_prompt = ""
        else:  
            try:
                driving_prompt = self.command_dict[my_scene_name][command_idx]
            except:
                # import pdb; pdb.set_trace()
                driving_prompt = ""
                pass
        driving_prompt = driving_prompt + ". " + caption
        
        # import pdb; pdb.set_trace()
        # print(driving_prompt)

        # frames = torch.Tensor(np.stack([image2arr(path) for path in frame_paths]))
        frames = np.stack([image2arr(path) for path in seek_path])

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        
        # no aug
        # frames = (frames - 127.5) / 127.5
        # frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        # frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)
        # NOTE aug, pil -> tensor
        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1,v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        try:
            if self.preload_all_data:
                return {
                    "instance_prompt": self.instance_prompts[index],
                    "instance_video": self.instance_videos[index],
                }
            else:
                prompt, video = self.load_sample(index)
                prompt, video = self.encode(prompt, video)
                return {
                    "instance_prompt": prompt,
                    "instance_video": video,
                }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

if __name__ == "__main__":
    dataset = NuscenesDatasetFPS1OneByOneForCogvidx(split="val")

    frames = dataset.frames_group
    scenes = dataset.scenes

    print(len(frames),len(scenes))

    data = {}

    for scene in scenes:
        fs = [f[len("/data/wangxd/nuscenes/"):] for f in frames[scene]]
        data[scene] = fs
    
    with open("nuscenes_val.json","w") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)






    # with open("nuscenes_train.json","r") as f:
    #     nuscenes = json.load(f)
    # with open("/data/wuzhirong/datasets/Nuscenes/vista_anno/nuScenes.json","r") as f:
    #     vista = json.load(f)
    # from tqdm import tqdm
    # for i in tqdm(range(len(nuscenes))):
    #     num_frames = len(nuscenes[i]["frames"])
    #     traj_relative = [[0,0]]
    #     speed, angle = [],[]
    #     for vista_index in range(len(vista)):
    #         if vista[vista_index]["frames"][0] == nuscenes[i]["frames"][0]:
    #             speed.append(vista[vista_index]['speed'][0])
    #             angle.append(vista[vista_index]['angle'][0])
    #             break
    #     if len(speed)==0:
    #         speed.append(None)
    #         angle.append(None)
    #     for frame in nuscenes[i]["frames"][1:]:
    #         found_traj = False
    #         for vista_index in range(len(vista)):
    #             vista_frames = vista[vista_index]["frames"]
    #             vista_traj = vista[vista_index]["traj"]
    #             vista_frame_id = -1
    #             fould_frame = False
    #             for f in vista_frames:
    #                 if str(f).startswith("samples"):
    #                     vista_frame_id+=1
    #                     if f==frame:
    #                         fould_frame=True
    #                         break
    #             if fould_frame and vista_frame_id>0 and vista_frame_id<5:
    #                 x = vista_traj[vista_frame_id*2] - vista_traj[vista_frame_id*2-2]
    #                 y = vista_traj[vista_frame_id*2+1] - vista_traj[vista_frame_id*2-1]
    #                 traj_relative.append([x,y])
    #                 speed.append(vista[vista_index]["speed"][vista_frame_id])
    #                 angle.append(vista[vista_index]["angle"][vista_frame_id])
    #                 found_traj = True
    #                 break
    #         if found_traj==False:
    #             traj_relative.append([None,None])
    #             speed.append(None)
    #             angle.append(None)
    #     assert num_frames==len(traj_relative)
    #     assert num_frames==len(speed)
    #     assert num_frames==len(angle)
    #     nuscenes[i]['num_frames'] = num_frames
    #     nuscenes[i]['traj_relative']=traj_relative
    #     nuscenes[i]['speed']=speed
    #     nuscenes[i]['angle']=angle
    #     # print(traj_relative)
    #     # exit(0)

    # with open("nuscenes_traj_train.json","w") as f:
    #     json.dump(nuscenes, f, ensure_ascii=False, indent=4)
