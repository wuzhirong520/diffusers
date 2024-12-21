
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

class NuscenesDatasetFPS1OneByOneTrajectoryForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wuzhirong/datasets/Nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 13, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video

        with open(os.path.join(data_root, f"nuscenes_{split}.json"),"r") as f:
            self.frames_group = json.load(f)
        
        self.scenes = list(self.frames_group.keys())

        if len(self.scenes) > max_samples:
            self.scenes = self.scenes[:max_samples]

        print('Total samples: %d' % len(self.scenes))

        # search annotations
        json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
        command_path = f'{data_root}/nusc_action_26keyframe_all_{split}.json'
        with open(command_path, 'r') as f:
            self.command_dict = json.load(f)

        vista_anno_path = os.path.join(data_root, f"vista_anno/nuScenes_{split}.json")
        with open(vista_anno_path,"r") as f:
            vista_anno = json.load(f)
            vista_anno_new = {}
            for v in vista_anno:
                vista_anno_new[v["frames"][0]]=v
            self.vista_anno = vista_anno_new
        
        # image transform
        self.transform = TT.Compose([
                TT.ToPILImage('RGB'),
                TT.Resize((height, width)),
                # TT.RandomResizedCrop((height, width), scale=(0.5, 1.), ratio=(1., 1.75)),
                # transforms.RandomHorizontalFlip(p=0.1),
                # TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                TT.ToTensor(),
                TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
            ])

    
    def __len__(self):
        return len(self.scenes)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def load_sample(self, index):
        # my_scene = self.samples[index]
        # my_scene_name = my_scene['scene']
        # my_sample_list = my_scene["video"]
        # inner_frame_len = len(my_sample_list)
        
        my_scene_name = self.scenes[index]
        all_frames = self.frames_group[my_scene_name]
        
        # suppose 13*2
        seek_start = random.randint(0, len(all_frames) - self.max_num_frames*2)
        if self.encode_prompt is None and self.encode_video is None:
            seek_start = 0
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
        # frames = np.stack([image2arr(path) for path in seek_path])
        frames = np.stack([image2arr(os.path.join(self.data_root, path)) for path in seek_path])

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

        trajs = []
        for path in seek_path:
            traj = torch.Tensor(self.vista_anno[path]['traj'])
            traj = torch.stack([traj[0::2],traj[1::2]],dim=1) # [5,2]
            trajs.append(traj)
        trajs = torch.stack(trajs).unsqueeze(0) # [1, 13, 5, 2]

        if self.encode_video is None:
            frames = seek_path

        return driving_prompt, frames, trajs

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1,v2.sample().to("cpu"))
        return prompt, video
        
    def __getitem__(self, index):
        try:
            prompt, video, trajs = self.load_sample(index)
            if self.encode_video is not None and self.encode_prompt is not None:
                prompt, video = self.encode(prompt, video)
            return {
                "instance_prompt": prompt,
                "instance_video": video,
                "trajectory" : trajs
            }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))



if __name__ == "__main__":
    train_dataset = NuscenesDatasetFPS1OneByOneTrajectoryForCogvidx(split="val")
    for i in tqdm(range(len(train_dataset))):
        t = train_dataset[i]
        print(t["trajectory"])
        # print(t["instance_video"])
        print(t["instance_prompt"])
        exit(0)