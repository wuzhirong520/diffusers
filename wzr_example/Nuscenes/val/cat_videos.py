import PIL.Image
import decord
decord.bridge.set_bridge("torch")

def get_frames(filename):
    video_reader = decord.VideoReader(filename, width=720, height=480)
    return video_reader.get_batch(list(range(0,33)))

import torch

file1 = "/root/PKU/diffusers/wzr_example/Nuscenes/val/scene-0012/video.mp4"
file2 = "/root/autodl-tmp/cogvideox-lora-single-node_test_full_withembedtrain_fiximg/validation_video_41_0_The_ego_car_moves_forward.mp4"
file3 = "/root/autodl-tmp/cogvideox-lora-single-node_test_full_withembedtrain/validation_video_2_0_The_ego_car_moves_forward.mp4"
file4 = "/root/autodl-tmp/cogvideox-lora-single-node_test_full_withembedtrain/validation_video_3_0_The_ego_car_moves_forward.mp4"


video1 = get_frames(file1)
video2 = get_frames(file2)
video3 = get_frames(file3)
video4 = get_frames(file4)

video = torch.cat([video1,video2],dim=2)
# video = torch.cat([torch.cat([video1,video2],dim=2),torch.cat([video3,video4],dim=2)],dim=1)

print(video.shape)

import PIL
frames = []

for i in range(video.shape[0]):
    frame = video[i].numpy()
    frame = PIL.Image.fromarray(frame)
    frames.append(frame)

import sys
sys.path.append("/root/PKU/diffusers/src")
from diffusers.utils import export_to_video

export_to_video(frames, f"video_cmp.mp4",fps=2)