import os    
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
# for i in range(1,11):
#     for action in actions:
#         path = f"./val1022/{action}-{i}.mp4"
#         new_path = f"./val1022_vista/{i}-{action}.mp4"
#         os.system(f"mv \"{path}\" \"{new_path}\"")
import PIL.Image
import decord
decord.bridge.set_bridge("torch")

def get_frames(filename):
    video_reader = decord.VideoReader(filename, width=720, height=480)
    return video_reader.get_batch(list(range(0,33)))

import torch

# scenes = ['0003',"0012","0013","0014","0015","0016","0017","0018"]

for idx in range(1, 15):
    videos=[]
    for action in actions:
        # video = get_frames(f"/root/PKU/diffusers/wzr_example/Infer/val1022_vista/{idx}-{action}.mp4")
        # video = get_frames(f"/root/PKU/diffusers/wzr_example/Infer/val1022/scene-{scenes[idx]}_{action}.mp4".replace(' ','_'))
        video = get_frames(f"/root/PKU/diffusers/wzr_example/Infer/val_campus/{action}-{idx}.mp4")
        videos.append(video)

    padding = torch.zeros_like(videos[0])
    video1 = torch.cat(videos[:4],dim=2)
    video2 = torch.cat(videos[4:8],dim=2)
    video3 = torch.cat(videos[8:12],dim=2)
    video4 = torch.cat(videos[12:13]+[padding]+[padding]+[padding],dim=2)
    video = torch.cat([video1,video2,video3,video4],dim=1)
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

    # export_to_video(frames, f"/root/PKU/diffusers/wzr_example/Infer/val1022_vista/{idx}.mp4",fps=8)
    # export_to_video(frames, f"/root/PKU/diffusers/wzr_example/Infer/scene-{scenes[idx]}.mp4",fps=8)
    export_to_video(frames, f"/root/PKU/diffusers/wzr_example/Infer/val_campus_all/{idx}.mp4",fps=8)