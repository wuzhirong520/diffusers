import os
import sys
sys.path.append("../src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
)

from diffusers.utils import load_image, export_to_video

from safetensors.torch import load_file

samples = list(os.listdir("/data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT"))
samples = [(s,"samples") for s in samples]
print(len(samples),samples[0])
sweeps = list(os.listdir("/data/wuzhirong/datasets/Nuscenes/sweeps/CAM_FRONT"))
sweeps = [(s,"sweeps") for s in sweeps]
print(len(sweeps),sweeps[0])
total = sorted(samples + sweeps, key = lambda x: x[0])
total = [s[1] + "/CAM_FRONT/" + s[0] for s in total]
print(len(total),total[:10])
exit(0)


pretrained_model_name_or_path = "/data/wuzhirong/hf-models/CogVideoX-2b"

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )

img_paths = [
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281638112460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281638612460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281639162460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281639662460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281640162460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281640662460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281641162460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281641662460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281642162460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281642662460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643162629.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643662460.jpg",
    "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281644162460.jpg",
]

# img_paths = [
#     "samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643162629.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643262460.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643362460.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643412462.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643512460.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643612460.jpg",
#     "samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643662460.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643762460.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643862460.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643912460.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281644012460.jpg",
#     "sweeps/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281644112460.jpg",
#     "samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281644162460.jpg"
# ]
vae.to("cuda")
vae.enable_slicing()
vae.enable_tiling()
# imgs = [load_image("/data/wuzhirong/datasets/Nuscenes/" + path) for path in img_paths]
imgs = [load_image("/data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/" + path) for path in img_paths]
from diffusers.video_processor import VideoProcessor
video_processor = VideoProcessor(vae_scale_factor=8)
imgs = [video_processor.preprocess(image, height=480, width=720).to(device="cuda",dtype=torch.float16) for image in imgs]
imgs = torch.stack(imgs).permute(1,2,0,3,4)
with torch.no_grad():
    latent = vae.encode(imgs).latent_dist.sample()
print(latent.shape)
# torch.save(latent,"latent.pt")
# latent = torch.load("latent.pt").to(device="cuda",dtype=torch.float16)
# print(latent.shape)
with torch.no_grad():
    video = vae.decode(latent).sample
video = video_processor.postprocess_video(video=video, output_type="pil")[0]
print(len(video))
for i in range(len(video)):
    video[i].save(f"./imgs3/{i:03d}.jpg")