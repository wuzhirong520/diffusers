import os
import sys
sys.path.append("../src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

from diffusers.models.transformers.cogvideox_transformer_3d_interpolate import CogVideoXTransformer3DModel
# from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_interpolate import CogVideoXImageToVideoPipeline


from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file

pretrained_model_name_or_path = "/data/wuzhirong/hf-models/CogVideoX-2b"

# tokenizer = AutoTokenizer.from_pretrained(
#         pretrained_model_name_or_path, subfolder="tokenizer", revision=None, 
#         torch_dtype=torch.float16,
#     )

# text_encoder = T5EncoderModel.from_pretrained(
#     pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
#     torch_dtype=torch.float16,
# )

# transformer = CogVideoXTransformer3DModel.from_pretrained(
#         "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-fps10-f13-1202-inherit1022/checkpoint-1000",
#         subfolder="transformer",
#         torch_dtype=torch.float16,
#         revision=None,
#         variant=None,
#         in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
#     )

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )

# scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",)

# components = {
#             "transformer": transformer,
#             "vae": vae,
#             "scheduler": scheduler,
#             "text_encoder": text_encoder,
#             "tokenizer": tokenizer,
#         }

# pipe = CogVideoXImageToVideoPipeline(**components, inject=True)
# pipe.enable_model_cpu_offload()
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()
# pipe.to("cuda")

print('Pipeline loaded!')

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
# vae.enable_slicing()
# vae.enable_tiling()
# imgs = [load_image("/data/wuzhirong/datasets/Nuscenes/" + path) for path in img_paths]
imgs = [load_image("/data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/" + path) for path in img_paths]
from diffusers.video_processor import VideoProcessor
video_processor = VideoProcessor(vae_scale_factor=8)
imgs = [video_processor.preprocess(image, height=480, width=720).to(device="cuda",dtype=torch.float16) for image in imgs]
imgs = torch.stack(imgs).permute(1,2,0,3,4) # [B, C, F, H, W]
latent = []
with torch.no_grad():
    for i in range(0, imgs.shape[2]):
        x  = vae.encode(imgs[:,:,i:i+1,:,:]).latent_dist.sample()
        print(x.shape)
        latent.append(x)
    latent = torch.concatenate(latent,dim=2)
    # latent = vae.encode(imgs).latent_dist.sample()
print(latent.shape)
# torch.save(latent,"latent.pt")
# latent = torch.load("latent.pt").to(device="cuda",dtype=torch.float16)
# print(latent.shape)
video = []
with torch.no_grad():
    for i in range(0, imgs.shape[2]):
        x = vae.decode(latent[:,:,i:i+1,:,:]).sample
        print(x.shape)
        video.append(x)
    video = torch.concatenate(video,dim=2)
    # video = vae.decode(latent).sample
    print(video.shape)
video = video_processor.postprocess_video(video=video, output_type="pil")[0]
print(len(video))
for i in range(len(video)):
    video[i].save(f"./imgs5/{i:03d}.jpg")

# image_path = "/home/user/wuzhirong/Projects/diffusers/wxd/scene/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643162629.jpg"
# last_image_path = "/home/user/wuzhirong/Projects/diffusers/wxd/scene/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281644162460.jpg"
# pipeline_args = {
#     "image": load_image(image_path),
#     "last_image": load_image(last_image_path),
#     "prompt": "go straight",
#     "guidance_scale": 6,
#     "use_dynamic_cfg": True,
#     "height": 480,
#     "width": 720,
#     "num_frames": 13
# }
# frames = pipe(**pipeline_args).frames[0]
# print(len(frames))
# for i in range(len(frames)):
#     frames[i].save(f"./imgs3/{i:03d}.jpg")

# while True:
#     image_dir = input("image dir: ")
#     image_paths = list(os.listdir(image_dir))
#     image_paths = [a for a in image_paths if ".png" or ".jpg" in a[-4:]]
#     image_paths.sort()
#     validation_prompt = input("prompt: ")
#     guidance_scale = input("cfg: ") # 6
    
#     total_frames = []
#     for idx in range(4):
#         image_path = image_paths[idx]
#         last_image_path = image_paths[idx+1]
#         pipeline_args = {
#             "image": load_image(os.path.join(image_dir, image_path)),
#             "last_image": load_image(os.path.join(image_dir, last_image_path)),
#             "prompt": validation_prompt,
#             "guidance_scale": int(guidance_scale),
#             "use_dynamic_cfg": True,
#             "height": 480,
#             "width": 720,
#             "num_frames": 13
#         }
#         frames = pipe(**pipeline_args).frames[0]
#         # import pdb; pdb.set_trace()
#         print(len(frames))
#         # total_frames.extend(frames if idx==(len(image_paths)-2) else frames[:-1])
#         total_frames.extend(frames[3:] if idx==(len(image_paths)-2) else frames[4:])
#     name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
#     print(len(total_frames))
#     export_to_video(total_frames, os.path.join(image_dir, f"{name_prefix}_cfg_{guidance_scale}_interpolate_test_1k.mp4"), fps=10)