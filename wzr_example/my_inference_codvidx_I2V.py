import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import sys
sys.path.append("/home/wuzhirong/PKU_new/diffusers/src")
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import torch

pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
prompt = ""
image = load_image("/mnt/sde/wuzhirong/PKU_new/diffusers/wzr_example/astronaut.jpg")

with torch.no_grad():
    video = pipe(image, prompt, use_dynamic_cfg=True)
    export_to_video(video.frames[0], "output_cogvideo_i2v_test_.mp4", fps=8)