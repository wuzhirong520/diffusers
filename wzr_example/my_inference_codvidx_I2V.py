import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
# os.environ['HF_HOME']="/root/autodl-fs/cache/huggingface/"
os.environ['HUGGINGFACE_HUB_CACHE']="/root/autodl-fs/huggingface/hub"
import sys
sys.path.append("/root/PKU/diffusers/src")
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import torch

from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from transformers import T5EncoderModel

# text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-2b", subfolder="text_encoder", torch_dtype=torch.float16)

# transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-2b", subfolder="transformer", torch_dtype=torch.float16, in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

transformer = CogVideoXTransformer3DModel.from_pretrained("/root/autodl-fs/CogVidx-2b-I2V-base-transfomer", torch_dtype=torch.float16)


# vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float16)

pipe = CogVideoXImageToVideoPipeline.from_pretrained("/root/autodl-fs/cogvidx-2b-I2V-Nuscenes_base0", 
                                                    #  text_encoder=text_encoder,
                                                    transformer=transformer,
                                                    #  vae=vae,
                                                    torch_dtype=torch.float16
                                                    ).to("cuda")

# transformer.save_pretrained("/root/autodl-fs/CogVidx-2b-I2V-base-transfomer")
# exit(0)

# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
# prompt = ""
# image = load_image("/root/PKU/diffusers/wzr_example/astronaut.jpg")

prompt = "The ego car moves forward at a steady pace, occasionally shifting slightly to the left and right as it navigates the curve of the road."
image = load_image("/root/PKU/diffusers/wzr_example/Nuscenes/val/scene-0012/0.jpg")

with torch.no_grad():
    video = pipe(image, prompt, num_frames=33)
    export_to_video(video.frames[0], "output_cogvideo_i2v_test.mp4", fps=8)