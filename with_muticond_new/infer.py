import os
import sys
sys.path.append("../src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

from diffusers.models.transformers.cogvideox_transformer_3d_wxd import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_inject_fbf import CogVideoXImageToVideoPipeline


from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file

pretrained_model_name_or_path = "/data/wuzhirong/hf-models/CogVideoX-2b"

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=None, 
        torch_dtype=torch.float16,
    )

text_encoder = T5EncoderModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
    torch_dtype=torch.float16,
)

transformer = CogVideoXTransformer3DModel.from_pretrained(
        "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-1219-fbf-noaug/checkpoint-3000",
        subfolder="transformer",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        # in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )

scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",)

components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

pipe = CogVideoXImageToVideoPipeline(**components, inject=True)
# pipe.enable_model_cpu_offload()
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()
pipe.to("cuda")

print('Pipeline loaded!')

image_path = "/data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281646662460.jpg"

pipeline_args = {
    "image": load_image(image_path),
    "prompt": "turn left",
    "guidance_scale": 6,
    "use_dynamic_cfg": True,
    "height": 480,
    "width": 720,
    "num_frames": 13,
    "num_inference_steps" : 50
}
frames = pipe(**pipeline_args).frames[0]
print(len(frames))
for i in range(len(frames)):
    frames[i].save(f"./infer_results/206/{i:03d}.jpg")
export_to_video(frames, "./infer_results/206.mp4")

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