import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import sys
sys.path.append("../src")
import numpy as np

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

from diffusers.models.transformers.cogvideox_transformer_3d_traj import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_inject_fbf_traj import CogVideoXImageToVideoPipeline


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
        "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-1219-fbf-noaug/checkpoint-5000",
        subfolder="transformer",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True
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
prompt = "Turn right, then go straight and speed up fast.. Cloudy. Daytime. An intersection with multiple lanes, traffic lights, and various vehicles. Traffic lights, vehicles (including a large truck), and the road markings. The scene then changes: Urban street with multiple lanes, buildings, and traffic lights. Traffic lights, vehicles (including a large truck), buildings. The scene then changes: Urban, with buildings on both sides of the road, a bicycle lane marked on the street, and a fire hydrant on the sidewalk. Buildings, road markings, fire hydrant."
trajs = torch.Tensor([[[[-2.2737e-13,  2.2737e-13],
          [ 3.9377e-02,  3.0478e+00],
          [ 1.3506e-01,  5.8453e+00],
          [ 5.2560e-01,  8.5601e+00],
          [ 1.2755e+00,  1.1166e+01]],

         [[ 1.1369e-13, -2.2737e-13],
          [ 2.7379e-01,  2.7296e+00],
          [ 9.1122e-01,  5.3660e+00],
          [ 1.9255e+00,  7.8668e+00],
          [ 3.3760e+00,  1.0244e+01]],

         [[-5.8620e-14,  0.0000e+00],
          [ 3.5779e-01,  2.6723e+00],
          [ 1.1688e+00,  5.3339e+00],
          [ 2.4927e+00,  7.8605e+00],
          [ 4.3479e+00,  1.0184e+01]],

         [[ 1.1369e-13,  0.0000e+00],
          [ 4.5479e-01,  2.8179e+00],
          [ 1.4783e+00,  5.6104e+00],
          [ 3.0164e+00,  8.2456e+00],
          [ 5.0429e+00,  1.0730e+01]],

         [[-2.2737e-13,  0.0000e+00],
          [ 4.5531e-01,  3.0171e+00],
          [ 1.4202e+00,  6.0742e+00],
          [ 2.7378e+00,  9.1553e+00],
          [ 3.9853e+00,  1.1694e+01]],

         [[ 2.2737e-13, -1.1369e-13],
          [ 2.3828e-01,  3.3427e+00],
          [ 5.8761e-01,  6.1501e+00],
          [ 1.1199e+00,  9.7936e+00],
          [ 1.7055e+00,  1.3576e+01]],

         [[ 0.0000e+00,  2.2737e-13],
          [ 7.5549e-02,  3.6807e+00],
          [ 1.8649e-01,  7.5060e+00],
          [ 3.6541e-01,  1.1554e+01],
          [ 5.7545e-01,  1.5744e+01]],

         [[ 0.0000e+00, -3.4106e-13],
          [ 5.9669e-02,  4.0512e+00],
          [ 1.4623e-01,  8.2464e+00],
          [ 2.4741e-01,  1.2606e+01],
          [ 3.3388e-01,  1.7088e+01]],

         [[ 2.2737e-13, -1.1369e-13],
          [ 2.3129e-02,  4.3609e+00],
          [ 2.9309e-02,  8.8433e+00],
          [ 1.0327e-02,  1.4479e+01],
          [-8.8081e-03,  1.9273e+01]],

         [[ 2.2737e-13,  1.1369e-13],
          [-9.6755e-04,  5.6360e+00],
          [-4.7851e-03,  1.0430e+01],
          [-4.9186e-02,  1.5332e+01],
          [-5.3884e-02,  2.0357e+01]],

         [[ 2.2737e-13, -2.2737e-13],
          [ 1.0260e-02,  4.8998e+00],
          [ 6.1606e-02,  9.9230e+00],
          [ 1.1204e-01,  1.3991e+01],
          [ 1.4284e-01,  1.8687e+01]],

         [[ 0.0000e+00,  0.0000e+00],
          [ 4.9743e-02,  4.0692e+00],
          [ 7.9775e-02,  8.7670e+00],
          [ 1.5748e-01,  1.4003e+01],
          [ 2.5400e-01,  1.9155e+01]],

         [[ 0.0000e+00,  0.0000e+00],
          [ 5.9307e-02,  5.2362e+00],
          [ 1.3773e-01,  1.0389e+01],
          [ 2.1294e-01,  1.5405e+01],
          [ 2.9330e-01,  2.1048e+01]]]])

from nuscenes_dataset_for_cogvidx import NuscenesDatasetFPS1OneByOneTrajectoryForCogvidx
val_dataset = NuscenesDatasetFPS1OneByOneTrajectoryForCogvidx(split="val")

item = val_dataset[13]
prompt, frames , trajs = item["instance_prompt"], item["instance_video"], item["trajectory"]

# print(prompt)
frames_ = [load_image(os.path.join(val_dataset.data_root, f)) for f in frames]
export_to_video(frames_, "./infer_results/x_gt.mp4")
# exit(0)

print(prompt)
print(frames[0])
print(trajs)

pipeline_args = {
    "image": load_image(os.path.join(val_dataset.data_root, frames[0])),
    "prompt": prompt,
    "trajectory": trajs, 
    "guidance_scale": 6,
    "use_dynamic_cfg": True,
    "height": 480,
    "width": 720,
    "num_frames": 13,
    "num_inference_steps" : 50
}
frames = pipe(**pipeline_args).frames[0]
print(len(frames))
# for i in range(len(frames)):
#     frames[i].save(f"./infer_results/x/{i:03d}.jpg")
export_to_video(frames, "./infer_results/x.mp4")

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