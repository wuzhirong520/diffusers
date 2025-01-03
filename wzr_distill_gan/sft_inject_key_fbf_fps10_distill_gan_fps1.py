# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from datetime import timedelta
from einops import rearrange, repeat
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import transformers
import accelerate
from accelerate.state import AcceleratorState
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from transformers.utils import ContextManagers

import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
sys.path.append("../src")

import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    # CogVideoXTransformer3DModel,
)
from diffusers.models.transformers.cogvideox_transformer_3d_wxd import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_inject_fbf import CogVideoXImageToVideoPipeline


from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
import numpy as np
from diffusers.image_processor import VaeImageProcessor

from nuscenes_dataset_for_cogvidx import NuscenesDatasetAllframesFPS10OneByOneForDistillCogvidx
from guidance_model import UnifiedModel


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default="/data/wangxd/nuscenes",
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Validation
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help="One or more image path(s) that is used during validation to verify that the model is learning. Multiple validation paths should be separated by the '--validation_prompt_seperator' string. These should correspond to the order of the validation prompts.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-i2v-sft",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default="center",
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=13, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    parser.add_argument(
        "--noised_image_dropout",
        type=float,
        default=0.05,
        help="Image condition dropout probability.",
    )
    parser.add_argument(
        "--denoised_image",
        action="store_true",
        default=False,
        help="use the clean first image",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--nccl_timeout", type=int, default=600, help="NCCL backend timeout in seconds.")

    parser.add_argument("--learning_rate_guidance",type=float,default=5e-5,help="")
    parser.add_argument("--generator_update_ratio", type=int,default=1,help="")
    parser.add_argument("--distill_loss_weight", type=float,default=1.0,help="")
    parser.add_argument("--gen_cls_loss_weight", type=float,default=1.0,help="")
    parser.add_argument("--diffusion_loss_weight", type=float,default=1.0,help="")
    parser.add_argument("--gan_loss_weight", type=float,default=1.0,help="")

    return parser.parse_args()


def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            video_path = f"final_video_{i}.mp4"
            export_to_video(video, os.path.join(repo_folder, video_path, fps=fps))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": video_path}},
            )

    model_description = f"""
# CogVideoX LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA weights for {base_model}.

The weights were trained using the [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_image_to_video_lora.py).

Was LoRA for the text encoder enabled? No.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video

pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name=["cogvideox-i2v-lora"])

# The LoRA adapter weights are determined by what was used for training.
# In this case, we assume `--lora_alpha` is 32 and `--rank` is 64.
# It can be made lower or higher from what was used in training to decrease or amplify the effect
# of the LoRA upto a tolerance, beyond which one might notice no effect at all or overflows.
pipe.set_adapters(["cogvideox-i2v-lora"], [32 / 64])

image = load_image("/path/to/image")
video = pipe(image=image, "{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "image-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "cogvideox",
        "cogvideox-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_tiling()
    # pipe.vae.enable_slicing()

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[0]
        pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_images)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)

        videos.append(image_pil)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{phase_name}_video_{epoch}_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    del pipe
    free_memory()

    return videos


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, 
    )
    
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, 
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, 
        )
        load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
        
        fine_transformer = CogVideoXTransformer3DModel.from_pretrained(
            "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps10-1222-fbf-noaug/checkpoint-5000",
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=args.revision,
            variant=args.variant,
            in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        print("Loaded frozen fine transformer")


    unified_model = UnifiedModel(args)
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",)

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    fine_transformer.requires_grad_(False)

    unified_model.requires_grad_(True)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            args.mixed_precision = accelerator.mixed_precision 
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            args.mixed_precision = accelerator.mixed_precision 

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    fine_transformer.to(accelerator.device, dtype=weight_dtype)
    fine_transformer.eval()

    unified_model.to(accelerator.device, dtype=weight_dtype)
    unified_model.train()

    if args.gradient_checkpointing:
        unified_model.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if accelerator.distributed_type == DistributedType.DEEPSPEED:
                    model = unwrap_model(model)
                # if isinstance(model, type(unwrap_model(unified_model))):
                #     model.feedforward_model.save_pretrained(os.path.join(output_dir, "transformer"))
                #     unwrap_model(model.guidance_model).fake_unet.save_pretrained(os.path.join(output_dir, "fake_transformer"))
                #     unwrap_model(model.guidance_model).discriminator.save_pretrained(os.path.join(output_dir, "discriminator"))
                if isinstance(model, type(unwrap_model(unified_model.feedforward_model))):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(model, type(unwrap_model(unified_model.guidance_model))):    
                    model.fake_unet.save_pretrained(os.path.join(output_dir, "fake_transformer"))
                    model.discriminator.save_pretrained(os.path.join(output_dir, "discriminator"))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                if accelerator.distributed_type != DistributedType.DEEPSPEED:
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")
        
        load_model = CogVideoXTransformer3DModel.from_pretrained(input_dir, subfolder="transformer")
        transformer_.register_to_config(**load_model.config)
        
        transformer_.load_state_dict(load_model.state_dict())
        del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )


    # # Make sure the trainable params are in float32.
    # if args.mixed_precision == "fp16":
    #     # only upcast trainable parameters (LoRA) into fp32
    #     cast_training_params([feedforward_model], dtype=torch.float32)
    
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    # Optimization parameters
    trainable_parameters = list(unified_model.parameters())
    transformer_parameters_with_lr = {"params": trainable_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Optimization Guidance Model parameters
    # guidance_trainable_parameters = list(guidance_model.parameters())
    # guidance_transformer_parameters_with_lr = {"params": guidance_trainable_parameters, "lr": args.learning_rate_guidance}
    # guidance_params_to_optimize = [guidance_transformer_parameters_with_lr]
    # optimizer_guidance = get_optimizer(args, guidance_params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    def encode_video(video):
        video = video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0)
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        image = video[:, :, :1].clone()

        # with torch.no_grad():
        frame_num = video.shape[2] # F
        frames_latent = []
        for idx in range(frame_num):
            
            frames_latent.append(vae.encode(video[:, :, idx:idx+1]).latent_dist.sample().cpu())
        
        latent_dist = torch.cat(frames_latent, dim=2) # frame by frame
        # print(latent_dist.requires_grad)
        # latent_dist.requires_grad_(False)
        # latent_dist = vae.encode(video).latent_dist # single-pass
        del frames_latent
        
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=image.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=image.dtype)
        noisy_image = torch.randn_like(image) * image_noise_sigma[:, None, None, None, None] + image
        if args.denoised_image:
            noisy_image = image
        image_latent_dist = vae.encode(noisy_image).latent_dist

        latent_dist = latent_dist.detach()
        
        return latent_dist, image_latent_dist
    
    def encode_prompt(prompt):
        return compute_prompt_embeddings(
            tokenizer,
            text_encoder,
            [prompt],
            unwrap_model(unified_model).feedforward_model.config.max_text_seq_length,
            accelerator.device,
            weight_dtype,
            requires_grad=False,
        )
    
    # Dataset and DataLoader
    train_dataset = NuscenesDatasetAllframesFPS10OneByOneForDistillCogvidx(
        data_root=args.instance_data_root,
        height=args.height,
        width=args.width,
        max_num_frames=args.max_num_frames,
        encode_video=encode_video,
        encode_prompt=encode_prompt
    )

    def collate_fn(examples):
        videos = []
        images = []
        for example in examples:
            latent_dist, image_latent_dist = example["instance_video"]

            latent_dist = latent_dist.to(accelerator.device)
            image_latent_dist = image_latent_dist.to(accelerator.device)

            # video_latents = latent_dist.sample() * vae.config.scaling_factor
            # image_latents = image_latent_dist.sample() * vae.config.scaling_factor
            video_latents = latent_dist * vae.config.scaling_factor
            # image_latents = image_latent_dist * vae.config.scaling_factor

            video_latents = video_latents.permute(0, 2, 1, 3, 4)
            # image_latents = image_latents.permute(0, 2, 1, 3, 4)

            # padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
            # latent_padding = image_latents.new_zeros(padding_shape)
            # image_latents = torch.cat([image_latents, latent_padding], dim=1) # copy and repeat

            # if random.random() < args.noised_image_dropout:
            #     image_latents = torch.zeros_like(image_latents)

            videos.append(video_latents)
            # images.append(image_latents)

        videos = torch.cat(videos)
        # images = torch.cat(images)
        videos = videos.to(memory_format=torch.contiguous_format).float()
        # images = images.to(memory_format=torch.contiguous_format).float()

        prompts = [example["instance_prompt"].to(accelerator.device) for example in examples]
        prompts = torch.cat(prompts)

        return {
            "videos": (videos, images),
            "prompts": prompts,
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )

    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    # unified_model, optimizer, optimizer_guidance, train_dataloader, lr_scheduler,lr_scheduler_guidance = accelerator.prepare(
    #     unified_model, optimizer, optimizer_guidance, train_dataloader, lr_scheduler,lr_scheduler_guidance
    # )
    # feedforward_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     feedforward_model, optimizer, train_dataloader, lr_scheduler
    # )
    # guidance_model, optimizer_guidance, lr_scheduler_guidance = accelerator.prepare(
    #     guidance_model, optimizer_guidance, lr_scheduler_guidance
    # )
    unified_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unified_model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-i2v-ft"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])
    # num_guidance_trainable_parameters = sum(param.numel() for model in guidance_params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    # logger.info(f"  Num guidance trainable parameters = {num_guidance_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    # For DeepSpeed training
    model_config = unwrap_model(unified_model).feedforward_model.config

    for epoch in range(first_epoch, args.num_train_epochs):
        unified_model.train()

        for step, batch in enumerate(train_dataloader):
            # print(unwrap_model(transformer).patch_embed.proj.weight)
            # print(unwrap_model(transformer).patch_embed.proj.weight.shape)
            models_to_accumulate = [unified_model]

            with accelerator.accumulate(models_to_accumulate):
                video_latents, _ = batch["videos"]
                prompt_embeds = batch["prompts"]
                
                # import pdb; pdb.set_trace()

                video_latents = video_latents.to(dtype=weight_dtype)  # [B, F, C, H, W]
                # image_latents = image_latents.to(dtype=weight_dtype)  # [B, F, C, H, W], copy and repeat

                batch_size, num_frames, num_channels, height, width = video_latents.shape
                
                assert batch_size == 1, "Batch size must be 1"
                
                key_frame_indeces = torch.arange(0, num_frames, 12)
                
                continue_video_latents = []

                for i in range(len(key_frame_indeces) - 1):
                    start = key_frame_indeces[i].item()
                    end = key_frame_indeces[i + 1].item()
                    frames = video_latents[0, start:end+1] # only take first item
                    continue_video_latents.append(frames)
                
                # import pdb; pdb.set_trace()

                continue_video_latents = torch.stack(continue_video_latents) # [B*12, 13, C, H, W]
                continue_image_latents = continue_video_latents[:, :1] # [B*12, 1, C, H, W]
                
                coarse_video_latents = continue_video_latents[:, -1:].clone() # [B*L, 1, C, H, W]
                coarse_video_latents = torch.cat([continue_video_latents[:1, :1].clone(), coarse_video_latents]) # [13, 1, C, H, W]
                
                coarse_video_latents = rearrange(coarse_video_latents, "(b l) f c h w -> b (l f) c h w", b=1)
                coarse_image_latents = coarse_video_latents[:, :1]
                coarse_padding_shape = (coarse_video_latents.shape[0], coarse_video_latents.shape[1] - 1, *coarse_video_latents.shape[2:])
                coarse_padding = coarse_image_latents.new_zeros(coarse_padding_shape)
                coarse_image_latents = torch.cat([coarse_image_latents, coarse_padding], dim=1)
                
                padding_shape = (continue_video_latents.shape[0], continue_video_latents.shape[1] - 1, *continue_video_latents.shape[2:])
                latent_padding = continue_image_latents.new_zeros(padding_shape)
                continue_image_latents = torch.cat([continue_image_latents, latent_padding], dim=1) # copy and repeat [B*L, 13, C, H, W]

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.height,
                        width=args.width,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )
                image_latent = coarse_image_latents[:, :1] # take the first frame

                logs = {}

                COMPUTE_GENERATOR_GRADIENT = step % args.generator_update_ratio == 0
                
                if COMPUTE_GENERATOR_GRADIENT:
                    
                    clean_x_0 = unified_model(
                        clean_x_0 = None,
                        continue_video_latents = continue_video_latents,
                        coarse_video_latents = coarse_video_latents,
                        coarse_image_latents = coarse_image_latents,
                        image_latent = image_latent,
                        prompt_embeds = prompt_embeds,
                        image_rotary_emb = image_rotary_emb,
                        real_unet = fine_transformer,
                        scheduler = scheduler,
                        losses_to_compute = ['clean_x_0'],
                    )['clean_x_0']

                    unwrap_model(unified_model).frozen_guidance_model()
                    generator_losses = unified_model(
                        clean_x_0 = clean_x_0,
                        continue_video_latents = continue_video_latents,
                        coarse_video_latents = coarse_video_latents,
                        coarse_image_latents = coarse_image_latents,
                        image_latent = image_latent,
                        prompt_embeds = prompt_embeds,
                        image_rotary_emb = image_rotary_emb,
                        real_unet = fine_transformer,
                        scheduler = scheduler,
                        # losses_to_compute = ['distill_loss', 'gen_cls_loss'])
                        losses_to_compute = ['distill_loss'])
                    
                    distill_loss = generator_losses['distill_loss']
                    # gen_cls_loss = generator_losses['gen_cls_loss']
                    
                    generator_loss = 0.0
                    if args.distill_loss_weight>0:
                        generator_loss += distill_loss * args.distill_loss_weight
                    # if args.gen_cls_loss_weight > 0:
                    #     generator_loss += gen_cls_loss * args.gen_cls_loss_weight
                    
                    accelerator.backward(generator_loss)

                    if accelerator.sync_gradients:
                        params_to_clip = trainable_parameters
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    if accelerator.state.deepspeed_plugin is None:
                        optimizer.step()
                        optimizer.zero_grad()

                    lr_scheduler.step()

                    logs["distill_loss"] = distill_loss.detach().item()
                    # logs["gen_cls_loss"] = gen_cls_loss.detach().item()
                    logs["generator_loss"] = generator_loss.detach().item()

                    unwrap_model(unified_model).unfrozen_guidance_model()
                else: 
                    
                    with torch.no_grad():
                        clean_x_0 = unified_model(
                            clean_x_0 = None,
                            continue_video_latents = continue_video_latents,
                            coarse_video_latents = coarse_video_latents,
                            coarse_image_latents = coarse_image_latents,
                            image_latent = image_latent,
                            prompt_embeds = prompt_embeds,
                            image_rotary_emb = image_rotary_emb,
                            real_unet = fine_transformer,
                            scheduler = scheduler,
                            losses_to_compute = ['clean_x_0'],
                        )['clean_x_0']

                    guidance_losses = unified_model(
                        clean_x_0 = clean_x_0,
                        continue_video_latents = continue_video_latents,
                        coarse_video_latents = coarse_video_latents,
                        coarse_image_latents = coarse_image_latents,
                        image_latent = image_latent,
                        prompt_embeds = prompt_embeds,
                        image_rotary_emb = image_rotary_emb,
                        real_unet = fine_transformer,
                        scheduler = scheduler,
                        losses_to_compute = ['diffusion_loss', 'gan_loss'])
                    
                    diffusion_loss = guidance_losses['diffusion_loss']
                    gan_loss = guidance_losses['gan_loss']

                    guidance_loss = 0.0
                    if args.diffusion_loss_weight>0:
                        guidance_loss += diffusion_loss * args.diffusion_loss_weight
                    if args.gan_loss_weight > 0:
                        guidance_loss += gan_loss * args.gan_loss_weight

                    accelerator.backward(guidance_loss)
                    if accelerator.sync_gradients:
                        params_to_clip = trainable_parameters
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    if accelerator.state.deepspeed_plugin is None:
                        optimizer.step()
                        optimizer.zero_grad()

                    lr_scheduler.step()

                    logs["diffusion_loss"] = diffusion_loss.detach().item()
                    logs["gan_loss"] = gan_loss.detach().item()
                    logs["guidance_loss"] = guidance_loss.detach().item()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        
                        # accelerator.save_state(save_path)
                        unwrap_model(unified_model).save_model(save_path)

                        logger.info(f"Saved state to {save_path}")

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if args.validation_prompt is not None and (step + 1) % args.validation_epochs == 0:
                    # Create pipeline
                    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=unwrap_model(unified_model).feedforward_model,
                        scheduler=scheduler,
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                        vae = vae,
                        text_encoder=text_encoder,
                        inject=True
                    )

                    validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                    validation_images = args.validation_images.split(args.validation_prompt_separator)

                    # for validation_image, validation_prompt in zip(validation_images, validation_prompts):
                    for validation_image in validation_images:
                        for validation_prompt in validation_prompts:
                            pipeline_args = {
                                "image": load_image(validation_image),
                                "prompt": validation_prompt,
                                "guidance_scale": args.guidance_scale,
                                "use_dynamic_cfg": args.use_dynamic_cfg,
                                "height": args.height,
                                "width": args.width,
                                "num_frames": 13, #hard code
                            }

                            validation_outputs = log_validation(
                                pipe=pipe,
                                args=args,
                                accelerator=accelerator,
                                pipeline_args=pipeline_args,
                                epoch=epoch,
                            )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(feedforward_model)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        
        feedforward_model.save_pretrained(os.path.join(args.output_dir, 'transformer'))

        # Cleanup trained models to save memory
        del transformer
        free_memory()

        # check saved model
        transformer_ = CogVideoXTransformer3DModel.from_pretrained(
            args.output_dir,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=args.revision,
            variant=args.variant,
            in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        transformer_.to(accelerator.device, dtype=weight_dtype)

        # Final test inference
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            vae = vae,
            text_encoder=text_encoder,
            transformer = transformer_
        )
        
        pipe.enable_model_cpu_offload()
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config,)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()

        # Run inference
        validation_outputs = []
        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            validation_images = args.validation_images.split(args.validation_prompt_separator)

            for validation_image, validation_prompt in zip(validation_images, validation_prompts):
                pipeline_args = {
                    "image": load_image(validation_image),
                    "prompt": validation_prompt,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": 13,
                }

                video = log_validation(
                    pipe=pipe,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=0,
                    is_final_validation=True,
                )
                validation_outputs.extend(video)

        if args.push_to_hub:
            validation_prompt = args.validation_prompt or ""
            validation_prompt = validation_prompt.split(args.validation_prompt_separator)[0]
            save_model_card(
                repo_id,
                videos=validation_outputs,
                base_model=args.pretrained_model_name_or_path,
                validation_prompt=validation_prompt,
                repo_folder=args.output_dir,
                fps=args.fps,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
