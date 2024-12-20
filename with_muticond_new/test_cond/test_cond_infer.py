import torch
import torch.nn as nn
from nuscenes_dataset import NuscenesDatasetForCogvidx
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

device = "cuda"

from einops import rearrange, repeat
import math
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.

    :return: an [N x dim] Tensor of positional embeddings.
    """

    if repeat_only:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device).unsqueeze(0).repeat(timesteps.shape[0],1)
        args = timesteps[:,:, None].float() * freqs[:,None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    embedding = rearrange(embedding, 'b n d -> b (n d)')
    return embedding

class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.L1 = nn.Linear(320,512,device=device)
        self.L2 = nn.Linear(512,128,device=device)
        self.L3 = nn.Linear(128,8,device=device)
        self.act = nn.Sigmoid()
    def forward(self, speed, angle):
        x = torch.cat([speed/3.6,angle/780.],dim=1)
        x = timestep_embedding(x, 32)
        x = self.act(self.L1(x))
        x = self.act(self.L2(x))
        x = self.L3(x)
        return x
import os
# model = torch.load("/data/wuzhirong/ckpts/test_traj_new/ckpt_09999.pt").to(device)
# model = torch.load("/data/wuzhirong/ckpts/test_traj_8_new/ckpt_07699.pt",weights_only=False).to(device)
ckpt_dir = "/data/wuzhirong/ckpts/test_traj_8_new_1"
ckpt_name = sorted(os.listdir(ckpt_dir))[-1]
print(ckpt_name)
model = torch.load(os.path.join(ckpt_dir, ckpt_name),weights_only=False).to(device)
model.eval()
print(model)

val_dataset = NuscenesDatasetForCogvidx("/data/wuzhirong/datasets/Nuscenes",split="val")
print(len(val_dataset))

selected = [206,197,683,1041,0, 3521,142]

def loss_func(gt,pred):
    return nn.L1Loss()(gt,pred)

import matplotlib.pyplot as plt
def visualize(gt, pred, index):
    plt.figure(figsize=(4, 3))
    plt.plot(gt[0::2], gt[1::2], marker='o', linestyle='-', color='r', label='traj')
    plt.plot(pred[0::2], pred[1::2], marker='o', linestyle='-', color='b', label='speedangle')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{index}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./results_images/{index}.jpg")

for i in selected:
    scene = val_dataset[i]
    print("************************* Scene",i, "****************************")
    gt = scene['traj'].unsqueeze(0).to(device)
    pred = model(scene['speed'].unsqueeze(0).to(device),scene['angle'].unsqueeze(0).to(device))
    gt_ = list(gt.detach().cpu().numpy()[0])
    pred_ = list(pred.detach().cpu().numpy()[0])
    print("GT  : ", gt_)
    print("Pred: ", pred_)
    print("MSE Loss: ", nn.MSELoss()(gt,pred).detach().cpu().numpy())
    print("L1  Loss: ", loss_func(gt,pred).detach().cpu().numpy())
    visualize(gt_,pred_,i)
    