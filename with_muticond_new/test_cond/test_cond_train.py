import torch
import torch.nn as nn
from nuscenes_dataset import NuscenesDatasetForCogvidx
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

train_dataset = NuscenesDatasetForCogvidx("/data/wuzhirong/datasets/Nuscenes",split="train")
print(len(train_dataset))
print(train_dataset[0])
device = "cuda"
def collate_fn(examples):
    new_example = {
        "traj":[],
        "speed":[],
        "angle":[]
    }
    for x in examples:
        new_example['traj'].append(x['traj'])
        new_example['speed'].append(x['speed'])
        new_example['angle'].append(x['angle'])
    new_example['traj'] = torch.stack(new_example['traj'])
    new_example['speed'] = torch.stack(new_example['speed'])
    new_example['angle'] = torch.stack(new_example['angle'])
    return new_example
        
train_dataloader = DataLoader(
    train_dataset,
    batch_size=8192,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

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

# model = MLP()
model = torch.load("/data/wuzhirong/ckpts/test_traj_8_new/ckpt_09999.pt").to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

ckpt_folder = "/data/wuzhirong/ckpts/test_traj_8_new_1"
import os
os.makedirs(ckpt_folder,exist_ok=True)
import wandb
wandb.init(project="train_traj")

def loss_func(gt,pred):
    return nn.MSELoss()(gt,pred)
    # return nn.L1Loss()(gt,pred)

for epoch in tqdm(range(100000)):
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        gt = batch["traj"].to(device)
        pred = model(batch['speed'].to(device),batch['angle'].to(device))
        loss = loss_func(gt,pred)
        wandb.log({"loss":loss.detach().item()})
        loss.backward()
        optimizer.step()
    if (epoch+1)%100 == 0:
        torch.save(model, os.path.join(ckpt_folder,f"ckpt_{epoch:05d}.pt"))

wandb.finish()