import os
import math
import random
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================
# Paths
# =========================
TRAIN_INDEX = r"F:/EgoSSL/epic-tfc/stage2_minidataset/index.csv"
OUT_DIR     = r"F:/EgoSSL/epic-tfc/stage3_pretrain"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Train config
# =========================
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
TEMPERATURE = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data shape: [T, D]
T_EXPECTED = 64
D_EXPECTED = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Augmentations (time-domain)
# -------------------------
def aug_jitter(x, sigma=0.02):
    return x + torch.randn_like(x) * sigma

def aug_scaling(x, sigma=0.1):
    # scale per-channel
    scale = torch.normal(mean=torch.ones(x.shape[-1], device=x.device), std=sigma)
    return x * scale

def aug_time_mask(x, max_masks=2, max_width=8):
    # x: [T, D]
    T = x.shape[0]
    out = x.clone()
    n = random.randint(0, max_masks)
    for _ in range(n):
        w = random.randint(1, max_width)
        s = random.randint(0, max(0, T - w))
        out[s:s+w] = 0
    return out

def make_view(x):
    # compose a few light augs
    x = aug_jitter(x, sigma=0.02)
    x = aug_scaling(x, sigma=0.05)
    x = aug_time_mask(x, max_masks=2, max_width=8)
    return x

# -------------------------
# Dataset
# -------------------------
class NpyIndexDataset(Dataset):
    def __init__(self, index_csv: str):
        self.df = pd.read_csv(index_csv)
        # keep only existing npy
        self.df = self.df[self.df["feature_path"].apply(lambda p: os.path.exists(p))].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = np.load(row["feature_path"]).astype(np.float32)  # [T, D]
        if x.shape != (T_EXPECTED, D_EXPECTED):
            raise RuntimeError(f"Unexpected shape {x.shape} in {row['feature_path']}")
        return torch.from_numpy(x)  # [T, D]

# -------------------------
# Model: 1D CNN encoder + projection head
# -------------------------
class Encoder1D(nn.Module):
    def __init__(self, d_in=4, d_hidden=128, d_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, d_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        h = self.net(x)             # [B, H, T]
        h = self.pool(h).squeeze(-1)  # [B, H]
        z = self.proj(h)            # [B, d_out]
        z = F.normalize(z, dim=-1)
        return z

# -------------------------
# NT-Xent loss (SimCLR-style)
# -------------------------
def nt_xent_loss(z1, z2, temperature=0.2):
    # z1,z2: [B, D]
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.t()) / temperature  # [2B,2B]
    # mask self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # positives: i<->i+B
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    loss = F.cross_entropy(sim, pos)
    return loss

def collate_views(batch):
    # batch: list of [T,D]
    x = torch.stack(batch, dim=0)         # [B,T,D]
    x = x.to(DEVICE)
    v1 = torch.stack([make_view(xx) for xx in x], dim=0)
    v2 = torch.stack([make_view(xx) for xx in x], dim=0)
    return v1, v2

def main():
    ds = NpyIndexDataset(TRAIN_INDEX)
    print(f"[INFO] Train samples: {len(ds)} from {TRAIN_INDEX}")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True,
                    collate_fn=lambda b: collate_views(b))

    model = Encoder1D(d_in=D_EXPECTED, d_hidden=128, d_out=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = 1e9
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for v1, v2 in dl:
            z1 = model(v1)
            z2 = model(v2)
            loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        m = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch {ep:03d} | loss {m:.4f}")

        # save best
        if m < best:
            best = m
            ckpt = {
                "model": model.state_dict(),
                "epoch": ep,
                "loss": best,
                "cfg": {"T": T_EXPECTED, "D": D_EXPECTED}
            }
            torch.save(ckpt, os.path.join(OUT_DIR, "best.pt"))

    torch.save({"model": model.state_dict()}, os.path.join(OUT_DIR, "last.pt"))
    print(f"[DONE] saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
