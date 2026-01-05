import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

INDEX_CSV  = r"F:/EgoSSL/epic-tfc/stage2_minidataset/index.csv"
CKPT_PATH  = r"F:/EgoSSL/epic-tfc/stage3_pretrain/best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_EXPECTED, D_EXPECTED = 64, 4
SEED = 42
VAL_RATIO = 0.2

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
        x = x.transpose(1, 2)      # [B,D,T]
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z

def load_index(path):
    df = pd.read_csv(path)
    df = df[df["feature_path"].apply(lambda p: os.path.exists(p))].copy()
    df = df.dropna(subset=["verb_class"]).copy()
    df["verb_class"] = df["verb_class"].astype(int)
    return df.reset_index(drop=True)

@torch.no_grad()
def embed(df, model):
    Xs, ys = [], []
    for _, r in df.iterrows():
        x = np.load(r["feature_path"]).astype(np.float32)
        if x.shape != (T_EXPECTED, D_EXPECTED):
            continue
        xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)  # [1,T,D]
        z = model(xt).squeeze(0).cpu().numpy()            # [128]
        Xs.append(z)
        ys.append(int(r["verb_class"]))
    return np.asarray(Xs), np.asarray(ys)

def main():
    df = load_index(INDEX_CSV)
    print(f"[INFO] total rows: {len(df)} from {INDEX_CSV}")

    # shuffle + split (hold-out)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(df))
    n_val = int(len(df) * VAL_RATIO)
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    tr = df.iloc[tr_idx].reset_index(drop=True)
    va = df.iloc[val_idx].reset_index(drop=True)
    print(f"[INFO] split: train={len(tr)}, val={len(va)} (hold-out {VAL_RATIO:.0%})")

    model = Encoder1D(d_in=D_EXPECTED).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    Xtr, ytr = embed(tr, model)
    Xva, yva = embed(va, model)
    print(f"[INFO] embedded: Xtr {Xtr.shape}, Xva {Xva.shape}")

    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    acc = accuracy_score(yva, pred)
    print(f"[RESULT] verb linear eval acc (hold-out) = {acc:.4f}")

if __name__ == "__main__":
    main()
