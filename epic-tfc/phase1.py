import os
import cv2
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from collections import Counter

# ======================
# CONFIG
# ======================
DATA_ROOT = "F:/EgoSSL/epic-tfc/data/EPIC-KITCHENS"
TRAIN_CSV = "F:/EgoSSL/epic-tfc/annotations/EPIC-Kitchens-100-Annotations/EPIC_100_train.csv"
OUT_ROOT = "F:/EgoSSL/epic-tfc/phase1_outputs"

T = 64
MAX_VIDEOS = 20
MIN_FRAMES_CHECK = 120
BATCH_SIZE = 64
EPOCHS = 50
EMB_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_ROOT, exist_ok=True)

# ======================
# STEP 1: Detect usable videos
# ======================
def is_video_usable(path, min_frames=120):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False
    count = 0
    while count < min_frames:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count >= min_frames

print("[Phase1] Scanning videos...")
usable_videos = []
skipped_videos = []

for pid in os.listdir(DATA_ROOT):
    vdir = os.path.join(DATA_ROOT, pid, "videos")
    if not os.path.isdir(vdir):
        continue
    for v in os.listdir(vdir):
        if not v.endswith(".MP4"):
            continue
        full = os.path.join(vdir, v)
        vid = v.replace(".MP4", "")
        if is_video_usable(full, MIN_FRAMES_CHECK):
            usable_videos.append(vid)
        else:
            skipped_videos.append(vid)

usable_videos = sorted(usable_videos)[:MAX_VIDEOS]

with open(os.path.join(OUT_ROOT, "logs_usable_videos.txt"), "w") as f:
    f.write("\n".join(usable_videos))

print(f"[INFO] usable videos: {len(usable_videos)}")

# ======================
# STEP 2: Build dataset
# ======================
df = pd.read_csv(TRAIN_CSV)
df = df[df["video_id"].isin(usable_videos)]
df = df.sample(frac=1, random_state=0)

dataset_dir = os.path.join(OUT_ROOT, "dataset")
os.makedirs(dataset_dir, exist_ok=True)
npy_dir = os.path.join(dataset_dir, "dataset_npys")
os.makedirs(npy_dir, exist_ok=True)

records = []

def extract_ts(video_path, start_s, stop_s):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    s = int(start_s * fps)
    e = int(stop_s * fps)
    idxs = np.linspace(s, e, T, endpoint=False).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224,224,3), np.uint8)
        frames.append(frame)
    cap.release()

    rgb = np.array([f.mean(axis=(0,1)) for f in frames])
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=(1,2))
    motion = np.concatenate([[0], motion])
    motion = (motion - motion.mean()) / (motion.std() + 1e-6)
    return np.concatenate([rgb, motion[:,None]], axis=1)

print("[Phase1] Building dataset...")
for _, r in tqdm(df.iterrows(), total=len(df)):
    vid = r["video_id"]
    vpath = os.path.join(DATA_ROOT, vid.split("_")[0], "videos", vid + ".MP4")
    try:
        x = extract_ts(vpath, r["start_timestamp"], r["stop_timestamp"])
        fname = f"{vid}_{int(r['start_timestamp']*1000)}.npy"
        np.save(os.path.join(npy_dir, fname), x)
        records.append({
            "file": fname,
            "verb": r["verb_class"]
        })
    except:
        continue

index = pd.DataFrame(records)
index.to_csv(os.path.join(dataset_dir, "index.csv"), index=False)

print(f"[INFO] dataset size: {len(index)}")

# ======================
# STEP 3: SSL model
# ======================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, EMB_DIM)

    def forward(self, x):
        x = x.transpose(1,2)
        h = self.net(x).squeeze(-1)
        z = F.normalize(self.fc(h), dim=1)
        return z

def nt_xent(z1, z2, t=0.2):
    z = torch.cat([z1,z2],0)
    sim = F.cosine_similarity(z[:,None,:], z[None,:,:], dim=2)
    N = z1.size(0)
    mask = torch.eye(2*N, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.cat([torch.diag(sim,N), torch.diag(sim,-N)])
    loss = -torch.log(torch.exp(pos/t) / torch.exp(sim/t).sum(dim=1))
    return loss.mean()

class SSLDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        x = np.load(os.path.join(npy_dir, self.df.iloc[i]["file"]))
        x = torch.tensor(x, dtype=torch.float32)
        return x + 0.01*torch.randn_like(x), x

loader = DataLoader(SSLDataset(index), batch_size=BATCH_SIZE, shuffle=True)

model = Encoder().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), 1e-3)
losses = []

print("[Phase1] SSL training...")
for ep in range(EPOCHS):
    for x1,x2 in loader:
        x1,x2 = x1.to(DEVICE), x2.to(DEVICE)
        loss = nt_xent(model(x1), model(x2))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

plt.plot(losses)
plt.title("SSL Loss")
plt.savefig(os.path.join(OUT_ROOT,"ssl_loss.png"))

torch.save(model.state_dict(), os.path.join(OUT_ROOT,"encoder.pt"))

# ======================
# STEP 4: Linear eval
# ======================
print("[Phase1] Linear eval...")
X,Y = [],[]
for _,r in index.iterrows():
    x = np.load(os.path.join(npy_dir, r["file"]))
    with torch.no_grad():
        z = model(torch.tensor(x[None],dtype=torch.float32).to(DEVICE)).cpu().numpy()[0]
    X.append(z); Y.append(int(r["verb"]))

X,Y = np.array(X), np.array(Y)
idx = np.random.permutation(len(X))
split = int(0.8*len(X))
clf = LogisticRegression(max_iter=2000).fit(X[idx[:split]], Y[idx[:split]])
acc = clf.score(X[idx[split:]], Y[idx[split:]])

print(f"[RESULT] linear eval acc = {acc:.4f}")

# t-SNE
tsne = TSNE(n_components=2, perplexity=30).fit_transform(X)
plt.figure()
plt.scatter(tsne[:,0], tsne[:,1], c=Y, s=5)
plt.title("Embedding t-SNE")
plt.savefig(os.path.join(OUT_ROOT,"tsne.png"))
