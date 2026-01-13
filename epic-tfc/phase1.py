import os
import cv2
import json
import time
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
from sklearn.metrics import accuracy_score
from collections import Counter

# ======================
# CONFIG
# ======================
DATA_ROOT = "F:/EgoSSL/epic-tfc/data/EPIC-KITCHENS"
TRAIN_CSV = "F:/EgoSSL/epic-tfc/annotations/EPIC-Kitchens-100-Annotations/EPIC_100_train.csv"
OUT_ROOT  = "F:/EgoSSL/epic-tfc/phase1_outputs"

T = 64
MAX_VIDEOS = 20               # 先跑 Phase-1 的规模即可
MIN_FRAMES_CHECK = 120        # 判定视频可用的最小可读帧数
BATCH_SIZE = 64
EPOCHS = 30                   # 先别太大；数据一扩，再加
EMB_DIM = 128
LR = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_ROOT, exist_ok=True)

# ======================
# Utils
# ======================
def ts_to_seconds(ts):
    """EPIC timestamp like 'HH:MM:SS.xx' -> seconds(float)."""
    if isinstance(ts, (int, float, np.floating)):
        return float(ts)
    ts = str(ts).strip()
    if ts == "" or ts.lower() == "nan":
        return None
    # allow 'HH:MM:SS' or 'HH:MM:SS.xx'
    parts = ts.split(":")
    if len(parts) != 3:
        return None
    h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
    return h * 3600 + m * 60 + s

def video_path_from_id(video_id):
    pid = video_id.split("_")[0]  # 'P01'
    return os.path.join(DATA_ROOT, pid, "videos", video_id + ".MP4")

def is_video_usable(path, min_frames=120):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 1e-6 or n_frames <= 0:
        cap.release()
        return False
    count = 0
    while count < min_frames:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count >= min_frames

# ======================
# STEP 1: Detect usable videos
# ======================
print("[Phase1] Scanning videos...")

usable_videos = []
skipped = []

for pid in os.listdir(DATA_ROOT):
    vdir = os.path.join(DATA_ROOT, pid, "videos")
    if not os.path.isdir(vdir):
        continue
    for fn in os.listdir(vdir):
        if not fn.upper().endswith(".MP4"):
            continue
        vid = fn[:-4]
        full = os.path.join(vdir, fn)
        if is_video_usable(full, MIN_FRAMES_CHECK):
            usable_videos.append(vid)
        else:
            skipped.append(vid)

usable_videos = sorted(usable_videos)[:MAX_VIDEOS]

with open(os.path.join(OUT_ROOT, "usable_videos.txt"), "w") as f:
    f.write("\n".join(usable_videos))
with open(os.path.join(OUT_ROOT, "skipped_videos.txt"), "w") as f:
    f.write("\n".join(sorted(skipped)))

print(f"[INFO] usable videos: {len(usable_videos)}")
for i, v in enumerate(usable_videos[:10]):
    print(f"  {i:02d}: {v}")

# ======================
# STEP 2: Build dataset (time-series features)
# ======================
df = pd.read_csv(TRAIN_CSV)

# EPIC uses 'start_timestamp'/'stop_timestamp' as strings
if "start_timestamp" not in df.columns or "stop_timestamp" not in df.columns:
    raise RuntimeError("Cannot find start_timestamp/stop_timestamp columns in TRAIN_CSV.")

df = df[df["video_id"].isin(usable_videos)].copy()
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

dataset_dir = os.path.join(OUT_ROOT, "dataset")
os.makedirs(dataset_dir, exist_ok=True)
npy_dir = os.path.join(dataset_dir, "dataset_npys")
os.makedirs(npy_dir, exist_ok=True)

records = []
failed = 0
printed_first_error = False

def extract_ts(video_path, start_sec, stop_sec):
    """Return (T,4) feature: RGB mean(3) + motion(1)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("OpenCV cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 1e-6 or n_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid fps/n_frames")

    s = int(start_sec * fps)
    e = int(stop_sec * fps)
    s = max(0, min(s, n_frames - 1))
    e = max(0, min(e, n_frames))
    if e <= s + 2:
        cap.release()
        raise RuntimeError("Too short segment")

    # sample frame indices
    idxs = np.linspace(s, e, T, endpoint=False).astype(int)
    idxs = np.clip(idxs, 0, n_frames - 1)

    frames = []
    # simple way: random access; OK for phase1
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((224, 224, 3), np.uint8)
        frames.append(frame)
    cap.release()

    rgb = np.array([f.mean(axis=(0, 1)) for f in frames], dtype=np.float32)  # (T,3)
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], dtype=np.float32)  # (T,H,W)
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    motion = np.concatenate([[0.0], motion]).astype(np.float32)  # (T,)
    motion = (motion - motion.mean()) / (motion.std() + 1e-6)
    x = np.concatenate([rgb, motion[:, None]], axis=1)  # (T,4)
    return x

print("[Phase1] Building dataset...")
for _, r in tqdm(df.iterrows(), total=len(df)):
    vid = r["video_id"]
    vpath = video_path_from_id(vid)
    if not os.path.exists(vpath):
        failed += 1
        continue

    start_sec = ts_to_seconds(r["start_timestamp"])
    stop_sec  = ts_to_seconds(r["stop_timestamp"])
    if start_sec is None or stop_sec is None:
        failed += 1
        continue

    try:
        x = extract_ts(vpath, start_sec, stop_sec)
        if x.shape != (T, 4):
            failed += 1
            continue

        fname = f"{vid}_{int(start_sec*1000)}_{int(stop_sec*1000)}.npy"
        np.save(os.path.join(npy_dir, fname), x)

        records.append({"file": fname, "verb": int(r["verb_class"])})
    except Exception as e:
        failed += 1
        if not printed_first_error:
            printed_first_error = True
            print("[DEBUG] First extraction error example:")
            print("  video_id:", vid)
            print("  video_path:", vpath)
            print("  start_timestamp:", r["start_timestamp"], "->", start_sec)
            print("  stop_timestamp :", r["stop_timestamp"], "->", stop_sec)
            print("  error:", repr(e))
        continue

index = pd.DataFrame(records)
index_path = os.path.join(dataset_dir, "index.csv")
index.to_csv(index_path, index=False)

print(f"[INFO] dataset size: {len(index)}")
print(f"[INFO] failed rows  : {failed}")
if len(index) == 0:
    raise RuntimeError("Dataset is empty. Check DEBUG error above and verify timestamps/paths.")

# ======================
# STEP 3: SSL model
# ======================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, EMB_DIM)

    def forward(self, x):
        # x: [B,T,4] -> [B,4,T]
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        z = F.normalize(self.fc(h), dim=1)
        return z

def augment(x):
    # x: [B,T,4]
    # simple jitter + scaling
    noise = 0.01 * torch.randn_like(x)
    scale = (1.0 + 0.05 * torch.randn(x.size(0), 1, x.size(2), device=x.device))
    return (x + noise) * scale

def nt_xent(z1, z2, temp=0.2):
    z = torch.cat([z1, z2], dim=0)  # [2B,D]
    sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=2)  # [2B,2B]
    B = z1.size(0)
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # [2B]
    loss = -torch.log(torch.exp(pos / temp) / torch.exp(sim / temp).sum(dim=1))
    return loss.mean()

class SSLDataset(Dataset):
    def __init__(self, df, root):
        self.df = df.reset_index(drop=True)
        self.root = root
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        x = np.load(os.path.join(self.root, self.df.iloc[i]["file"])).astype(np.float32)  # (T,4)
        return torch.from_numpy(x)

ds = SSLDataset(index, npy_dir)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = Encoder().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

print("[Phase1] SSL training...")
losses = []
start_time = time.time()

model.train()
for ep in range(1, EPOCHS + 1):
    ep_losses = []
    for x in loader:
        x = x.to(DEVICE)              # [B,T,4]
        x1 = augment(x)
        x2 = augment(x)

        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent(z1, z2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
        ep_losses.append(float(loss.item()))

    print(f"[SSL] epoch {ep:03d}/{EPOCHS}  loss={np.mean(ep_losses):.4f}")

ssl_dir = os.path.join(OUT_ROOT, "ssl")
os.makedirs(ssl_dir, exist_ok=True)

plt.figure()
plt.plot(losses)
plt.title("SSL NT-Xent Loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig(os.path.join(ssl_dir, "loss_curve.png"), dpi=200)

ckpt_path = os.path.join(ssl_dir, "encoder.pt")
torch.save(model.state_dict(), ckpt_path)

print(f"[INFO] SSL done. steps={len(losses)}  time={(time.time()-start_time):.1f}s")
print(f"[INFO] saved encoder: {ckpt_path}")

# ======================
# STEP 4: Linear eval (hold-out) + baselines + plots
# ======================
print("[Phase1] Linear eval...")

# build embeddings
model.eval()
X, Y = [], []
with torch.no_grad():
    for _, r in index.iterrows():
        x = np.load(os.path.join(npy_dir, r["file"])).astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)  # [1,T,4]
        z = model(xt).squeeze(0).cpu().numpy()
        X.append(z); Y.append(int(r["verb"]))
X = np.asarray(X)
Y = np.asarray(Y)

# majority baseline
cnt = Counter(Y.tolist())
major = cnt.most_common(1)[0]
major_acc = major[1] / len(Y)

# hold-out split
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(X))
n_val = max(1, int(0.2 * len(X)))
va_idx = perm[:n_val]
tr_idx = perm[n_val:]

clf = LogisticRegression(max_iter=3000, n_jobs=1)
clf.fit(X[tr_idx], Y[tr_idx])
pred = clf.predict(X[va_idx])
acc = accuracy_score(Y[va_idx], pred)

eval_dir = os.path.join(OUT_ROOT, "eval")
os.makedirs(eval_dir, exist_ok=True)

with open(os.path.join(eval_dir, "results.txt"), "w") as f:
    f.write(f"num_samples: {len(X)}\n")
    f.write(f"num_classes: {len(cnt)}\n")
    f.write(f"major_class: {major[0]} count={major[1]} majority_acc={major_acc:.4f}\n")
    f.write(f"holdout_acc: {acc:.4f}\n")

print(f"[RESULT] majority baseline acc = {major_acc:.4f}")
print(f"[RESULT] verb linear eval acc (hold-out) = {acc:.4f}")

# t-SNE plot (optional but nice)
try:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
    Z2 = tsne.fit_transform(X)

    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=Y, s=6)
    plt.title("Embedding t-SNE (colored by verb)")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "tsne.png"), dpi=200)
except Exception as e:
    print("[WARN] t-SNE skipped:", repr(e))

print("\n=== Phase-1 DONE ===")
print(f"Outputs in: {OUT_ROOT}")
