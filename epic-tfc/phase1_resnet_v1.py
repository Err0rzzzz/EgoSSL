import os
import time
import json
import random
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE

# -----------------------
# CONFIG (edit here only)
# -----------------------
DATA_ROOT = "F:/EgoSSL/epic-tfc/data/EPIC-KITCHENS"
TRAIN_CSV = "F:/EgoSSL/epic-tfc/annotations/EPIC-Kitchens-100-Annotations/EPIC_100_train.csv"
OUT_ROOT  = "F:/EgoSSL/epic-tfc/phase1_resnet_outputs"

T = 64
MAX_VIDEOS = 20                 # 先小规模验证；后续可加到 50/100
MIN_FRAMES_CHECK = 120
MAX_ROWS_FROM_CSV = 4000        # 控制 segments 数量，防止第一轮跑太大
MIN_SEG_DUR_S = 0.8             # 过滤太短 segment
MAX_SEG_DUR_S = 8.0             # 过滤太长 segment（可按需调）

# Feature extraction
USE_MOTION_1D = False           # True: 512 + 1 -> 513；False: only 512
RESNET_ARCH = "resnet18"        # resnet18 最快最稳
FRAME_SIZE = 224                # ResNet 输入
BATCH_FRAMES = 128              # 每次喂 ResNet 的帧数（按 GPU/显存调）

# SSL training
BATCH_SIZE = 64
EPOCHS = 30
EMB_DIM = 128
LR = 1e-3
TEMP = 0.2

# Evaluation
EVAL_SEEDS = [0, 1, 2, 3, 4]
HOLDOUT_RATIO = 0.2
TOPK_VERBS = 10                 # 0 表示不做 topK 子集评测；建议 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_ROOT, exist_ok=True)

# -----------------------
# Utils
# -----------------------
def ts_to_seconds(ts):
    """'HH:MM:SS.xx' -> seconds(float)."""
    if isinstance(ts, (int, float, np.floating)):
        return float(ts)
    ts = str(ts).strip()
    if ts == "" or ts.lower() == "nan":
        return None
    parts = ts.split(":")
    if len(parts) != 3:
        return None
    h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
    return h * 3600 + m * 60 + s

def video_path_from_id(video_id):
    pid = video_id.split("_")[0]
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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------
# ResNet feature extractor
# -----------------------
def build_resnet_feature_extractor(arch="resnet18", device="cuda"):
    import torchvision.models as models
    import torchvision.transforms as T

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = 512
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    # remove FC, keep avgpool output
    model.fc = nn.Identity()
    model.eval().to(device)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((FRAME_SIZE, FRAME_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    return model, transform, feat_dim

# -----------------------
# STEP 1: scan usable videos
# -----------------------
print("[Phase1-ResNet] Scanning videos...")
usable_videos = []
skipped_videos = []

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
            skipped_videos.append(vid)

usable_videos = sorted(usable_videos)[:MAX_VIDEOS]

with open(os.path.join(OUT_ROOT, "usable_videos.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(usable_videos))
with open(os.path.join(OUT_ROOT, "skipped_videos.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(skipped_videos)))

print(f"[INFO] usable videos: {len(usable_videos)} (capped by MAX_VIDEOS={MAX_VIDEOS})")
if len(usable_videos) == 0:
    raise RuntimeError("No usable videos found. You likely have only partial videos locally.")

# -----------------------
# STEP 2: build dataset index (segments)
# -----------------------
df = pd.read_csv(TRAIN_CSV)
df = df[df["video_id"].isin(usable_videos)].copy()

# parse start/stop to seconds
df["start_s"] = df["start_timestamp"].apply(ts_to_seconds)
df["stop_s"]  = df["stop_timestamp"].apply(ts_to_seconds)
df = df.dropna(subset=["start_s", "stop_s"]).copy()
df["dur_s"] = df["stop_s"] - df["start_s"]
df = df[(df["dur_s"] >= MIN_SEG_DUR_S) & (df["dur_s"] <= MAX_SEG_DUR_S)].copy()

# shuffle + cap rows
df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
if MAX_ROWS_FROM_CSV and len(df) > MAX_ROWS_FROM_CSV:
    df = df.iloc[:MAX_ROWS_FROM_CSV].copy()

print(f"[INFO] candidate segments: {len(df)} (after duration filter + cap={MAX_ROWS_FROM_CSV})")
if len(df) == 0:
    raise RuntimeError("No segments after filtering. Relax duration constraints or increase usable videos.")

# output dirs
dataset_dir = os.path.join(OUT_ROOT, "dataset")
npy_dir = os.path.join(dataset_dir, "dataset_npys")
os.makedirs(npy_dir, exist_ok=True)

# -----------------------
# Video sampling & frame extraction
# -----------------------
def sample_frame_indices(fps, n_frames, start_s, stop_s, T=64):
    s = int(start_s * fps)
    e = int(stop_s  * fps)
    s = max(0, min(s, n_frames - 1))
    e = max(0, min(e, n_frames))
    if e <= s + 2:
        return None
    idxs = np.linspace(s, e, T, endpoint=False).astype(int)
    idxs = np.clip(idxs, 0, n_frames - 1)
    return idxs

def read_frames_at_indices(video_path, idxs):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret or frame is None:
            # mark as failure
            cap.release()
            return None
        frames.append(frame)  # BGR uint8
    cap.release()
    return frames

def compute_motion_1d(frames_bgr):
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames_bgr]
    gray = np.stack(gray, axis=0)
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=(1,2))
    motion = np.concatenate([[0.0], motion]).astype(np.float32)
    motion = (motion - motion.mean()) / (motion.std() + 1e-6)
    return motion  # (T,)

# -----------------------
# Extract ResNet features for each segment and save npy
# -----------------------
print("[Phase1-ResNet] Building dataset with ResNet frame embeddings...")
resnet, transform, FEAT_DIM = build_resnet_feature_extractor(RESNET_ARCH, DEVICE)
D = FEAT_DIM + (1 if USE_MOTION_1D else 0)

records = []
failed = 0
first_err_printed = False

def frames_to_resnet_feats(frames_bgr):
    """
    frames_bgr: list of length T, each HxWx3 BGR uint8
    returns: (T, FEAT_DIM) float32
    """
    # convert to RGB and apply transforms
    imgs = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        imgs.append(transform(rgb))  # tensor [3,224,224]
    x = torch.stack(imgs, dim=0)  # [T,3,224,224]
    feats = []
    with torch.no_grad():
        for i in range(0, x.size(0), BATCH_FRAMES):
            xb = x[i:i+BATCH_FRAMES].to(DEVICE)
            fb = resnet(xb)  # [b,FEAT_DIM]
            feats.append(fb.detach().cpu())
    feats = torch.cat(feats, dim=0).numpy().astype(np.float32)  # (T,FEAT_DIM)
    return feats

for _, r in tqdm(df.iterrows(), total=len(df)):
    vid = r["video_id"]
    vpath = video_path_from_id(vid)
    if not os.path.exists(vpath):
        failed += 1
        continue

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        cap.release()
        failed += 1
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    idxs = sample_frame_indices(fps, n_frames, r["start_s"], r["stop_s"], T=T)
    if idxs is None:
        failed += 1
        continue

    try:
        frames = read_frames_at_indices(vpath, idxs)
        if frames is None:
            failed += 1
            continue

        feats = frames_to_resnet_feats(frames)  # (T,FEAT_DIM)

        if USE_MOTION_1D:
            motion = compute_motion_1d(frames)[:, None]  # (T,1)
            x = np.concatenate([feats, motion], axis=1).astype(np.float32)  # (T,D)
        else:
            x = feats

        if x.shape != (T, D):
            failed += 1
            continue

        fname = f"{vid}_{int(r['start_s']*1000)}_{int(r['stop_s']*1000)}_T{T}_D{D}.npy"
        np.save(os.path.join(npy_dir, fname), x)

        records.append({"file": fname, "verb": int(r["verb_class"])})
    except Exception as e:
        failed += 1
        if not first_err_printed:
            first_err_printed = True
            print("[DEBUG] First extraction error example:")
            print("  video_id:", vid)
            print("  video_path:", vpath)
            print("  start:", r["start_timestamp"], "->", r["start_s"])
            print("  stop :", r["stop_timestamp"],  "->", r["stop_s"])
            print("  err  :", repr(e))
        continue

index = pd.DataFrame(records)
index_path = os.path.join(dataset_dir, "index.csv")
index.to_csv(index_path, index=False)

print(f"[INFO] dataset size: {len(index)}  (failed rows: {failed})")
if len(index) == 0:
    raise RuntimeError("Dataset is empty. Most likely all segments hit unreadable frames or videos are still partial.")

# -----------------------
# STEP 3: SSL model (same as before but input D changes)
# -----------------------
class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim=128):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        # x: [B,T,D] -> [B,D,T]
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        z = F.normalize(self.fc(h), dim=1)
        return z

def augment(x):
    # x: [B,T,D]
    # jitter + scaling + time mask
    noise = 0.02 * torch.randn_like(x)
    scale = 1.0 + 0.10 * torch.randn(x.size(0), 1, x.size(2), device=x.device)
    y = (x + noise) * scale
    # time mask
    if x.size(1) >= 8:
        B, Tt, Dd = y.shape
        mlen = max(1, Tt // 8)
        for b in range(B):
            s = torch.randint(0, Tt - mlen + 1, (1,), device=y.device).item()
            y[b, s:s+mlen, :] = 0
    return y

def nt_xent(z1, z2, temp=0.2):
    z = torch.cat([z1, z2], dim=0)  # [2B,emb]
    sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=2)  # [2B,2B]
    B = z1.size(0)
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)
    loss = -torch.log(torch.exp(pos / temp) / torch.exp(sim / temp).sum(dim=1))
    return loss.mean()

class SSLDataset(Dataset):
    def __init__(self, df, root):
        self.df = df.reset_index(drop=True)
        self.root = root
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        x = np.load(os.path.join(self.root, self.df.iloc[i]["file"])).astype(np.float32)  # (T,D)
        return torch.from_numpy(x)

ssl_ds = SSLDataset(index, npy_dir)
ssl_loader = DataLoader(ssl_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = Encoder(in_dim=D, emb_dim=EMB_DIM).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

print("[Phase1-ResNet] SSL training...")
losses = []
t0 = time.time()

model.train()
for ep in range(1, EPOCHS + 1):
    ep_losses = []
    for x in ssl_loader:
        x = x.to(DEVICE)  # [B,T,D]
        x1 = augment(x)
        x2 = augment(x)
        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent(z1, z2, temp=TEMP)

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
plt.title("SSL NT-Xent Loss (ResNet features)")
plt.xlabel("step")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig(os.path.join(ssl_dir, "loss_curve.png"), dpi=200)

ckpt_path = os.path.join(ssl_dir, "encoder.pt")
torch.save(model.state_dict(), ckpt_path)

print(f"[INFO] SSL done. steps={len(losses)}  time={(time.time()-t0):.1f}s")
print(f"[INFO] saved encoder: {ckpt_path}")

# -----------------------
# STEP 4: Embedding + evaluation (multi-seed hold-out)
# -----------------------
print("[Phase1-ResNet] Building embeddings for eval...")

model.eval()
X, Y = [], []
with torch.no_grad():
    for _, r in tqdm(index.iterrows(), total=len(index)):
        x = np.load(os.path.join(npy_dir, r["file"])).astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)  # [1,T,D]
        z = model(xt).squeeze(0).cpu().numpy()
        X.append(z)
        Y.append(int(r["verb"]))
X = np.asarray(X)
Y = np.asarray(Y)

cnt = Counter(Y.tolist())
major_class, major_count = cnt.most_common(1)[0]
major_acc = major_count / len(Y)

def eval_holdout(seed, X, Y):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(HOLDOUT_RATIO * len(X)))
    va = perm[:n_val]
    tr = perm[n_val:]
    clf = LogisticRegression(max_iter=4000, n_jobs=1)
    clf.fit(X[tr], Y[tr])
    pred = clf.predict(X[va])
    acc = accuracy_score(Y[va], pred)
    f1m = f1_score(Y[va], pred, average="macro")
    return acc, f1m

accs, f1ms = [], []
for s in EVAL_SEEDS:
    a, f1m = eval_holdout(s, X, Y)
    accs.append(a); f1ms.append(f1m)

eval_dir = os.path.join(OUT_ROOT, "eval")
os.makedirs(eval_dir, exist_ok=True)

summary = {
    "num_samples": int(len(X)),
    "num_classes": int(len(cnt)),
    "major_class": int(major_class),
    "majority_acc": float(major_acc),
    "holdout_ratio": float(HOLDOUT_RATIO),
    "seeds": list(EVAL_SEEDS),
    "accs": [float(x) for x in accs],
    "macro_f1s": [float(x) for x in f1ms],
    "acc_mean": float(np.mean(accs)),
    "acc_std": float(np.std(accs)),
    "macro_f1_mean": float(np.mean(f1ms)),
    "macro_f1_std": float(np.std(f1ms)),
    "feature_dim_D": int(D),
    "T": int(T),
    "resnet_arch": RESNET_ARCH,
    "use_motion_1d": bool(USE_MOTION_1D),
}

with open(os.path.join(eval_dir, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[RESULT] majority baseline acc = {major_acc:.4f}")
print(f"[RESULT] hold-out acc (mean±std over {len(EVAL_SEEDS)} seeds) = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"[RESULT] hold-out macro-F1 (mean±std) = {np.mean(f1ms):.4f} ± {np.std(f1ms):.4f}")

# optional: Top-K verbs subset eval
if TOPK_VERBS and TOPK_VERBS > 0:
    topk = [c for c, _ in cnt.most_common(TOPK_VERBS)]
    mask = np.isin(Y, topk)
    Xk, Yk = X[mask], Y[mask]
    cntk = Counter(Yk.tolist())
    majk = cntk.most_common(1)[0][1] / len(Yk)

    accs_k, f1ms_k = [], []
    for s in EVAL_SEEDS:
        a, f1m = eval_holdout(s, Xk, Yk)
        accs_k.append(a); f1ms_k.append(f1m)

    topk_summary = {
        "topK": int(TOPK_VERBS),
        "num_samples": int(len(Xk)),
        "num_classes": int(len(cntk)),
        "majority_acc": float(majk),
        "acc_mean": float(np.mean(accs_k)),
        "acc_std": float(np.std(accs_k)),
        "macro_f1_mean": float(np.mean(f1ms_k)),
        "macro_f1_std": float(np.std(f1ms_k)),
    }
    with open(os.path.join(eval_dir, f"summary_top{TOPK_VERBS}.json"), "w", encoding="utf-8") as f:
        json.dump(topk_summary, f, indent=2)

    print(f"[RESULT] TOP{TOPK_VERBS} subset majority acc = {majk:.4f}")
    print(f"[RESULT] TOP{TOPK_VERBS} subset acc mean±std = {np.mean(accs_k):.4f} ± {np.std(accs_k):.4f}")
    print(f"[RESULT] TOP{TOPK_VERBS} subset macro-F1 mean±std = {np.mean(f1ms_k):.4f} ± {np.std(f1ms_k):.4f}")

# t-SNE plot (for visualization)
try:
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    Z2 = tsne.fit_transform(X)
    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=Y, s=6)
    plt.title("Embedding t-SNE (colored by verb)")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "tsne.png"), dpi=200)
except Exception as e:
    print("[WARN] t-SNE skipped:", repr(e))

print("\n=== Phase-1 ResNet DONE ===")
print(f"Outputs in: {OUT_ROOT}")
