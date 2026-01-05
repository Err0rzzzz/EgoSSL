import os
import re
import random
import numpy as np
import pandas as pd
import cv2

# =========================
# Paths
# =========================
DATA_ROOT = r"F:/EgoSSL/epic-tfc/data/EPIC_KITCHENS".replace("EPIC_KITCHENS","EPIC-KITCHENS")
ANN_ROOT  = r"F:/EgoSSL/epic-tfc/annotations/EPIC-Kitchens-100-Annotations"
VAL_CSV   = os.path.join(ANN_ROOT, "EPIC_100_validation.csv")

OUT_ROOT  = r"F:/EgoSSL/epic-tfc/stage2_valdataset"
OUT_NPY_DIR = os.path.join(OUT_ROOT, "dataset_npys")
OUT_INDEX_CSV = os.path.join(OUT_ROOT, "index.csv")
OUT_MONTAGE_DIR = os.path.join(OUT_ROOT, "montages")

# =========================
# Config
# =========================
# 建议先用同一个视频，减少解码/路径变量；后续再扩到多视频
VIDEO_ID = "P01_05"

T = 64
N_SAMPLES = 200          # val 小集即可
SAVE_MONTAGE_TOPK = 20

SEED = 42
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 8.0
MIN_VIDEO_SIZE_MB = 150

random.seed(SEED)
np.random.seed(SEED)

def video_path_from_id(video_id: str) -> str:
    p = video_id.split("_")[0]
    return os.path.join(DATA_ROOT, p, "videos", f"{video_id}.MP4")

def ts_to_seconds(ts: str) -> float:
    ts = str(ts).strip()
    m = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", ts)
    if not m:
        raise ValueError(f"Unrecognized timestamp format: {ts}")
    hh, mm, ss = m.groups()
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def pick_start_end_cols(df: pd.DataFrame):
    start_candidates = ["start_timestamp", "start_time", "start"]
    end_candidates   = ["stop_timestamp", "end_timestamp", "stop_time", "end", "stop"]
    start_col = next((c for c in start_candidates if c in df.columns), None)
    end_col   = next((c for c in end_candidates   if c in df.columns), None)
    if start_col is None or end_col is None:
        raise RuntimeError(f"Cannot find start/end columns. CSV columns: {df.columns.tolist()}")
    return start_col, end_col

def file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except:
        return 0.0

def is_video_readable(path: str) -> bool:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return False
    ok, frame = cap.read()
    cap.release()
    return ok and frame is not None

def uniform_sample_indices(start_f: int, end_f: int, num: int) -> np.ndarray:
    if end_f <= start_f:
        end_f = start_f + 1
    return np.linspace(start_f, end_f - 1, num=num).round().astype(int)

def save_montage(frames_bgr, out_path, cols=8):
    valid = [f for f in frames_bgr if f is not None]
    if len(valid) == 0:
        return
    h0, w0 = valid[0].shape[:2]
    fixed = []
    for f in frames_bgr:
        if f is None:
            fixed.append(np.zeros((h0, w0, 3), dtype=np.uint8))
        else:
            if f.shape[:2] != (h0, w0):
                f = cv2.resize(f, (w0, h0))
            fixed.append(f)

    rows = int(np.ceil(len(fixed) / cols))
    canvas = np.zeros((rows * h0, cols * w0, 3), dtype=np.uint8)
    for i, f in enumerate(fixed):
        r = i // cols
        c = i % cols
        canvas[r*h0:(r+1)*h0, c*w0:(c+1)*w0] = f
    cv2.imwrite(out_path, canvas)

def extract_features_for_segment(cap, fps, n_frames, start_sec, end_sec, T):
    start_f = max(0, int(np.floor(start_sec * fps)))
    end_f   = min(n_frames, int(np.ceil(end_sec * fps)))
    idxs = uniform_sample_indices(start_f, end_f, T)

    rgb_feats, motion_feats, frames_small = [], [], []
    prev_gray = None

    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            rgb_feats.append([0.0, 0.0, 0.0])
            motion_feats.append([0.0])
            frames_small.append(None)
            prev_gray = None
            continue

        # montage downsample
        h, w = frame.shape[:2]
        scale = 256 / max(h, w)
        if scale < 1.0:
            frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_small = frame
        frames_small.append(frame_small)

        bgr_mean = frame.mean(axis=(0, 1))
        rgb_mean = bgr_mean[::-1]
        rgb_feats.append(rgb_mean.tolist())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = 0.0 if prev_gray is None else float(np.mean(np.abs(gray - prev_gray)))
        motion_feats.append([motion])
        prev_gray = gray

    rgb = np.asarray(rgb_feats, dtype=np.float32)
    mot = np.asarray(motion_feats, dtype=np.float32)

    eps = 1e-6
    mot = (mot - mot.mean()) / (mot.std() + eps)

    x = np.concatenate([rgb, mot], axis=1)  # [T, 4]
    return x, frames_small, (start_f, end_f)

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(OUT_NPY_DIR, exist_ok=True)
    if SAVE_MONTAGE_TOPK > 0:
        os.makedirs(OUT_MONTAGE_DIR, exist_ok=True)

    print(f"[INFO] VAL_CSV   = {VAL_CSV}")
    print(f"[INFO] OUT_ROOT  = {OUT_ROOT}")

    df = pd.read_csv(VAL_CSV)
    start_col, end_col = pick_start_end_cols(df)

    # ---- NEW: auto-pick a local readable video from validation split ----
    df["video_path"] = df["video_id"].apply(video_path_from_id)
    df["exists"] = df["video_path"].apply(os.path.exists)
    cand = df[df["exists"]].copy()

    if len(cand) == 0:
        raise RuntimeError("No validation videos found locally. You may need to download some videos that belong to the validation split.")

    # filter by size, then by OpenCV readability
    cand["size_mb"] = cand["video_path"].apply(file_size_mb)
    cand = cand[cand["size_mb"] >= MIN_VIDEO_SIZE_MB].copy()
    if len(cand) == 0:
        raise RuntimeError(f"Validation videos exist locally but all are < {MIN_VIDEO_SIZE_MB}MB (likely partial).")

    # pick first readable video_id
    picked_video_id = None
    picked_video_path = None
    for vid, sub in cand.groupby("video_id"):
        vp = sub["video_path"].iloc[0]
        if is_video_readable(vp):
            picked_video_id = vid
            picked_video_path = vp
            break

    if picked_video_id is None:
        raise RuntimeError("Found validation videos locally, but none are readable by OpenCV (codec/partial files).")

    print(f"[INFO] picked VIDEO_ID = {picked_video_id}")
    print(f"[INFO] picked video_path = {picked_video_path}")

    # now restrict df to that picked video
    dfv = df[df["video_id"] == picked_video_id].copy()

    cap = cv2.VideoCapture(picked_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 1e-6 or n_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: fps={fps}, n_frames={n_frames}")

    def safe_sec(x):
        try:
            return ts_to_seconds(x)
        except:
            return np.nan

    dfv["start_sec"] = dfv[start_col].apply(safe_sec)
    dfv["end_sec"]   = dfv[end_col].apply(safe_sec)
    dfv = dfv.dropna(subset=["start_sec", "end_sec"]).copy()
    dfv["dur"] = dfv["end_sec"] - dfv["start_sec"]
    dfv = dfv[(dfv["dur"] >= MIN_DURATION_SEC) & (dfv["dur"] <= MAX_DURATION_SEC)].copy()

    if len(dfv) == 0:
        cap.release()
        raise RuntimeError("No segments remain after duration filtering; relax MIN/MAX_DURATION_SEC.")

    dfv = dfv.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df_sel = dfv.head(min(N_SAMPLES, len(dfv))).copy()

    records, saved = [], 0
    for _, row in df_sel.iterrows():
        start_sec = float(row["start_sec"])
        end_sec   = float(row["end_sec"])
        verb = int(row["verb_class"]) if "verb_class" in row and not pd.isna(row["verb_class"]) else None
        noun = int(row["noun_class"]) if "noun_class" in row and not pd.isna(row["noun_class"]) else None

        try:
            x, frames_small, (start_f, end_f) = extract_features_for_segment(cap, fps, n_frames, start_sec, end_sec, T)
        except:
            continue

        if np.allclose(x[:, :3].sum(), 0.0, atol=1e-3):
            continue

        tag = f"{picked_video_id}_{int(start_sec*1000)}_{int(end_sec*1000)}_T{T}_D4"
        npy_path = os.path.join(OUT_NPY_DIR, f"{tag}.npy")
        np.save(npy_path, x)

        montage_path = ""
        if SAVE_MONTAGE_TOPK > 0 and saved < SAVE_MONTAGE_TOPK:
            montage_path = os.path.join(OUT_MONTAGE_DIR, f"{tag}.jpg")
            save_montage(frames_small, montage_path, cols=8)

        records.append({
            "sample_id": tag,
            "video_id": picked_video_id,
            "video_path": picked_video_path,
            "start_timestamp": row[start_col],
            "stop_timestamp": row[end_col],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": end_sec - start_sec,
            "fps": fps,
            "verb_class": verb,
            "noun_class": noun,
            "feature_path": npy_path,
            "montage_path": montage_path,
        })
        saved += 1

    cap.release()

    idx = pd.DataFrame(records)
    idx.to_csv(OUT_INDEX_CSV, index=False)

    print("\n=== VAL DATASET BUILT ===")
    print(f"Saved samples: {saved}")
    print(f"Index CSV    : {OUT_INDEX_CSV}")
    print(idx.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
