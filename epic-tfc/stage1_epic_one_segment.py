import os
import re
import random
import numpy as np
import pandas as pd
import cv2

# =========================
# paths
# =========================
DATA_ROOT = r"F:/EgoSSL/epic-tfc/data/EPIC-KITCHENS"
ANN_ROOT  = r"F:/EgoSSL/epic-tfc/annotations/EPIC-Kitchens-100-Annotations"
TRAIN_CSV = os.path.join(ANN_ROOT, "EPIC_100_train.csv")

# Output
OUT_DIR = r"F:/EgoSSL/epic-tfc/stage1_outputs"

# Time-series length (frames)
T = 64

# Heuristics to avoid partial files
MIN_VIDEO_SIZE_MB = 150  # skip very small files (often partial)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def video_path_from_id(video_id: str) -> str:
    # video_id like "P01_05"
    p = video_id.split("_")[0]  # "P01"
    return os.path.join(DATA_ROOT, p, "videos", f"{video_id}.MP4")

def ts_to_seconds(ts: str) -> float:
    """
    Parse timestamps like HH:MM:SS(.ms)
    """
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
        raise RuntimeError(f"Cannot find start/end columns in CSV. Columns: {df.columns.tolist()}")
    return start_col, end_col

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

def extract_frames_and_rgbmean(video_path: str, start_sec: float, end_sec: float, T: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps <= 1e-6:
        cap.release()
        raise RuntimeError(f"FPS invalid for {video_path}: {fps}")

    start_f = max(0, int(np.floor(start_sec * fps)))
    end_f   = min(n_frames, int(np.ceil(end_sec * fps)))

    idxs = uniform_sample_indices(start_f, end_f, T)

    frames = []
    feats = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            # fallback zeros
            feats.append([0.0, 0.0, 0.0])
            frames.append(None)
            continue

        # store a small version of frame for montage (downsample)
        h, w = frame.shape[:2]
        scale = 256 / max(h, w)
        if scale < 1.0:
            frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_small = frame
        frames.append(frame_small)

        # feature: RGB mean (OpenCV is BGR)
        bgr_mean = frame.mean(axis=(0, 1))
        rgb_mean = bgr_mean[::-1]
        feats.append(rgb_mean.tolist())

    cap.release()
    return np.asarray(feats, dtype=np.float32), frames, fps, n_frames, start_f, end_f

def save_montage(frames, out_path, cols=8):
    """
    frames: list of small BGR frames or None
    """
    valid = [f for f in frames if f is not None]
    if len(valid) == 0:
        return

    # Replace None with black frames of same size as first valid
    h0, w0 = valid[0].shape[:2]
    fixed = []
    for f in frames:
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

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] TRAIN_CSV = {TRAIN_CSV}")
    print(f"[INFO] OUT_DIR   = {OUT_DIR}")

    df = pd.read_csv(TRAIN_CSV)
    start_col, end_col = pick_start_end_cols(df)

    # Build local video paths
    df["video_path"] = df["video_id"].apply(video_path_from_id)
    df["exists"] = df["video_path"].apply(os.path.exists)

    df_local = df[df["exists"]].copy()
    if len(df_local) == 0:
        print("[ERROR] No local videos matched expected paths.")
        print(df["video_path"].head(10).to_string(index=False))
        return

    # Filter by file size to avoid partials
    def size_mb(p):
        try:
            return os.path.getsize(p) / (1024 * 1024)
        except:
            return 0.0

    df_local["size_mb"] = df_local["video_path"].apply(size_mb)
    df_local = df_local[df_local["size_mb"] >= MIN_VIDEO_SIZE_MB].copy()
    if len(df_local) == 0:
        print(f"[ERROR] Local videos exist but all are < {MIN_VIDEO_SIZE_MB} MB. Lower MIN_VIDEO_SIZE_MB or finish downloads.")
        return

    # Prefer P01/P02 and also prefer specific video_ids you already have
    preferred_ids = ["P01_05", "P01_06", "P02_05", "P02_06", "P02_07", "P02_08", "P02_09", "P02_10"]
    df_pref = df_local[df_local["video_id"].isin(preferred_ids)]
    pool = df_pref if len(df_pref) > 0 else df_local

    # Randomize to find a readable segment quickly
    pool = pool.sample(min(len(pool), 2000), random_state=SEED).reset_index(drop=True)

    # Try rows until we find one with readable video and valid segment
    for i in range(len(pool)):
        row = pool.iloc[i]
        video_id = row["video_id"]
        vp = row["video_path"]

        # quick video readability check
        if not is_video_readable(vp):
            continue

        try:
            start_sec = ts_to_seconds(row[start_col])
            end_sec   = ts_to_seconds(row[end_col])
        except:
            continue

        if end_sec <= start_sec:
            continue
        if (end_sec - start_sec) < 0.3:  # ignore too short
            continue

        verb = row["verb_class"] if "verb_class" in pool.columns else None
        noun = row["noun_class"] if "noun_class" in pool.columns else None

        try:
            x, frames, fps, n_frames, start_f, end_f = extract_frames_and_rgbmean(vp, start_sec, end_sec, T)
        except Exception as e:
            # skip problematic videos/segments
            continue

        # Save outputs
        tag = f"{video_id}_{int(start_sec*1000)}_{int(end_sec*1000)}_T{T}"
        npy_path = os.path.join(OUT_DIR, f"{tag}_rgbmean.npy")
        jpg_path = os.path.join(OUT_DIR, f"{tag}_montage.jpg")
        np.save(npy_path, x)
        save_montage(frames, jpg_path, cols=8)

        print("\n=== Stage-1 SUCCESS ===")
        print(f"video_id      : {video_id}")
        print(f"video_path    : {vp}")
        print(f"video_size_mb : {row['size_mb']:.1f} MB")
        print(f"{start_col}   : {row[start_col]} ({start_sec:.3f}s)")
        print(f"{end_col}     : {row[end_col]} ({end_sec:.3f}s)")
        print(f"duration      : {end_sec - start_sec:.3f}s")
        print(f"fps           : {fps:.3f}")
        print(f"frames_total  : {n_frames}")
        print(f"frame_range   : [{start_f}, {end_f})")
        print(f"verb_class    : {verb}")
        print(f"noun_class    : {noun}")
        print(f"x.shape       : {x.shape}  (expected [{T}, 3])")
        print(f"saved npy     : {npy_path}")
        print(f"saved montage : {jpg_path}")
        return

    print("[ERROR] Could not find a readable segment from local videos after scanning candidates.")
    print("Possible reasons: videos are partial, OpenCV codec issue, or CSV timestamps don't match.")
    print("Try lowering MIN_VIDEO_SIZE_MB or ensure at least one MP4 plays normally in your player.")

if __name__ == "__main__":
    main()
