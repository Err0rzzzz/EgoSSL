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

OUT_ROOT  = r"F:/EgoSSL/epic-tfc/stage2_minidataset"
OUT_NPY_DIR = os.path.join(OUT_ROOT, "dataset_npys")
OUT_INDEX_CSV = os.path.join(OUT_ROOT, "index.csv")
OUT_MONTAGE_DIR = os.path.join(OUT_ROOT, "montages")

# =========================
# Dataset config
# =========================
VIDEO_ID = "P01_05"        # start from one known-good video
T = 64                    # frames per segment
N_SAMPLES = 1000          # build a small dataset first (increase later)
SAVE_MONTAGE_TOPK = 20    # save montages for first K samples (set 0 to disable)

# Robustness / speed
SEED = 42
MIN_DURATION_SEC = 0.5     # skip ultra-short segments
MAX_DURATION_SEC = 8.0     # optionally cap long segments (keeps sampling stable)
MIN_VIDEO_SIZE_MB = 150    # skip tiny partial files

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

def extract_features_for_segment(cap: cv2.VideoCapture, fps: float, n_frames: int,
                                 start_sec: float, end_sec: float, T: int):
    """
    Returns:
      x: [T, 4] = [rgb_mean(3), motion(1)]
      frames_small: list of small BGR frames (for montage)
    """
    start_f = max(0, int(np.floor(start_sec * fps)))
    end_f   = min(n_frames, int(np.ceil(end_sec * fps)))
    idxs = uniform_sample_indices(start_f, end_f, T)

    rgb_feats = []
    motion_feats = []
    frames_small = []

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

        # montage frame (downsample)
        h, w = frame.shape[:2]
        scale = 256 / max(h, w)
        if scale < 1.0:
            frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_small = frame
        frames_small.append(frame_small)

        # RGB mean
        bgr_mean = frame.mean(axis=(0, 1))
        rgb_mean = bgr_mean[::-1]
        rgb_feats.append(rgb_mean.tolist())

        # Motion energy (frame diff in gray)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is None:
            motion = 0.0
        else:
            motion = float(np.mean(np.abs(gray - prev_gray)))
        motion_feats.append([motion])
        prev_gray = gray

    rgb = np.asarray(rgb_feats, dtype=np.float32)       # [T, 3]
    mot = np.asarray(motion_feats, dtype=np.float32)    # [T, 1]

    # Normalize motion per-segment (optional but usually helpful)
    # Keep it simple: z-score with epsilon
    eps = 1e-6
    mot = (mot - mot.mean()) / (mot.std() + eps)

    x = np.concatenate([rgb, mot], axis=1)              # [T, 4]
    return x, frames_small, (start_f, end_f)

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(OUT_NPY_DIR, exist_ok=True)
    if SAVE_MONTAGE_TOPK > 0:
        os.makedirs(OUT_MONTAGE_DIR, exist_ok=True)

    print(f"[INFO] VIDEO_ID    = {VIDEO_ID}")
    print(f"[INFO] DATA_ROOT   = {DATA_ROOT}")
    print(f"[INFO] TRAIN_CSV   = {TRAIN_CSV}")
    print(f"[INFO] OUT_ROOT    = {OUT_ROOT}")
    print(f"[INFO] N_SAMPLES   = {N_SAMPLES}, T={T}, D=4 (RGB mean + motion)")

    df = pd.read_csv(TRAIN_CSV)
    start_col, end_col = pick_start_end_cols(df)

    df = df[df["video_id"] == VIDEO_ID].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows found for video_id={VIDEO_ID} in train CSV.")

    vp = video_path_from_id(VIDEO_ID)
    if not os.path.exists(vp):
        raise RuntimeError(f"Video file not found: {vp}")

    size_mb = file_size_mb(vp)
    print(f"[INFO] video_path  = {vp}")
    print(f"[INFO] video_size  = {size_mb:.1f} MB")
    if size_mb < MIN_VIDEO_SIZE_MB:
        raise RuntimeError(f"Video too small (<{MIN_VIDEO_SIZE_MB}MB). Likely partial: {vp}")

    if not is_video_readable(vp):
        raise RuntimeError(f"Video not readable by OpenCV: {vp}")

    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 1e-6 or n_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: fps={fps}, n_frames={n_frames}")

    print(f"[INFO] fps={fps:.3f}, n_frames={n_frames}")

    # Prepare candidate segments with duration constraints
    def safe_sec(x):
        try:
            return ts_to_seconds(x)
        except:
            return np.nan

    df["start_sec"] = df[start_col].apply(safe_sec)
    df["end_sec"]   = df[end_col].apply(safe_sec)
    df = df.dropna(subset=["start_sec", "end_sec"]).copy()
    df["dur"] = df["end_sec"] - df["start_sec"]
    df = df[(df["dur"] >= MIN_DURATION_SEC) & (df["dur"] <= MAX_DURATION_SEC)].copy()

    if len(df) == 0:
        cap.release()
        raise RuntimeError("No segments remain after duration filtering. Relax MIN/MAX_DURATION_SEC.")

    # Sample segments
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df_sel = df.head(min(N_SAMPLES, len(df))).copy()

    records = []
    saved = 0

    for i, row in df_sel.iterrows():
        start_sec = float(row["start_sec"])
        end_sec   = float(row["end_sec"])

        verb = int(row["verb_class"]) if "verb_class" in row and not pd.isna(row["verb_class"]) else None
        noun = int(row["noun_class"]) if "noun_class" in row and not pd.isna(row["noun_class"]) else None

        try:
            x, frames_small, (start_f, end_f) = extract_features_for_segment(
                cap, fps, n_frames, start_sec, end_sec, T
            )
        except Exception:
            continue

        # Basic sanity: avoid all-zero / decode failure segments
        if np.allclose(x[:, :3].sum(), 0.0, atol=1e-3):
            continue

        tag = f"{VIDEO_ID}_{int(start_sec*1000)}_{int(end_sec*1000)}_T{T}_D4"
        npy_path = os.path.join(OUT_NPY_DIR, f"{tag}.npy")
        np.save(npy_path, x)

        montage_path = ""
        if SAVE_MONTAGE_TOPK > 0 and saved < SAVE_MONTAGE_TOPK:
            montage_path = os.path.join(OUT_MONTAGE_DIR, f"{tag}.jpg")
            save_montage(frames_small, montage_path, cols=8)

        records.append({
            "sample_id": tag,
            "video_id": VIDEO_ID,
            "video_path": vp,
            "start_timestamp": row[start_col],
            "stop_timestamp": row[end_col],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": end_sec - start_sec,
            "start_frame": start_f,
            "end_frame": end_f,
            "fps": fps,
            "verb_class": verb,
            "noun_class": noun,
            "feature_path": npy_path,
            "montage_path": montage_path,
        })
        saved += 1

        if saved % 50 == 0:
            print(f"[INFO] saved {saved}/{len(df_sel)} samples...")

        if saved >= N_SAMPLES:
            break

    cap.release()

    if saved == 0:
        raise RuntimeError("No samples saved. Possibly decode failures or overly strict filters.")

    idx = pd.DataFrame(records)
    idx.to_csv(OUT_INDEX_CSV, index=False)

    print("\n=== Stage-2 (data build) SUCCESS ===")
    print(f"Saved samples : {saved}")
    print(f"Index CSV     : {OUT_INDEX_CSV}")
    print(f"NPY dir       : {OUT_NPY_DIR}")
    if SAVE_MONTAGE_TOPK > 0:
        print(f"Montages dir  : {OUT_MONTAGE_DIR}")

    # Quick peek
    print("\n[INFO] index.csv head:")
    print(idx.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
