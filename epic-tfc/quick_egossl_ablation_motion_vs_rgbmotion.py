# Author: Zhengyu He
# Purpose: Minimal ablation for "to what extent" claim:
#          compare motion-only (1D) vs RGB+motion (4D) sensor-like time-series features.

import os
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# ----------------------------
# Utils: annotation loading
# ----------------------------
REQ_SETS = {
    "video": {"video_id", "video"},
    "start": {"start_timestamp", "start_time", "start"},
    "end": {"stop_timestamp", "end_timestamp", "stop_time", "end_time", "end"},
    "verb": {"verb", "verb_class", "verb_id"},
}

EXCLUDE_KEYWORDS = [
    "retrieval",
    "narration",
    "narrations",
    "missing_timestamp",
    "missing",
]

_CAP_CACHE: Dict[str, cv2.VideoCapture] = {}


def normalize_cols(cols: List[str]) -> set:
    return {c.strip().lower() for c in cols}


def has_required_columns(df: pd.DataFrame) -> Tuple[bool, Dict[str, str]]:
    cols = normalize_cols(list(df.columns))

    def pick(name_set: set) -> Optional[str]:
        for n in name_set:
            if n in cols:
                for c in df.columns:
                    if c.strip().lower() == n:
                        return c
        return None

    video_col = pick(REQ_SETS["video"])
    start_col = pick(REQ_SETS["start"])
    end_col = pick(REQ_SETS["end"])
    verb_col = pick(REQ_SETS["verb"])

    ok = all([video_col, start_col, end_col, verb_col])
    picked = {"video": video_col, "start": start_col, "end": end_col, "verb": verb_col}
    return ok, picked


def parse_timestamp_to_seconds(ts: str) -> Optional[float]:
    if ts is None:
        return None
    if not isinstance(ts, str):
        ts = str(ts)
    ts = ts.strip()
    if not ts:
        return None
    m = re.match(r"(\d+):(\d+):(\d+)(\.\d+)?", ts)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3))
    frac = float(m.group(4)) if m.group(4) else 0.0
    return hh * 3600 + mm * 60 + ss + frac


def load_actions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ok, picked = has_required_columns(df)
    if not ok:
        raise ValueError(
            f"CSV is not an action segment annotation:\n{path}\n"
            f"Columns: {list(df.columns)}\n"
            f"Need video_id/start_timestamp/stop_timestamp/verb (or compatible names)."
        )

    df["_video_id"] = df[picked["video"]].astype(str)
    df["_start_ts"] = df[picked["start"]].astype(str)
    df["_end_ts"] = df[picked["end"]].astype(str)
    df["_verb"] = df[picked["verb"]]
    df["_start_sec"] = df["_start_ts"].apply(parse_timestamp_to_seconds)
    df["_end_sec"] = df["_end_ts"].apply(parse_timestamp_to_seconds)

    df = df[df["_start_sec"].notna() & df["_end_sec"].notna()].copy()
    df = df[df["_end_sec"] > df["_start_sec"]].copy()
    return df


# ----------------------------
# Video I/O & feature extraction
# ----------------------------
def get_cap(video_path: str) -> Optional[cv2.VideoCapture]:
    cap = _CAP_CACHE.get(video_path, None)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        _CAP_CACHE[video_path] = cap
    if not cap.isOpened():
        return None
    return cap


def extract_timeseries_features(
    video_path: str,
    start_sec: float,
    end_sec: float,
    feature_mode: str,     # "motion_only" or "rgb_motion"
    sample_hz: float = 4.0,
    max_len: int = 32,
) -> Optional[np.ndarray]:
    cap = get_cap(video_path)
    if cap is None:
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        return None

    seg_len = end_sec - start_sec
    if seg_len <= 0.2:
        return None

    T = int(min(max_len, max(8, round(seg_len * sample_hz))))
    times = np.linspace(start_sec, end_sec, T, endpoint=False)

    prev_gray = None
    feats = []

    for tsec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec * 1000.0))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = 0.0 if prev_gray is None else float(np.mean(np.abs(gray - prev_gray)))
        prev_gray = gray

        if feature_mode == "motion_only":
            feats.append([motion])
        elif feature_mode == "rgb_motion":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mean_rgb = frame_rgb.reshape(-1, 3).mean(axis=0)
            feats.append([mean_rgb[0], mean_rgb[1], mean_rgb[2], motion])
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

    X = np.asarray(feats, dtype=np.float32)

    # normalize
    if feature_mode == "motion_only":
        X[:, 0] = X[:, 0] / (X[:, 0].max() + 1e-6)
    else:
        X[:, :3] = X[:, :3] / 255.0
        X[:, 3] = X[:, 3] / (X[:, 3].max() + 1e-6)

    return X


def segment_embedding_mean_std(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return np.concatenate([mu, sd], axis=0)


def majority_baseline_macro_f1(y: np.ndarray) -> float:
    vals, counts = np.unique(y, return_counts=True)
    maj = vals[np.argmax(counts)]
    y_pred = np.full_like(y, maj)
    return f1_score(y, y_pred, average="macro")


def build_video_map_from_integrity(data_root: str, integrity_csv: str, participants: List[str]) -> Dict[str, str]:
    integ = pd.read_csv(integrity_csv)
    ok_paths = set(integ[(integ["ok"] == 1) & (integ["suspicious"] == 0)]["path"].astype(str).tolist())

    vid_map: Dict[str, str] = {}
    for p in participants:
        mp4s = glob.glob(os.path.join(data_root, p, "videos", "*.MP4")) + \
               glob.glob(os.path.join(data_root, p, "videos", "*.mp4"))
        for mp4 in mp4s:
            if mp4 not in ok_paths:
                continue
            base = os.path.splitext(os.path.basename(mp4))[0]  # e.g. P01_01
            vid_map[base] = mp4
    return vid_map


def save_run_summary_row(summary_csv: Path, row: Dict):
    df_row = pd.DataFrame([row])
    if summary_csv.exists():
        df_old = pd.read_csv(summary_csv)
        df_all = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_all = df_row
    df_all.to_csv(summary_csv, index=False)


# ----------------------------
# Main: run one feature mode
# ----------------------------
def run_one(
    df_actions: pd.DataFrame,
    vid_map: Dict[str, str],
    feature_mode: str,
    max_segments: int,
    sample_hz: float,
    max_len: int,
    seed: int,
    out_dir: Path,
    top_k_verbs: int = 8,
) -> Dict:
    rng = np.random.RandomState(seed)

    df = df_actions[df_actions["_video_id"].isin(list(vid_map.keys()))].copy()
    if len(df) == 0:
        raise RuntimeError("No segments match available videos. Check video_id naming vs MP4 basenames.")

    if len(df) > max_segments:
        df = df.sample(max_segments, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    X_list: List[np.ndarray] = []
    y_list: List = []
    seg_meta: List[Tuple[str, str, str]] = []
    failed = 0

    for i, row in df.iterrows():
        video_id = row["_video_id"]
        vp = vid_map.get(video_id, None)
        if vp is None:
            continue

        X = extract_timeseries_features(
            video_path=vp,
            start_sec=float(row["_start_sec"]),
            end_sec=float(row["_end_sec"]),
            feature_mode=feature_mode,
            sample_hz=sample_hz,
            max_len=max_len,
        )
        if X is None:
            failed += 1
            continue

        X_list.append(X)
        y_list.append(row["_verb"])
        seg_meta.append((video_id, row["_start_ts"], row["_end_ts"]))

        if (i + 1) % 50 == 0:
            print(f"[{feature_mode}] Processed {i+1}/{len(df)} ... ok={len(X_list)}, failed={failed}")

    if len(X_list) < 30:
        raise RuntimeError(f"Too few segments extracted for {feature_mode}: {len(X_list)}")

    # Save segment list for reproducibility
    pd.DataFrame(seg_meta, columns=["video_id", "start_ts", "end_ts"]).to_csv(out_dir / "segments_filtered.csv", index=False)

    # Embed + quick eval
    E = np.stack([segment_embedding_mean_std(x) for x in X_list], axis=0)
    y = np.asarray(y_list)

    maj_f1 = majority_baseline_macro_f1(y)

    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(E[tr], y[tr])
    pred = clf.predict(E[te])
    macro_f1 = f1_score(y[te], pred, average="macro")

    # --------- Plots ----------
    # Random segments time-series
    n_show = min(12, len(X_list))
    pick = rng.choice(len(X_list), size=n_show, replace=False)

    plt.figure(figsize=(10, 8))
    for j, pi in enumerate(pick, 1):
        X = X_list[pi]
        t = np.arange(len(X))
        plt.subplot(4, 3, j)

        if feature_mode == "motion_only":
            plt.plot(t, X[:, 0], label="motion")
        else:
            plt.plot(t, X[:, 0], label="R")
            plt.plot(t, X[:, 1], label="G")
            plt.plot(t, X[:, 2], label="B")
            plt.plot(t, X[:, 3], label="motion")

        plt.title(f"verb={y_list[pi]}")
        plt.xticks([])
        if j == 1:
            plt.legend(fontsize=7)
    plt.suptitle(f"Random segments ({feature_mode})", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "viz_random_segments.png", dpi=200)
    plt.close()

    # Top verbs motion mean profile (only if motion exists)
    y_series = pd.Series(y_list)
    top_verbs = y_series.value_counts().head(top_k_verbs).index.tolist()

    def resample_to_len(X: np.ndarray, L: int) -> np.ndarray:
        src_t = np.linspace(0, 1, len(X))
        dst_t = np.linspace(0, 1, L)
        return np.stack([np.interp(dst_t, src_t, X[:, d]) for d in range(X.shape[1])], axis=1)

    plt.figure(figsize=(10, 6))
    for v in top_verbs:
        idx_v = [k for k, yy in enumerate(y_list) if yy == v]
        curves = [resample_to_len(X_list[k], max_len) for k in idx_v]
        mean_curve = np.mean(np.stack(curves, axis=0), axis=0)

        if feature_mode == "motion_only":
            plt.plot(mean_curve[:, 0], label=f"verb={v}")
        else:
            plt.plot(mean_curve[:, 3], label=f"verb={v}")
    plt.title(f"Top verbs: mean motion profile ({feature_mode})")
    plt.xlabel("Normalized time")
    plt.ylabel("motion proxy")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "viz_top_verbs_motion_mean.png", dpi=200)
    plt.close()

    # PCA of embeddings (color top verbs vs other)
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(E)

    plt.figure(figsize=(8, 6))
    color_tag = np.array([str(yy) if yy in top_verbs else "other" for yy in y_list])
    for tag in np.unique(color_tag):
        m = (color_tag == tag)
        plt.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, label=tag)
    plt.title(f"Embedding PCA ({feature_mode})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "viz_embedding_pca.png", dpi=200)
    plt.close()

    # Metrics txt
    with open(out_dir / "quick_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"feature_mode: {feature_mode}\n")
        f.write(f"num_segments_extracted: {len(X_list)}\n")
        f.write(f"failed_segments: {failed}\n")
        f.write(f"sample_hz: {sample_hz}\n")
        f.write(f"max_len: {max_len}\n")
        f.write(f"majority_macro_f1: {maj_f1:.6f}\n")
        f.write(f"linear_probe_macro_f1: {macro_f1:.6f}\n")

    return {
        "feature_mode": feature_mode,
        "extracted": len(X_list),
        "failed": failed,
        "maj_macro_f1": float(maj_f1),
        "linear_macro_f1": float(macro_f1),
    }


def release_caps():
    for cap in _CAP_CACHE.values():
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
    _CAP_CACHE.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/EPIC-KITCHENS")
    ap.add_argument("--integrity_csv", type=str, default="outputs/video_integrity_report.csv")
    ap.add_argument("--actions_csv", type=str, required=True)

    ap.add_argument("--participants", type=str, default="P01,P02")
    ap.add_argument("--max_segments", type=int, default=800)
    ap.add_argument("--sample_hz", type=float, default=4.0)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_root", type=str, default="outputs")
    ap.add_argument("--run_name", type=str, default=None,
                    help="Output folder name. If not set, auto-generated from args.")
    ap.add_argument("--top_k_verbs", type=int, default=8)

    args = ap.parse_args()

    # Optional: reduce decode failures by increasing ffmpeg read attempts.
    # You can set in terminal before running:
    #   set OPENCV_FFMPEG_READ_ATTEMPTS=100000
    # This script does not enforce it, but supports it.

    participants = [p.strip() for p in args.participants.split(",") if p.strip()]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.run_name is None:
        args.run_name = f"ablation_seg{args.max_segments}_hz{int(args.sample_hz)}_len{args.max_len}_seed{args.seed}"
    run_dir = out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build video map
    if not os.path.exists(args.integrity_csv):
        raise FileNotFoundError(
            f"Integrity report not found: {args.integrity_csv}\n"
            f"Please run your integrity checker first."
        )
    vid_map = build_video_map_from_integrity(args.data_root, args.integrity_csv, participants)
    if not vid_map:
        raise RuntimeError("No OK videos found after integrity filtering. Check data_root/integrity_csv.")

    # Load actions
    print(f"Using action annotations: {args.actions_csv}")
    df_actions = load_actions_csv(args.actions_csv)
    print(f"Segments total in CSV (after timestamp cleaning): {len(df_actions)}")

    # Two runs inside one run folder, so they share same sampled segments distribution size,
    # but sampling is independent per mode (same seed).
    modes = ["motion_only", "rgb_motion"]

    results = []
    for mode in modes:
        mode_dir = run_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== Running feature_mode={mode} =====")
        res = run_one(
            df_actions=df_actions,
            vid_map=vid_map,
            feature_mode=mode,
            max_segments=args.max_segments,
            sample_hz=args.sample_hz,
            max_len=args.max_len,
            seed=args.seed,
            out_dir=mode_dir,
            top_k_verbs=args.top_k_verbs,
        )
        results.append(res)

    # Write a small comparison text
    comp_path = run_dir / "comparison.txt"
    with open(comp_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                f"{r['feature_mode']}: extracted={r['extracted']} failed={r['failed']} "
                f"maj_macro_f1={r['maj_macro_f1']:.6f} linear_macro_f1={r['linear_macro_f1']:.6f}\n"
            )

    # Append to global summary CSV (one row per mode)
    summary_csv = out_root / "ablation_summary.csv"
    for r in results:
        row = {
            "run_name": args.run_name,
            "feature_mode": r["feature_mode"],
            "participants": args.participants,
            "max_segments": args.max_segments,
            "sample_hz": args.sample_hz,
            "max_len": args.max_len,
            "seed": args.seed,
            "extracted": r["extracted"],
            "failed": r["failed"],
            "majority_macro_f1": r["maj_macro_f1"],
            "linear_macro_f1": r["linear_macro_f1"],
        }
        save_run_summary_row(summary_csv, row)

    release_caps()

    print(f"\nSaved ablation outputs to: {run_dir}")
    print(f"- {comp_path}")
    print(f"- {summary_csv}")
    print("Inside run folder you have two subfolders:")
    print("  motion_only/ and rgb_motion/ each containing PNGs + quick_metrics.txt")


if __name__ == "__main__":
    main()
