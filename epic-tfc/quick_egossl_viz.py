# Author: Zhengyu He
# Project: EgoSSL (pilot visualization for egocentric sensor-like abstraction)

import os
import re
import glob
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# ----------------------------
# Annotation scanning & selection
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
    "object",              # some object-only files
    "hand",                # to avoid wrong benchmark CSVs if they lack timestamps
]


def normalize_cols(cols: List[str]) -> set:
    return {c.strip().lower() for c in cols}


def has_required_columns(df: pd.DataFrame) -> Tuple[bool, Dict[str, str]]:
    """
    Check whether df contains required columns: video_id, start timestamp, end timestamp, verb.
    Return (ok, picked_cols mapping).
    """
    cols = normalize_cols(list(df.columns))

    def pick(name_set: set) -> Optional[str]:
        for n in name_set:
            if n in cols:
                # return original column name (case preserved)
                for c in df.columns:
                    if c.strip().lower() == n:
                        return c
        return None

    video_col = pick(REQ_SETS["video"])
    start_col = pick(REQ_SETS["start"])
    end_col = pick(REQ_SETS["end"])
    verb_col = pick(REQ_SETS["verb"])

    ok = all([video_col, start_col, end_col, verb_col])
    picked = {
        "video": video_col,
        "start": start_col,
        "end": end_col,
        "verb": verb_col,
    }
    return ok, picked


def find_action_annotation_csv_strict(ann_root: str, max_probe: int = 80) -> str:
    """
    Strictly find an action annotation CSV:
      - Must contain video_id + start/end timestamps + verb columns
      - Exclude retrieval/narrations/missing_timestamp files
    If none found, print candidates and raise with actionable info.
    """
    csvs = glob.glob(os.path.join(ann_root, "**", "*.csv"), recursive=True)
    csvs = [p for p in csvs if os.path.isfile(p)]

    # exclude by filename keyword
    filtered = []
    for p in csvs:
        low = p.lower()
        if any(k in low for k in EXCLUDE_KEYWORDS):
            continue
        filtered.append(p)

    # probe candidates (limit for speed)
    probe_list = filtered[:max_probe]

    scored = []
    inspected = []
    for p in probe_list:
        try:
            df_head = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        ok, picked = has_required_columns(df_head)
        inspected.append((p, ok, list(df_head.columns)))
        if ok:
            # score by filename hints
            name = os.path.basename(p).lower()
            s = 0
            if "action" in name: s += 4
            if "segment" in name or "segments" in name: s += 4
            if "train" in name: s += 2
            if "validation" in name or "val" in name: s += 1
            if "test" in name: s += 1
            if "epic" in name: s += 2
            if "100" in name or "55" in name: s += 1
            scored.append((s, p))

    if scored:
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1]

    # If no strict match, print debug list
    print("\n[ERROR] No valid action-segment CSV found with required columns.")
    print("Required: video_id (or video), start_timestamp, stop_timestamp (or end_timestamp), verb (or verb_class)")
    print("\nScanned candidates (first 30 shown):")
    shown = 0
    for p, ok, cols in inspected[:30]:
        shown += 1
        print(f"  - {p}")
        print(f"    columns: {cols}")
    raise FileNotFoundError(
        "Could not automatically select a valid action annotation CSV.\n"
        "Please pass one explicitly via --actions_csv <path_to_csv>.\n"
        "Tip: choose a CSV that has video_id/start_timestamp/stop_timestamp/verb columns."
    )


def load_actions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ok, picked = has_required_columns(df)
    if not ok:
        raise ValueError(
            f"Selected CSV is not an action segment annotation:\n{path}\n"
            f"Columns: {list(df.columns)}\n"
            f"Need video_id/start_timestamp/stop_timestamp/verb (or compatible names)."
        )

    video_col = picked["video"]
    start_ts_col = picked["start"]
    end_ts_col = picked["end"]
    verb_col = picked["verb"]

    # noun is optional
    noun_col = None
    for n in ["noun", "noun_class", "noun_id"]:
        for c in df.columns:
            if c.strip().lower() == n:
                noun_col = c
                break
        if noun_col:
            break

    df["_video_id"] = df[video_col].astype(str)
    df["_start_ts"] = df[start_ts_col].astype(str)
    df["_end_ts"] = df[end_ts_col].astype(str)
    df["_verb"] = df[verb_col]
    df["_noun"] = df[noun_col] if noun_col is not None else -1

    return df


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


# ----------------------------
# Fast feature extraction (cached VideoCapture)
# ----------------------------
_CAP_CACHE: Dict[str, cv2.VideoCapture] = {}


def get_cap(video_path: str) -> Optional[cv2.VideoCapture]:
    cap = _CAP_CACHE.get(video_path, None)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        _CAP_CACHE[video_path] = cap
    if not cap.isOpened():
        return None
    return cap


def extract_timeseries_features_fast(
    video_path: str,
    start_sec: float,
    end_sec: float,
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

    feats = []
    prev_gray = None

    for tsec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec * 1000.0))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mean_rgb = frame_rgb.reshape(-1, 3).mean(axis=0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = 0.0 if prev_gray is None else float(np.mean(np.abs(gray - prev_gray)))
        prev_gray = gray

        feats.append([mean_rgb[0], mean_rgb[1], mean_rgb[2], motion])

    X = np.asarray(feats, dtype=np.float32)
    X[:, :3] = X[:, :3] / 255.0
    X[:, 3] = X[:, 3] / (X[:, 3].max() + 1e-6)
    return X


def segment_embedding(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return np.concatenate([mu, sd], axis=0)


def majority_baseline_macro_f1(y: np.ndarray) -> float:
    vals, counts = np.unique(y, return_counts=True)
    maj = vals[np.argmax(counts)]
    y_pred = np.full_like(y, maj)
    return f1_score(y, y_pred, average="macro")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/EPIC-KITCHENS")
    ap.add_argument("--ann_root", type=str, default="annotations")
    ap.add_argument("--actions_csv", type=str, default="",
                    help="Explicit path to action segments CSV (recommended if auto-selection fails)")
    ap.add_argument("--integrity_csv", type=str, default="outputs/video_integrity_report.csv")
    ap.add_argument("--participants", type=str, default="P01,P02")
    ap.add_argument("--max_segments", type=int, default=200)
    ap.add_argument("--sample_hz", type=float, default=4.0)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--top_k_verbs", type=int, default=8)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.integrity_csv):
        raise FileNotFoundError(
            f"Integrity report not found: {args.integrity_csv}\n"
            f"Run: python check_videos_integrity.py"
        )

    integ = pd.read_csv(args.integrity_csv)
    ok_paths = set(integ[(integ["ok"] == 1) & (integ["suspicious"] == 0)]["path"].astype(str).tolist())

    participants = [p.strip() for p in args.participants.split(",") if p.strip()]
    vid_map: Dict[str, str] = {}
    for p in participants:
        mp4s = glob.glob(os.path.join(args.data_root, p, "videos", "*.MP4")) + \
               glob.glob(os.path.join(args.data_root, p, "videos", "*.mp4"))
        for mp4 in mp4s:
            if mp4 not in ok_paths:
                continue
            base = os.path.splitext(os.path.basename(mp4))[0]  # e.g., P01_01
            vid_map[base] = mp4

    if not vid_map:
        raise RuntimeError("No OK videos found after integrity filtering. Check your data_root and integrity report.")

    # Choose action annotation CSV
    if args.actions_csv and os.path.exists(args.actions_csv):
        actions_csv = args.actions_csv
    else:
        actions_csv = find_action_annotation_csv_strict(args.ann_root)

    print(f"Using action annotations: {actions_csv}")
    df = load_actions_csv(actions_csv)

    # Filter to available videos (P01/P02 only)
    df = df[df["_video_id"].isin(list(vid_map.keys()))].copy()
    print(f"Segments after filtering by available videos: {len(df)}")

    df["_start_sec"] = df["_start_ts"].apply(parse_timestamp_to_seconds)
    df["_end_sec"] = df["_end_ts"].apply(parse_timestamp_to_seconds)
    df = df[df["_start_sec"].notna() & df["_end_sec"].notna()].copy()
    df = df[df["_end_sec"] > df["_start_sec"]].copy()

    if len(df) == 0:
        raise RuntimeError(
            "No valid segments after timestamp parsing.\n"
            "Try passing --actions_csv explicitly to a known action segments CSV."
        )

    # sample for speed
    if len(df) > args.max_segments:
        df = df.sample(args.max_segments, random_state=0).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Extract
    X_list: List[np.ndarray] = []
    y_list: List = []
    seg_meta: List[Tuple[str, str, str]] = []
    failed = 0

    for i, row in df.iterrows():
        vid = row["_video_id"]
        video_path = vid_map.get(vid, None)
        if video_path is None:
            continue

        X = extract_timeseries_features_fast(
            video_path=video_path,
            start_sec=float(row["_start_sec"]),
            end_sec=float(row["_end_sec"]),
            sample_hz=args.sample_hz,
            max_len=args.max_len
        )
        if X is None:
            failed += 1
            continue

        X_list.append(X)
        y_list.append(row["_verb"])
        seg_meta.append((vid, row["_start_ts"], row["_end_ts"]))

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)} segments... ok={len(X_list)}, failed={failed}")

    print(f"Extracted {len(X_list)} segments, failed {failed}.")
    if len(X_list) < 30:
        raise RuntimeError(
            "Too few segments extracted.\n"
            "Likely annotation mismatch (video_id not matching MP4 basenames) or decode issues.\n"
            "Next: pass --actions_csv explicitly to a correct action segments file."
        )

    # Save segments
    pd.DataFrame(seg_meta, columns=["video_id", "start_ts", "end_ts"]).to_csv(
        os.path.join(args.out_dir, "segments_filtered.csv"), index=False
    )

    # Embedding + quick eval
    E = np.stack([segment_embedding(x) for x in X_list], axis=0)
    y = np.asarray(y_list)

    maj_f1 = majority_baseline_macro_f1(y)

    rng = np.random.RandomState(0)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(E[tr], y[tr])
    pred = clf.predict(E[te])
    macro_f1 = f1_score(y[te], pred, average="macro")

    print(f"Majority baseline (macro-F1): {maj_f1:.4f}")
    print(f"Linear probe (macro-F1):      {macro_f1:.4f}")

    # -------- Plots ----------
    # 1) Random segments
    n_show = min(12, len(X_list))
    pick = rng.choice(len(X_list), size=n_show, replace=False)
    plt.figure(figsize=(10, 8))
    for i, pi in enumerate(pick, 1):
        X = X_list[pi]
        t = np.arange(len(X))
        plt.subplot(4, 3, i)
        plt.plot(t, X[:, 0], label="R")
        plt.plot(t, X[:, 1], label="G")
        plt.plot(t, X[:, 2], label="B")
        plt.plot(t, X[:, 3], label="motion")
        plt.title(f"verb={y_list[pi]}")
        plt.xticks([])
        if i == 1:
            plt.legend(fontsize=7)
    plt.suptitle("Random action segments: sensor-like time-series features", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "viz_random_segments.png"), dpi=200)
    plt.close()

    # 2) Top verbs mean motion curve
    y_series = pd.Series(y_list)
    top_verbs = y_series.value_counts().head(args.top_k_verbs).index.tolist()

    def resample_to_len(X: np.ndarray, L: int) -> np.ndarray:
        src_t = np.linspace(0, 1, len(X))
        dst_t = np.linspace(0, 1, L)
        return np.stack([np.interp(dst_t, src_t, X[:, d]) for d in range(X.shape[1])], axis=1)

    plt.figure(figsize=(10, 6))
    for v in top_verbs:
        idx_v = [i for i, yy in enumerate(y_list) if yy == v]
        curves = [resample_to_len(X_list[i], args.max_len) for i in idx_v]
        mean_curve = np.mean(np.stack(curves, axis=0), axis=0)
        plt.plot(mean_curve[:, 3], label=f"verb={v} (motion)")

    plt.title("Top verbs: mean motion profile (normalized)")
    plt.xlabel("Normalized time")
    plt.ylabel("motion proxy")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "viz_top_verbs_motion_mean.png"), dpi=200)
    plt.close()

    # 3) PCA scatter
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(E)

    plt.figure(figsize=(8, 6))
    color_tag = np.array([str(yy) if yy in top_verbs else "other" for yy in y_list])
    for tag in np.unique(color_tag):
        m = (color_tag == tag)
        plt.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, label=tag)

    plt.title("Segment embedding (mean+std) PCA visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "viz_embedding_pca.png"), dpi=200)
    plt.close()

    # metrics
    with open(os.path.join(args.out_dir, "quick_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Num segments extracted: {len(X_list)}\n")
        f.write(f"Participants: {args.participants}\n")
        f.write(f"sample_hz: {args.sample_hz}, max_len: {args.max_len}\n")
        f.write(f"Majority baseline macro-F1: {maj_f1:.6f}\n")
        f.write(f"Linear probe macro-F1: {macro_f1:.6f}\n")
        f.write(f"Failed segments: {failed}\n")
        f.write(f"Action annotations used: {actions_csv}\n")

    print(f"\nSaved figures to: {args.out_dir}/")
    print(" - viz_random_segments.png")
    print(" - viz_top_verbs_motion_mean.png")
    print(" - viz_embedding_pca.png")
    print(" - quick_metrics.txt")
    print(" - segments_filtered.csv")

    # release caps
    for cap in _CAP_CACHE.values():
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
