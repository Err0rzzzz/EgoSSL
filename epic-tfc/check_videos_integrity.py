import os
import glob
import csv
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import cv2


@dataclass
class VideoCheckResult:
    path: str
    ok: bool
    suspicious: bool
    reason: str
    fps: Optional[float]
    frame_count: Optional[int]
    duration_sec: Optional[float]
    file_size_mb: float


def try_read_frame(cap: cv2.VideoCapture, frame_idx: int) -> bool:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx, 0))
    ret, frame = cap.read()
    return bool(ret) and frame is not None and frame.size > 0


def check_one_video(path: str, min_size_mb: float = 5.0) -> VideoCheckResult:
    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    if file_size_mb < min_size_mb:
        return VideoCheckResult(
            path=path, ok=False, suspicious=True,
            reason=f"File too small ({file_size_mb:.2f} MB) -> likely incomplete download",
            fps=None, frame_count=None, duration_sec=None, file_size_mb=file_size_mb
        )

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return VideoCheckResult(
            path=path, ok=False, suspicious=True,
            reason="OpenCV cannot open video (corrupted or incomplete)",
            fps=None, frame_count=None, duration_sec=None, file_size_mb=file_size_mb
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else None

    # Try first frame
    if not try_read_frame(cap, 0):
        cap.release()
        return VideoCheckResult(
            path=path, ok=False, suspicious=True,
            reason="Cannot read first frame (corrupted/incomplete)",
            fps=fps, frame_count=frame_count, duration_sec=None, file_size_mb=file_size_mb
        )

    # Try near-end frame
    suspicious = False
    reason = "OK"
    duration_sec = None

    if fps is not None and frame_count is not None and fps > 0 and frame_count > 0:
        duration_sec = frame_count / fps
        # Read frame near the end (frame_count - 5)
        if not try_read_frame(cap, frame_count - 5):
            suspicious = True
            reason = "Cannot read near-end frame -> likely truncated download"
    else:
        # If we can't get fps/frame_count, still try to seek to a large position
        if not try_read_frame(cap, 10_000_000):
            suspicious = True
            reason = "Cannot seek/read far frame; metadata missing; may be truncated"

    cap.release()
    ok = (reason == "OK") or (suspicious is False)

    return VideoCheckResult(
        path=path, ok=ok, suspicious=suspicious, reason=reason,
        fps=fps, frame_count=frame_count, duration_sec=duration_sec,
        file_size_mb=file_size_mb
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/EPIC-KITCHENS",
                    help="Root folder containing P01/P02 folders, e.g., data/EPIC-KITCHENS")
    ap.add_argument("--participants", type=str, default="P01,P02",
                    help="Comma-separated participants to check, e.g., P01,P02")
    ap.add_argument("--min_size_mb", type=float, default=5.0,
                    help="Mark very small MP4 as suspicious/incomplete")
    ap.add_argument("--out_csv", type=str, default="outputs/video_integrity_report.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    participants = [p.strip() for p in args.participants.split(",") if p.strip()]
    mp4_paths = []
    for p in participants:
        mp4_paths.extend(glob.glob(os.path.join(args.root, p, "videos", "*.MP4")))
        mp4_paths.extend(glob.glob(os.path.join(args.root, p, "videos", "*.mp4")))

    mp4_paths = sorted(set(mp4_paths))
    print(f"Found {len(mp4_paths)} video files under {args.root} for {participants}")

    results: List[VideoCheckResult] = []
    for i, path in enumerate(mp4_paths, 1):
        r = check_one_video(path, min_size_mb=args.min_size_mb)
        results.append(r)
        status = "OK" if (r.ok and not r.suspicious) else ("SUSPICIOUS" if r.suspicious else "BAD")
        print(f"[{i:04d}/{len(mp4_paths):04d}] {status}: {os.path.basename(path)} | {r.reason}")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "ok", "suspicious", "reason", "fps", "frame_count", "duration_sec", "file_size_mb"])
        for r in results:
            w.writerow([r.path, int(r.ok), int(r.suspicious), r.reason, r.fps, r.frame_count, r.duration_sec, f"{r.file_size_mb:.2f}"])

    print(f"\nSaved report: {args.out_csv}")
    print("Tip: Filter rows where suspicious==1 or ok==0 to see incomplete videos.")


if __name__ == "__main__":
    main()
