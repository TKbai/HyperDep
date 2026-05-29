# tools/compute_dvlog_stats.py
from pathlib import Path
import csv
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dvlog-dataset")
    parser.add_argument("--fold", default="train")
    parser.add_argument("--out", default="dvlog_stats.npz")
    args = parser.parse_args()

    root = Path(args.root)
    rows = list(csv.reader(open(root / "labels.csv", "r", encoding="utf-8")))

    total_sum = None
    total_sumsq = None
    total_count = 0

    for row in rows:
        if len(row) < 5:
            continue

        sid = row[0].strip()
        if sid.lower() in {"id", "sample_id", "video_id"}:
            continue

        fold = row[4].strip().lower()
        if fold != args.fold.lower():
            continue

        v_path = root / sid / f"{sid}_visual.npy"
        a_path = root / sid / f"{sid}_acoustic.npy"
        if not v_path.exists() or not a_path.exists():
            continue

        v = np.load(v_path).astype(np.float64)
        a = np.load(a_path).astype(np.float64)

        t = min(v.shape[0], a.shape[0])
        x = np.concatenate([v[:t], a[:t]], axis=1)

        if total_sum is None:
            total_sum = x.sum(axis=0)
            total_sumsq = (x ** 2).sum(axis=0)
        else:
            total_sum += x.sum(axis=0)
            total_sumsq += (x ** 2).sum(axis=0)

        total_count += x.shape[0]

    mean = total_sum / total_count
    var = total_sumsq / total_count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-6))

    np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print("saved:", args.out)
    print("feature dim:", mean.shape[0])
    print("count:", total_count)


if __name__ == "__main__":
    main()