import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="LMVD/processed/manifest.csv",
        help="LMVD processed manifest path",
    )

    parser.add_argument(
        "--out",
        default="LMVD/processed/lmvd_stats.npz",
        help="output npz stats path",
    )

    parser.add_argument(
        "--sample-frames-per-video",
        type=int,
        default=300,
        help="number of frames sampled per training video for percentile estimation",
    )

    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="lower percentile for robust clipping",
    )

    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.0,
        help="upper percentile for robust clipping",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=9,
    )

    return parser.parse_args()


def load_feature(path):
    arr = np.load(path).astype(np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected feature [T, D], got {arr.shape} from {path}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr


def main():
    args = parse_args()

    manifest = pd.read_csv(args.manifest)
    train_df = manifest[manifest["fold"] == "train"].reset_index(drop=True)

    if len(train_df) == 0:
        raise RuntimeError("No train samples found in manifest.")

    rng = np.random.RandomState(args.seed)

    print("=" * 80)
    print("manifest:", args.manifest)
    print("num train samples:", len(train_df))
    print("out:", args.out)
    print("=" * 80)

    # ------------------------------------------------------------
    # Pass 1: sample frames for robust clipping percentiles
    # ------------------------------------------------------------
    sampled_frames = []

    for idx, row in train_df.iterrows():
        feat = load_feature(row["feature_path"])
        t = feat.shape[0]

        if t <= 0:
            continue

        n = min(args.sample_frames_per_video, t)

        if t <= n:
            chosen = np.arange(t)
        else:
            chosen = rng.choice(t, size=n, replace=False)

        sampled_frames.append(feat[chosen])

        if (idx + 1) % 100 == 0 or idx + 1 == len(train_df):
            print(f"sample pass: {idx + 1}/{len(train_df)}")

    sampled = np.concatenate(sampled_frames, axis=0).astype(np.float32)

    print()
    print("sampled shape:", sampled.shape)
    print("raw sampled min/max/mean:", float(sampled.min()), float(sampled.max()), float(sampled.mean()))

    clip_low = np.percentile(sampled, args.lower_percentile, axis=0).astype(np.float32)
    clip_high = np.percentile(sampled, args.upper_percentile, axis=0).astype(np.float32)

    # avoid invalid intervals
    bad = clip_high <= clip_low
    clip_high[bad] = clip_low[bad] + 1.0

    print("clip_low shape:", clip_low.shape)
    print("clip_high shape:", clip_high.shape)

    # ------------------------------------------------------------
    # Pass 2: compute mean/std after clipping
    # ------------------------------------------------------------
    total_count = 0
    total_sum = None
    total_sq_sum = None

    for idx, row in train_df.iterrows():
        feat = load_feature(row["feature_path"])

        feat = np.clip(feat, clip_low, clip_high)

        if total_sum is None:
            dim = feat.shape[1]
            total_sum = np.zeros(dim, dtype=np.float64)
            total_sq_sum = np.zeros(dim, dtype=np.float64)

        total_sum += feat.sum(axis=0, dtype=np.float64)
        total_sq_sum += np.square(feat, dtype=np.float64).sum(axis=0)
        total_count += feat.shape[0]

        if (idx + 1) % 100 == 0 or idx + 1 == len(train_df):
            print(f"stat pass: {idx + 1}/{len(train_df)}")

    mean = total_sum / max(total_count, 1)
    var = total_sq_sum / max(total_count, 1) - mean ** 2
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    std[std < 1e-6] = 1.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        mean=mean,
        std=std,
        clip_low=clip_low,
        clip_high=clip_high,
        total_count=np.asarray([total_count], dtype=np.int64),
        lower_percentile=np.asarray([args.lower_percentile], dtype=np.float32),
        upper_percentile=np.asarray([args.upper_percentile], dtype=np.float32),
    )

    print()
    print("=" * 80)
    print("saved stats:", out_path)
    print("feature dim:", mean.shape[0])
    print("total train frames:", total_count)
    print("mean range:", float(mean.min()), float(mean.max()))
    print("std range :", float(std.min()), float(std.max()))
    print("clip_low range :", float(clip_low.min()), float(clip_low.max()))
    print("clip_high range:", float(clip_high.min()), float(clip_high.max()))
    print("LMVD stats computation finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()