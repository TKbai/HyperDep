import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


META_COLS = {
    "frame",
    "face_id",
    "timestamp",
    "confidence",
    "success",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="LMVD",
        help="LMVD root directory",
    )

    parser.add_argument(
        "--out-dir",
        default="LMVD/processed",
        help="output directory for processed features and manifest",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=9,
        help="random seed for stratified split",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="debug only; process at most this many samples",
    )

    return parser.parse_args()


def find_dir(root, names):
    lower_names = {x.lower() for x in names}
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in lower_names:
            return p
    return None


def normalize_sid(stem):
    first = stem.split("_")[0]
    try:
        return f"{int(first):03d}"
    except Exception:
        return first


def sid_int_string(stem):
    first = stem.split("_")[0]
    try:
        return str(int(first))
    except Exception:
        return first


def read_label_from_csv(path):
    """
    LMVD label csv usually stores label in column header:
        columns: ['1'] or ['0']
    """
    df = pd.read_csv(path)
    cols = list(df.columns)

    if len(cols) > 0:
        raw = str(cols[0]).strip()
        if raw in ["0", "0.0"]:
            return 0
        if raw in ["1", "1.0"]:
            return 1

    if df.shape[0] > 0:
        raw = str(df.iloc[0, 0]).strip()
        if raw in ["0", "0.0"]:
            return 0
        if raw in ["1", "1.0"]:
            return 1

    raise ValueError(f"Cannot parse label from {path}")


def load_video_feature(path):
    """
    Read OpenFace-style video CSV.

    Important:
    Some columns may be read as object because of mixed types.
    We do not rely on pandas numeric type inference.
    Instead:
        1. strip column names
        2. drop meta columns
        3. force all remaining columns to numeric
        4. replace NaN/Inf with 0
    """

    df = pd.read_csv(path, low_memory=False)

    # Strip column names to avoid mismatches like " confidence".
    df.columns = [str(c).strip() for c in df.columns]

    keep_indices = []
    keep_cols = []

    for i, c in enumerate(df.columns):
        if c.strip().lower() in META_COLS:
            continue
        keep_indices.append(i)
        keep_cols.append(c)

    feat_df = df.iloc[:, keep_indices]
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    feat = feat_df.values.astype(np.float32)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    return feat, keep_cols


def load_audio_feature(path):
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr)

    if arr.dtype == object:
        arr = np.asarray(arr.tolist(), dtype=np.float32)
    else:
        arr = arr.astype(np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(f"Expected audio feature [T, D], got {arr.shape} from {path}")

    # Most LMVD audio is [T, 128].
    # If it is accidentally [128, T], transpose it.
    if arr.shape[0] == 128 and arr.shape[1] != 128:
        arr = arr.T

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr


def average_pool_video_to_audio_len(video, target_len):
    """
    Align high-frame-rate video features to audio feature length.

    video: [T_video, D]
    target_len: T_audio

    return: [T_audio, D]
    """

    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")

    t_video, dim = video.shape

    if t_video <= 0:
        return np.zeros((target_len, dim), dtype=np.float32)

    if t_video == target_len:
        return video.astype(np.float32)

    boundaries = np.linspace(0, t_video, target_len + 1)
    out = np.zeros((target_len, dim), dtype=np.float32)

    for i in range(target_len):
        s = int(np.floor(boundaries[i]))
        e = int(np.floor(boundaries[i + 1]))

        s = max(0, min(s, t_video - 1))
        e = max(s + 1, min(e, t_video))

        out[i] = video[s:e].mean(axis=0)

    return out


def build_stratified_split(records, seed=9, train_ratio=0.7, val_ratio=0.1):
    rng = np.random.RandomState(seed)

    by_label = {}
    for r in records:
        by_label.setdefault(int(r["label"]), []).append(r)

    split_map = {}

    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)

        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

        for r in train_items:
            split_map[r["sid"]] = "train"
        for r in val_items:
            split_map[r["sid"]] = "valid"
        for r in test_items:
            split_map[r["sid"]] = "test"

    return split_map


def main():
    args = parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    fused_dir = out_dir / "fused"

    out_dir.mkdir(parents=True, exist_ok=True)
    fused_dir.mkdir(parents=True, exist_ok=True)

    label_dir = find_dir(root, ["label"])
    video_dir = find_dir(root, ["video_feature", "Video_feature"])
    audio_dir = find_dir(root, ["audio_feature", "Audio_feature"])

    print("=" * 80)
    print("root      :", root.resolve())
    print("label_dir :", label_dir)
    print("video_dir :", video_dir)
    print("audio_dir :", audio_dir)
    print("out_dir   :", out_dir)
    print("=" * 80)

    if label_dir is None:
        raise RuntimeError("Cannot find label directory.")
    if video_dir is None:
        raise RuntimeError("Cannot find video feature directory.")
    if audio_dir is None:
        raise RuntimeError("Cannot find audio feature directory.")

    label_files = sorted(label_dir.rglob("*.csv")) + sorted(label_dir.rglob("*.CSV"))
    video_files = sorted(video_dir.rglob("*.csv")) + sorted(video_dir.rglob("*.CSV"))
    audio_files = sorted(audio_dir.rglob("*.npy")) + sorted(audio_dir.rglob("*.npz"))

    video_map = {}
    for p in video_files:
        sid = normalize_sid(p.stem)
        video_map[sid] = p

    audio_map = {}
    for p in audio_files:
        sid = normalize_sid(p.stem)
        audio_map[sid] = p

    records = []

    for label_path in label_files:
        sid = normalize_sid(label_path.stem)
        label = read_label_from_csv(label_path)

        video_path = video_map.get(sid)
        audio_path = audio_map.get(sid)

        if video_path is None or audio_path is None:
            continue

        records.append(
            {
                "sid": sid,
                "label": int(label),
                "label_path": str(label_path),
                "video_path": str(video_path),
                "audio_path": str(audio_path),
            }
        )

    records = sorted(records, key=lambda x: int(x["sid"]))

    if args.max_samples is not None:
        records = records[:args.max_samples]

    print("matched records:", len(records))
    print("label counter:", Counter([r["label"] for r in records]))

    split_map = build_stratified_split(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    manifest_rows = []

    video_dim_ref = None
    audio_dim_ref = None
    fused_dim_ref = None

    video_length_list = []
    audio_length_list = []
    fused_length_list = []

    for idx, r in enumerate(records):
        sid = r["sid"]

        video, video_cols = load_video_feature(r["video_path"])
        audio = load_audio_feature(r["audio_path"])

        if audio.shape[0] <= 0:
            print("skip empty audio:", sid)
            continue

        aligned_video = average_pool_video_to_audio_len(
            video,
            target_len=audio.shape[0],
        )

        if aligned_video.shape[0] != audio.shape[0]:
            raise RuntimeError(
                f"alignment failed for sid={sid}: "
                f"video {aligned_video.shape}, audio {audio.shape}"
            )

        fused = np.concatenate([aligned_video, audio], axis=1).astype(np.float32)

        video_dim = aligned_video.shape[1]
        audio_dim = audio.shape[1]
        fused_dim = fused.shape[1]

        if video_dim_ref is None:
            video_dim_ref = video_dim
            audio_dim_ref = audio_dim
            fused_dim_ref = fused_dim
            print()
            print("Detected dims:")
            print("video_dim:", video_dim_ref)
            print("audio_dim:", audio_dim_ref)
            print("fused_dim:", fused_dim_ref)
            print("first 20 video cols:", video_cols[:20])
            print("last 20 video cols :", video_cols[-20:])

        if video_dim != video_dim_ref:
            raise RuntimeError(f"video_dim mismatch for sid={sid}: {video_dim} vs {video_dim_ref}")

        if audio_dim != audio_dim_ref:
            raise RuntimeError(f"audio_dim mismatch for sid={sid}: {audio_dim} vs {audio_dim_ref}")

        out_path = fused_dir / f"{sid}.npy"
        np.save(out_path, fused)

        video_length_list.append(video.shape[0])
        audio_length_list.append(audio.shape[0])
        fused_length_list.append(fused.shape[0])

        manifest_rows.append(
            {
                "sid": sid,
                "label": r["label"],
                "fold": split_map[sid],
                "feature_path": str(out_path),
                "video_path": r["video_path"],
                "audio_path": r["audio_path"],
                "video_len_raw": int(video.shape[0]),
                "audio_len": int(audio.shape[0]),
                "seq_len": int(fused.shape[0]),
                "video_dim": int(video_dim),
                "audio_dim": int(audio_dim),
                "fused_dim": int(fused_dim),
            }
        )

        if (idx + 1) % 100 == 0 or idx + 1 == len(records):
            print(f"processed {idx + 1}/{len(records)}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    dims = {
        "video_dim": int(video_dim_ref),
        "audio_dim": int(audio_dim_ref),
        "fused_dim": int(fused_dim_ref),
        "num_samples": int(len(manifest_rows)),
        "label_counter": {str(k): int(v) for k, v in Counter(manifest["label"]).items()},
        "fold_counter": {str(k): int(v) for k, v in Counter(manifest["fold"]).items()},
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "video_length_raw": {
            "min": int(np.min(video_length_list)),
            "median": float(np.median(video_length_list)),
            "max": int(np.max(video_length_list)),
        },
        "audio_length": {
            "min": int(np.min(audio_length_list)),
            "median": float(np.median(audio_length_list)),
            "max": int(np.max(audio_length_list)),
        },
        "fused_length": {
            "min": int(np.min(fused_length_list)),
            "median": float(np.median(fused_length_list)),
            "max": int(np.max(fused_length_list)),
        },
    }

    dims_path = out_dir / "dims.json"
    with open(dims_path, "w", encoding="utf-8") as f:
        json.dump(dims, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print("Saved manifest:", manifest_path)
    print("Saved dims    :", dims_path)
    print("=" * 80)

    print()
    print("manifest shape:", manifest.shape)
    print("label counter:")
    print(Counter(manifest["label"]))
    print("fold counter:")
    print(Counter(manifest["fold"]))

    print()
    print("dims:")
    print(json.dumps(dims, indent=2, ensure_ascii=False))

    print()
    print("LMVD feature preparation finished.")


if __name__ == "__main__":
    main()