import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class LMVDDataset(Dataset):
    """
    LMVD processed feature dataset.

    Each sample:
        feature: [T, 593]
            first 465 dims are visual features
            next 128 dims are audio features

        label: scalar 0/1
    """

    def __init__(
        self,
        manifest_path="LMVD/processed_811/manifest.csv",
        fold="train",
        max_seqlen=200,
        random_crop=True,
        stats_path="LMVD/processed_811/lmvd_stats.npz",
        norm_clip=10.0,
    ):
        super().__init__()

        self.manifest_path = manifest_path
        self.fold = fold
        self.max_seqlen = int(max_seqlen)
        self.random_crop = bool(random_crop)
        self.norm_clip = float(norm_clip)

        df = pd.read_csv(manifest_path)

        if "fold" not in df.columns:
            raise KeyError(f"manifest missing fold column: {manifest_path}")

        if "label" not in df.columns:
            raise KeyError(f"manifest missing label column: {manifest_path}")

        if "feature_path" not in df.columns:
            raise KeyError(f"manifest missing feature_path column: {manifest_path}")

        self.df = df[df["fold"] == fold].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(f"No samples found for fold={fold} in {manifest_path}")

        self.mean = None
        self.std = None
        self.clip_low = None
        self.clip_high = None

        if stats_path is not None:
            stats = np.load(stats_path)

            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

            if "clip_low" in stats.files and "clip_high" in stats.files:
                self.clip_low = stats["clip_low"].astype(np.float32)
                self.clip_high = stats["clip_high"].astype(np.float32)

        self.samples = []

        for _, row in self.df.iterrows():
            self.samples.append(
                (
                    str(row["sid"]),
                    str(row["feature_path"]),
                    int(row["label"]),
                    int(row["seq_len"]) if "seq_len" in row else -1,
                )
            )

    def __len__(self):
        return len(self.samples)

    def _load_feature(self, path):
        feat = np.load(path).astype(np.float32)

        if feat.ndim != 2:
            raise ValueError(f"Expected feature [T, D], got {feat.shape} from {path}")

        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        if self.clip_low is not None and self.clip_high is not None:
            feat = np.clip(feat, self.clip_low, self.clip_high)

        if self.mean is not None and self.std is not None:
            feat = (feat - self.mean) / self.std

        if self.norm_clip is not None and self.norm_clip > 0:
            feat = np.clip(feat, -self.norm_clip, self.norm_clip)

        return feat.astype(np.float32)

    def _crop_or_pad(self, feat):
        t, d = feat.shape
        l = self.max_seqlen

        out = np.zeros((l, d), dtype=np.float32)

        if t <= 0:
            return out

        if t <= l:
            out[:t] = feat
            return out

        if self.random_crop:
            start = np.random.randint(0, t - l + 1)
        else:
            start = max((t - l) // 2, 0)

        out[:] = feat[start:start + l]

        return out

    def __getitem__(self, index):
        sid, feature_path, label, seq_len = self.samples[index]

        feat = self._load_feature(feature_path)
        feat = self._crop_or_pad(feat)

        return torch.from_numpy(feat).float(), torch.tensor(label).float()
