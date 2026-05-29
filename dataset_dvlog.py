# dataset_dvlog.py
from pathlib import Path
import csv
import numpy as np
import torch.utils.data as data

from preprocess import process_feat


class DVlogDataset(data.Dataset):
    """
    D-Vlog dataset adapter for HyperVD.

    Expected structure:
      root/
        labels.csv
        <sid>/<sid>_visual.npy
        <sid>/<sid>_acoustic.npy

    labels.csv expected columns:
      sid,label,*,gender,fold
    where label in {depression, non-depression}
    and fold in {train, valid, test}.
    """

    def __init__(
        self,
        root,
        fold="train",
        gender="both",
        max_seqlen=596,
        random_crop=True,
        stats_path=None,
    ):
        self.root = Path(root)
        self.fold = fold
        self.gender = gender
        self.max_seqlen = int(max_seqlen)
        self.random_crop = random_crop and (fold == "train")
        self.samples = []

        label_file = self.root / "labels.csv"
        if not label_file.exists():
            raise FileNotFoundError(f"Cannot find {label_file}")

        with open(label_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue

                sid = row[0].strip()
                label_text = row[1].strip().lower()
                sample_gender = row[3].strip().lower()
                sample_fold = row[4].strip().lower()

                # skip possible header
                if sid.lower() in {"id", "sample_id", "video_id"}:
                    continue

                if sample_fold != self.fold.lower():
                    continue

                if self.gender != "both" and sample_gender != self.gender.lower():
                    continue

                visual_path = self.root / sid / f"{sid}_visual.npy"
                acoustic_path = self.root / sid / f"{sid}_acoustic.npy"

                if not visual_path.exists() or not acoustic_path.exists():
                    raise FileNotFoundError(
                        f"Missing feature files for {sid}: "
                        f"{visual_path}, {acoustic_path}"
                    )

                label = 1.0 if label_text == "depression" else 0.0
                self.samples.append((sid, visual_path, acoustic_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for fold={fold}, gender={gender}. "
                f"Check labels.csv fold names."
            )

        self.mean = None
        self.std = None
        if stats_path is not None and Path(stats_path).exists():
            stats = np.load(stats_path)
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sid, visual_path, acoustic_path, label = self.samples[index]

        visual = np.load(visual_path).astype(np.float32)      # [T_v, 136]
        acoustic = np.load(acoustic_path).astype(np.float32)  # [T_a, 25]

        # Align sequence length.
        t = min(visual.shape[0], acoustic.shape[0])
        visual = visual[:t]
        acoustic = acoustic[:t]

        # Concatenate as [visual, acoustic].
        feat = np.concatenate([visual, acoustic], axis=1).astype(np.float32)

        # Optional global train-set z-score.
        if self.mean is not None and self.std is not None:
            feat = (feat - self.mean) / self.std

        # HyperVD temporal graph is O(T^2), so do not feed very long full vlogs.
        feat = process_feat(
            feat,
            self.max_seqlen,
            is_random=self.random_crop,
        ).astype(np.float32)

        return feat, np.float32(label)