import numpy as np

from dataset_dvlog import DVlogDataset


class DVlogMultiWindowDataset(DVlogDataset):
    """
    Multi-window D-Vlog dataset.

    Return:
        windows: [K, max_seqlen, 161]
        label  : scalar

    During training:
        random_windows=True, randomly sample K windows.

    During validation/test:
        random_windows=False, use evenly-spaced deterministic windows.
    """

    def __init__(
        self,
        root,
        fold="train",
        gender="both",
        max_seqlen=200,
        num_windows=3,
        random_windows=True,
        stats_path=None,
    ):
        super().__init__(
            root=root,
            fold=fold,
            gender=gender,
            max_seqlen=max_seqlen,
            random_crop=False,
            stats_path=stats_path,
        )

        self.num_windows = int(num_windows)
        self.random_windows = bool(random_windows)

        if self.num_windows < 1:
            raise ValueError(f"num_windows must be >= 1, got {self.num_windows}")

    def _make_windows(self, feat):
        """
        feat: [T, C], already normalized.
        return: [K, L, C]
        """

        t, c = feat.shape
        k = self.num_windows
        l = self.max_seqlen

        windows = np.zeros((k, l, c), dtype=np.float32)

        if t <= 0:
            return windows

        if t <= l:
            # Duplicate the padded short sequence K times.
            valid_len = min(t, l)
            for i in range(k):
                windows[i, :valid_len] = feat[:valid_len]
            return windows

        max_start = t - l

        if self.random_windows:
            starts = np.random.randint(0, max_start + 1, size=k)
        else:
            starts = np.linspace(0, max_start, k)
            starts = np.round(starts).astype(np.int64)

        for i, s in enumerate(starts):
            windows[i] = feat[s:s + l]

        return windows

    def __getitem__(self, index):
        sid, visual_path, acoustic_path, label = self.samples[index]

        visual = np.load(visual_path).astype(np.float32)
        acoustic = np.load(acoustic_path).astype(np.float32)

        t = min(visual.shape[0], acoustic.shape[0])
        visual = visual[:t]
        acoustic = acoustic[:t]

        feat = np.concatenate([visual, acoustic], axis=1).astype(np.float32)

        if self.mean is not None and self.std is not None:
            feat = (feat - self.mean) / self.std

        windows = self._make_windows(feat)

        return windows.astype(np.float32), np.float32(label)