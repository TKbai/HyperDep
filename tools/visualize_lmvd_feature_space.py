import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import option
from model import Model
from dataset_lmvd import LMVDDataset
from torch.utils.data import ConcatDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest-path", default="LMVD/processed_811/manifest.csv")
    parser.add_argument("--stats-path", default="LMVD/processed_811/lmvd_stats.npz")
    parser.add_argument("--fold", default="test")
    parser.add_argument("--max-seqlen", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9)

    parser.add_argument(
        "--euclid-ckpt",
        default="./ckpt_lmvd/hypervd_lmvd_seq300_detour_a05_euclidean_seed9_best.pkl",
    )

    parser.add_argument(
        "--hyper-ckpt",
        default="./ckpt_lmvd/hypervd_lmvd_seq300_detour_a05_selauprc_thrf1pos_seed9_best.pkl",
    )

    parser.add_argument(
        "--out-prefix",
        default="figures_lmvd/lmvd_feature_space_seed9",
    )


    parser.add_argument(
        "--vis-level",
        type=str,
        default="video",
        choices=["video", "pseudo_snippet"],
    )

    parser.add_argument(
        "--snippet-mode",
        type=str,
        default="all_valid",
        choices=["all_valid", "sampled"],
        help="Pseudo-snippet collection mode: all_valid or sampled.",
    )

    parser.add_argument(
        "--snippet-sample-k",
        type=int,
        default=8,
        help="Number of valid snippets sampled per video for pseudo-snippet visualization."
    )

    parser.add_argument(
        "--hyper-snippet-sample-k",
        type=int,
        default=None,
        help="Optional larger sample-k only for hyperbolic panel. For visualization only.",
    )

    parser.add_argument(
        "--raw-snippet-sample-k",
        type=int,
        default=None,
        help=(
            "Optional larger sample-k only for the raw/vanilla panel. "
            "If None, use --snippet-sample-k."
        ),
    )

    return parser.parse_args()


def flatten_valid_snippets_sampled(seq_feat, seq_len, labels, sample_k=8, seed=9):
    """
    seq_feat: [B, T, D]
    seq_len : [B]
    labels  : [B]

    return:
        feat_flat  : [N, D]
        label_flat : [N]
        index_info : list of sampled snippet indices for each video
    """
    seq_feat = np.asarray(seq_feat)
    seq_len = np.asarray(seq_len).astype(int)
    labels = np.asarray(labels).astype(int)

    rng = np.random.default_rng(seed)

    feat_list = []
    label_list = []
    index_info = []

    for i in range(seq_feat.shape[0]):
        L = int(seq_len[i])
        L = max(1, min(L, seq_feat.shape[1]))

        if L <= sample_k:
            idx = np.arange(L)
        else:
            idx = np.sort(rng.choice(L, size=sample_k, replace=False))

        feat_i = seq_feat[i, idx]
        label_i = np.full((len(idx),), labels[i], dtype=np.int64)

        feat_list.append(feat_i)
        label_list.append(label_i)
        index_info.append(idx)

    feat_flat = np.concatenate(feat_list, axis=0)
    label_flat = np.concatenate(label_list, axis=0)

    return feat_flat, label_flat, index_info

def flatten_valid_snippets(seq_feat, seq_len, labels):
    """
    seq_feat: [B, T, D]
    seq_len : [B]
    labels  : [B]

    return:
        feat_flat  : [N, D]
        label_flat : [N]
    """
    seq_feat = np.asarray(seq_feat)
    seq_len = np.asarray(seq_len).astype(int)
    labels = np.asarray(labels).astype(int)

    feat_list = []
    label_list = []

    for i in range(seq_feat.shape[0]):
        L = int(seq_len[i])
        L = max(1, min(L, seq_feat.shape[1]))

        feat_i = seq_feat[i, :L]          # [L, D]
        label_i = np.full((L,), labels[i], dtype=np.int64)

        feat_list.append(feat_i)
        label_list.append(label_i)

    feat_flat = np.concatenate(feat_list, axis=0)
    label_flat = np.concatenate(label_list, axis=0)

    return feat_flat, label_flat



def flatten_snippets_by_mode(
    seq_feat,
    seq_len,
    labels,
    snippet_mode="all_valid",
    sample_k=8,
    seed=9,
):
    """
    Flatten sequence features for pseudo-snippet visualization.

    Returns:
        feat_flat: [N, D]
        label_flat: [N]
        index_info: sampled/used indices for each video
    """
    if snippet_mode == "all_valid":
        feat_flat, label_flat = flatten_valid_snippets(
            seq_feat,
            seq_len,
            labels,
        )

        seq_feat = np.asarray(seq_feat)
        seq_len = np.asarray(seq_len).astype(int)
        index_info = []
        for i in range(seq_feat.shape[0]):
            L = int(seq_len[i])
            L = max(1, min(L, seq_feat.shape[1]))
            index_info.append(np.arange(L))

        return feat_flat, label_flat, index_info

    if snippet_mode == "sampled":
        return flatten_valid_snippets_sampled(
            seq_feat,
            seq_len,
            labels,
            sample_k=sample_k,
            seed=seed,
        )

    raise ValueError(f"Unknown snippet_mode: {snippet_mode}")


def compute_seq_len(x):
    """
    x: [B, T, C]
    padded positions are all zeros.
    """
    seq_len = torch.sum(
        torch.max(torch.abs(x), dim=2)[0] > 0,
        dim=1,
    )
    return torch.clamp(seq_len, min=1)

def trim_to_batch_max_len(x):
    """
    Match train/eval behavior:
        compute seq_len first, then trim input to batch max valid length.
    """
    seq_len = compute_seq_len(x)

    max_len = int(seq_len.max().item())
    max_len = max(1, max_len)

    x = x[:, :max_len, :]

    return x, seq_len


def trim_to_batch_max_len(x):
    """
    Match train/eval behavior:
        compute seq_len first, then trim input to batch max valid length.

    x: [B, T, C]
    return:
        x_trimmed: [B, max_len, C]
        seq_len: [B]
    """
    seq_len = compute_seq_len(x)

    max_len = int(seq_len.max().item())
    max_len = max(1, max_len)

    x = x[:, :max_len, :]

    return x, seq_len


def masked_mean(x, seq_len):
    """
    x: [B, T, D]
    seq_len: [B]
    return: [B, D]

    Use torch.where instead of x * mask because:
        0 * NaN = NaN
    """
    b, t, d = x.shape

    valid_len = seq_len.long().to(x.device)
    valid_len = torch.clamp(valid_len, min=1, max=t)

    mask = (
        torch.arange(t, device=x.device).unsqueeze(0)
        < valid_len.unsqueeze(1)
    )

    x = torch.where(
        mask.unsqueeze(-1),
        x,
        torch.zeros_like(x),
    )

    denom = valid_len.unsqueeze(-1).to(x.dtype)

    return x.sum(dim=1) / denom


def load_model_from_ckpt(ckpt_path, device, expected_manifold=None):
    """
    Load model using args saved in checkpoint.

    This avoids visualization-time argument mismatch.
    """
    try:
        ckpt = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            ckpt_path,
            map_location=device,
        )

    if "args" not in ckpt:
        raise RuntimeError(
            f"Checkpoint does not contain saved args: {ckpt_path}"
        )

    saved_args = dict(ckpt["args"])

    default_args = vars(option.parser.parse_args([]))
    default_args.update(saved_args)

    default_args["device"] = device

    if "temporal_gamma" not in default_args:
        default_args["temporal_gamma"] = 2.718281828459045

    model_args = argparse.Namespace(**default_args)

    if expected_manifold is not None:
        ckpt_manifold = str(getattr(model_args, "manifold", "")).lower()
        if ckpt_manifold != expected_manifold.lower():
            raise RuntimeError(
                f"Checkpoint manifold mismatch: "
                f"expected {expected_manifold}, got {model_args.manifold}"
            )

    model = Model(model_args).to(device)

    result = model.load_state_dict(
        ckpt["model_state_dict"],
        strict=False,
    )

    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)

    print("=" * 80)
    print("Loaded checkpoint:", ckpt_path)
    print("manifold          :", getattr(model_args, "manifold", "unknown"))
    print("feature_refiner   :", getattr(model_args, "feature_adj_refiner", "none"))
    print("fusion            :", getattr(model_args, "fusion", "unknown"))
    print("pooling           :", getattr(model_args, "pooling", "unknown"))
    print("pool_alpha        :", getattr(model_args, "pool_alpha", "unknown"))
    print("adj_mode          :", getattr(model_args, "adj_mode", "unknown"))
    print("adj_threshold     :", getattr(model_args, "adj_threshold", "unknown"))
    print("temporal_gamma    :", getattr(model_args, "temporal_gamma", "unknown"))
    print("graph_branch      :", getattr(model_args, "graph_branch", "unknown"))
    print("epoch             :", ckpt.get("epoch", "unknown"))
    print("best_score        :", ckpt.get("best_score", "unknown"))
    print("best_threshold    :", ckpt.get("best_threshold", "unknown"))
    print("selection_metric  :", ckpt.get("selection_metric", "unknown"))
    print("threshold_metric  :", ckpt.get("threshold_metric", "unknown"))
    print("missing_keys      :", missing_keys)
    print("unexpected_keys   :", unexpected_keys)
    print("=" * 80)

    if len(unexpected_keys) > 0:
        raise RuntimeError(
            f"Unexpected keys when loading checkpoint:\n{unexpected_keys}"
        )

    if len(missing_keys) > 0:
        raise RuntimeError(
            f"Missing keys when loading checkpoint:\n{missing_keys}"
        )

    model.eval()

    return model


def collect_raw_features(
    loader,
    vis_level="video",
    snippet_mode="all_valid",
    snippet_sample_k=8,
    seed=9,
):
    feats = []
    labels = []

    for batch_idx, (x, y) in enumerate(loader):
        x, seq_len = trim_to_batch_max_len(x)

        x_np = x.float().cpu().numpy()
        y_np = y.numpy().astype(int)
        seq_len_np = seq_len.numpy().astype(int)

        if vis_level == "video":
            feat = masked_mean(x.float(), seq_len).cpu().numpy()
            feats.append(feat)
            labels.append(y_np)

        elif vis_level == "pseudo_snippet":
            feat_flat, label_flat, _ = flatten_snippets_by_mode(
                x_np,
                seq_len_np,
                y_np,
                snippet_mode=snippet_mode,
                sample_k=snippet_sample_k,
                seed=seed + batch_idx,
            )
            feats.append(feat_flat)
            labels.append(label_flat)

        else:
            raise ValueError(vis_level)

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    return feats, labels

def collect_model_embeddings(
    model,
    loader,
    device,
    is_hyper=False,
    vis_level="video",
    snippet_mode="all_valid",
    snippet_sample_k=8,
    seed=9,
):
    feats = []
    probs = []
    labels = []

    for batch_idx, (x, y) in enumerate(loader):
        x, seq_len = trim_to_batch_max_len(x)

        x = x.float().to(device)
        seq_len = seq_len.to(device)
        y_np = y.numpy().astype(int)
        seq_len_np = seq_len.cpu().numpy().astype(int)

        with torch.no_grad():
            if vis_level == "video":
                if is_hyper:
                    out = model(
                        x,
                        seq_len,
                        return_embedding=True,
                        return_tangent_embedding=True,
                    )
                else:
                    out = model(
                        x,
                        seq_len,
                        return_embedding=True,
                    )

                video_prob, frame_prob, emb = out[:3]
                feat_np = emb.detach().cpu().numpy()
                prob_np = video_prob.detach().cpu().numpy().reshape(-1)

                feats.append(feat_np)
                probs.append(prob_np)
                labels.append(y_np)

            elif vis_level == "pseudo_snippet":
                if is_hyper:
                    out = model(
                        x,
                        seq_len,
                        return_embedding=True,
                        return_tangent_embedding=True,
                        return_sequence_embedding=True,
                    )
                else:
                    out = model(
                        x,
                        seq_len,
                        return_embedding=True,
                        return_sequence_embedding=True,
                    )

                # The last returned tensor must be the sequence-level embedding:
                # Euclidean:  (..., seq_feat) with shape [B, T, 64]
                # Hyperbolic: (..., seq_feat) with shape [B, T, 62]
                video_prob = out[0]
                seq_feat = out[-1]

                if not torch.is_tensor(seq_feat):
                    raise TypeError(
                        f"Expected sequence embedding tensor as the last output, "
                        f"got {type(seq_feat)} with output length={len(out)}"
                    )

                if seq_feat.dim() != 3:
                    raise ValueError(
                        f"Expected seq_feat shape [B,T,D], got {seq_feat.shape}; "
                        f"output length={len(out)}"
                    )

                seq_feat_np = seq_feat.detach().cpu().numpy()
                video_prob_np = video_prob.detach().cpu().numpy().reshape(-1)

                feat_flat, label_flat, index_info = flatten_snippets_by_mode(
                    seq_feat_np,
                    seq_len_np,
                    y_np,
                    snippet_mode=snippet_mode,
                    sample_k=snippet_sample_k,
                    seed=seed + batch_idx,
                )

                prob_flat = np.concatenate(
                    [
                        np.full((len(index_info[i]),), video_prob_np[i], dtype=np.float32)
                        for i in range(len(video_prob_np))
                    ],
                    axis=0,
                )

                feats.append(feat_flat)
                probs.append(prob_flat)
                labels.append(label_flat)

            else:
                raise ValueError(vis_level)

    feats = np.concatenate(feats, axis=0)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)

    return feats, probs, labels

def reduce_to_2d(x, seed=9):
    """
    Standardize -> PCA -> t-SNE.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x = StandardScaler().fit_transform(x)

    n_samples, n_features = x.shape
    pca_dim = min(50, n_samples - 1, n_features)

    if pca_dim >= 2 and n_features > pca_dim:
        x = PCA(
            n_components=pca_dim,
            random_state=seed,
        ).fit_transform(x)

    perplexity = min(30, max(5, (n_samples - 1) // 3))

    z = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x)

    return z

def normalize_2d_to_unit_square(z, eps=1e-12):
    """
    Normalize 2D coordinates to [0, 1] x [0, 1].

    This is only for visualization, similar to many t-SNE figures.
    It does not change the embedding itself.
    """
    z = np.asarray(z, dtype=np.float32)

    z_min = z.min(axis=0, keepdims=True)
    z_span = np.ptp(z, axis=0, keepdims=True)

    z_norm = (z - z_min) / (z_span + eps)

    return z_norm


def plot_panel(ax, z, y, title):
    y = np.asarray(y).astype(int)

    # Cleaner but stronger colors
    color0 = "#2A9D8F"   # Non-depressed: teal
    color1 = "#E76F51"   # Depressed: coral red

    colors = np.where(y == 0, color0, color1)

    # Shuffle drawing order to avoid one class always covering the other
    rng = np.random.default_rng(9)
    order = rng.permutation(len(y))

    ax.scatter(
        z[order, 0],
        z[order, 1],
        s=8,
        alpha=0.78,
        c=colors[order],
        edgecolors="none",
        rasterized=True,
    )

    ax.set_title(title, fontsize=16, fontfamily="serif")

    ticks = np.linspace(0.0, 1.0, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.tick_params(
        axis="both",
        labelsize=7,
        width=0.6,
        length=3,
    )

    ax.grid(
        True,
        linewidth=0.5,
        alpha=0.40,
    )

    ax.set_aspect("equal", adjustable="box")


def main():
    args = parse_args()

    device = (
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and int(args.cuda) >= 0
        else "cpu"
    )

    print("device:", device)

    folds = [x.strip() for x in args.fold.split(",") if x.strip()]

    datasets = [
        LMVDDataset(
            manifest_path=args.manifest_path,
            fold=fold,
            max_seqlen=args.max_seqlen,
            random_crop=False,
            stats_path=args.stats_path,
            norm_clip=10.0,
        )
        for fold in folds
    ]

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    print("visualization folds:", folds)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    print("num samples:", len(dataset))

    # ------------------------------------------------------------
    # 1. Vanilla raw features
    # ------------------------------------------------------------
    raw_sample_k = (
        args.raw_snippet_sample_k
        if args.raw_snippet_sample_k is not None
        else args.snippet_sample_k
    )

    hyper_sample_k = (
        args.hyper_snippet_sample_k
        if args.hyper_snippet_sample_k is not None
        else args.snippet_sample_k
    )

    print("raw_snippet_sample_k   :", raw_sample_k)
    print("snippet_sample_k       :", args.snippet_sample_k)
    print("hyper_snippet_sample_k :", hyper_sample_k)

    raw_features, labels_raw = collect_raw_features(
        loader,
        vis_level=args.vis_level,
        snippet_mode=args.snippet_mode,
        snippet_sample_k=raw_sample_k,
        seed=args.seed,
    )

    print("raw_features:", raw_features.shape)
#    print("labels:", labels.shape, "label counts:", np.bincount(labels))

    # ------------------------------------------------------------
    # 2. Euclidean counterpart
    # ------------------------------------------------------------
    euclid_model = load_model_from_ckpt(
        ckpt_path=args.euclid_ckpt,
        device=device,
        expected_manifold="Euclidean",
    )

    euclid_features, euclid_prob, labels_e = collect_model_embeddings(
        euclid_model,
        loader,
        device=device,
        is_hyper=False,
        vis_level=args.vis_level,
        snippet_mode=args.snippet_mode,
        snippet_sample_k=args.snippet_sample_k,
        seed=args.seed,
    )

    if labels_raw.shape == labels_e.shape:
        if not np.all(labels_raw == labels_e):
            raise RuntimeError("Label mismatch between vanilla and euclidean features.")
    else:
        print(
            "WARNING: raw/vanilla panel uses a different number of pseudo-snippets. "
            f"raw N={len(labels_raw)}, euclid N={len(labels_e)}. "
            "This is for visualization only."
        )

    # ------------------------------------------------------------
    # 3. Hyperbolic model
    # ------------------------------------------------------------
    hyper_sample_k = (
        args.hyper_snippet_sample_k
        if args.hyper_snippet_sample_k is not None
        else args.snippet_sample_k
    )
    
    hyper_model = load_model_from_ckpt(
        ckpt_path=args.hyper_ckpt,
        device=device,
        expected_manifold="Lorentz",
    )

    hyper_features, hyper_prob, labels_h = collect_model_embeddings(
        hyper_model,
        loader,
        device=device,
        is_hyper=True,
        vis_level=args.vis_level,
        snippet_mode=args.snippet_mode,
        snippet_sample_k=hyper_sample_k,
        seed=args.seed,
    )

    if labels_raw.shape == labels_h.shape:
        if not np.all(labels_raw == labels_h):
            raise RuntimeError("Label mismatch between vanilla and hyperbolic features.")
    else:
        print(
            "WARNING: hyperbolic panel uses a different number of pseudo-snippets. "
            f"raw N={len(labels_raw)}, hyper N={len(labels_h)}. "
            "This is for visualization only."
        )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = out_prefix.with_suffix(".npz")

    # ------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------
    z_raw = reduce_to_2d(raw_features, seed=args.seed)
    z_euclid = reduce_to_2d(euclid_features, seed=args.seed)
    z_hyper = reduce_to_2d(hyper_features, seed=args.seed)

    # Normalize each t-SNE panel to [0, 1] for paper-style coordinates.
    z_raw = normalize_2d_to_unit_square(z_raw)
    z_euclid = normalize_2d_to_unit_square(z_euclid)
    z_hyper = normalize_2d_to_unit_square(z_hyper)

    np.savez(
        npz_path,
        labels_raw=labels_raw,
        labels_e=labels_e,
        labels_h=labels_h,
        raw_features=raw_features,
        euclid_features=euclid_features,
        hyper_features=hyper_features,
        euclid_prob=euclid_prob,
        hyper_prob=hyper_prob,
        z_raw=z_raw,
        z_euclid=z_euclid,
        z_hyper=z_hyper,
        raw_snippet_sample_k=raw_sample_k,
        snippet_sample_k=args.snippet_sample_k,
        hyper_snippet_sample_k=hyper_sample_k,
    )
    print("saved embeddings and 2D coordinates:", npz_path)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    plot_panel(axes[0], z_raw, labels_raw, "Vanilla")

    plot_panel(axes[1], z_euclid, labels_e, "trained in euclidean space")

    plot_panel(
        axes[2],
        z_hyper,
        labels_h,
        "trained in hyperbolic space",
    )





    plt.tight_layout()

    for suffix in [".png", ".pdf", ".svg"]:
        out_path = out_prefix.with_suffix(suffix)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print("saved figure:", out_path)

    print("LMVD feature space visualization finished.")


if __name__ == "__main__":
    main()
