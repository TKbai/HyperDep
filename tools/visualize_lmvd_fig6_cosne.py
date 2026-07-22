import argparse
from pathlib import Path
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import option
from model import Model
from dataset_lmvd import LMVDDataset


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
        "--hyper-ckpt",
        required=True,
        help="Lorentz checkpoint path",
    )

    parser.add_argument(
        "--out-prefix",
        required=True,
    )

    parser.add_argument(
        "--snippet-mode",
        type=str,
        default="sampled",
        choices=["all_valid", "sampled"],
    )

    parser.add_argument(
        "--snippet-sample-k",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--trained-branch",
        type=str,
        default="feature",
        choices=["feature", "temporal"],
        help="Which Lorentz branch to visualize as trained embeddings.",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=2500,
        help="Subsample points for CO-SNE-like optimization.",
    )

    parser.add_argument(
        "--cosne-steps",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--cosne-lr",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--origin-weight",
        type=float,
        default=0.15,
    )

    return parser.parse_args()


def compute_seq_len(x):
    seq_len = torch.sum(torch.max(torch.abs(x), dim=2)[0] > 0, dim=1)
    return torch.clamp(seq_len, min=1)


def trim_to_batch_max_len(x):
    seq_len = compute_seq_len(x)
    max_len = int(seq_len.max().item())
    max_len = max(1, max_len)
    return x[:, :max_len, :], seq_len


def load_lorentz_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    args_dict = vars(option.parser.parse_args([]))
    args_dict.update(dict(ckpt["args"]))
    args_dict["device"] = device

    if "temporal_gamma" not in args_dict:
        args_dict["temporal_gamma"] = 2.718281828459045

    model_args = argparse.Namespace(**args_dict)

    if str(model_args.manifold).lower() != "lorentz":
        raise RuntimeError(f"Expected Lorentz checkpoint, got {model_args.manifold}")

    model = Model(model_args).to(device)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    print("=" * 80)
    print("Loaded checkpoint:", ckpt_path)
    print("manifold        :", model_args.manifold)
    print("feature_refiner :", getattr(model_args, "feature_adj_refiner", "none"))
    print("epoch           :", ckpt.get("epoch", "unknown"))
    print("best_score      :", ckpt.get("best_score", "unknown"))
    print("missing_keys    :", list(result.missing_keys))
    print("unexpected_keys :", list(result.unexpected_keys))
    print("=" * 80)

    if len(result.missing_keys) > 0 or len(result.unexpected_keys) > 0:
        raise RuntimeError("Checkpoint mismatch.")

    model.eval()
    return model


def attach_lorentz_hooks(model):
    """
    Capture:
      proj_x  : input Lorentz points before HFSGCN
      x1_geo  : output of HFSGCN
      x2_geo  : output of HTRGCN
    """

    orig_hfsg = model.HFSGCN.encode
    orig_htrg = model.HTRGCN.encode

    def wrapped_hfsg(x, adj):
        model._last_proj_x = x.detach()
        out = orig_hfsg(x, adj)
        model._last_x1_geo = out.detach()
        return out

    def wrapped_htrg(x, adj):
        out = orig_htrg(x, adj)
        model._last_x2_geo = out.detach()
        return out

    model.HFSGCN.encode = wrapped_hfsg
    model.HTRGCN.encode = wrapped_htrg


def lorentz_to_poincare(x, k=1.0, eps=1e-8):
    """
    x: [..., D+1] Lorentz point
    return: [..., D] Poincaré ball coordinate
    """
    if torch.is_tensor(k):
        k = float(k.detach().cpu().item())

    sqrt_k = k ** 0.5

    time = x[..., 0:1]
    spatial = x[..., 1:]

    p = spatial / torch.clamp(time + sqrt_k, min=eps)

    # numerical safety
    norm = torch.linalg.norm(p, dim=-1, keepdim=True)
    max_norm = 1.0 - 1e-5
    p = torch.where(norm >= max_norm, p / norm.clamp_min(eps) * max_norm, p)

    return p


def flatten_by_mode(seq_feat, seq_len, labels, mode="sampled", sample_k=24, seed=9):
    """
    seq_feat: [B, T, D]
    """
    seq_feat = np.asarray(seq_feat)
    seq_len = np.asarray(seq_len).astype(int)
    labels = np.asarray(labels).astype(int)

    rng = np.random.default_rng(seed)

    feat_list = []
    label_list = []

    for i in range(seq_feat.shape[0]):
        L = int(seq_len[i])
        L = max(1, min(L, seq_feat.shape[1]))

        if mode == "all_valid":
            idx = np.arange(L)
        elif mode == "sampled":
            if L <= sample_k:
                idx = np.arange(L)
            else:
                idx = np.sort(rng.choice(L, size=sample_k, replace=False))
        else:
            raise ValueError(mode)

        feat_list.append(seq_feat[i, idx])
        label_list.append(np.full((len(idx),), labels[i], dtype=np.int64))

    return np.concatenate(feat_list, axis=0), np.concatenate(label_list, axis=0)


def collect_poincare_embeddings(model, loader, device, args):
    vanilla_list = []
    trained_list = []
    label_list = []

    k = getattr(model.manifold, "k", torch.tensor(1.0))

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, seq_len = trim_to_batch_max_len(x)

            x = x.float().to(device)
            seq_len = seq_len.to(device)

            _ = model(x, seq_len)

            proj_x = model._last_proj_x
            x1_geo = model._last_x1_geo
            x2_geo = model._last_x2_geo

            if args.trained_branch == "feature":
                trained_geo = x1_geo
            else:
                trained_geo = x2_geo

            vanilla_p = lorentz_to_poincare(proj_x, k=k)
            trained_p = lorentz_to_poincare(trained_geo, k=k)

            vanilla_np = vanilla_p.detach().cpu().numpy()
            trained_np = trained_p.detach().cpu().numpy()
            seq_len_np = seq_len.detach().cpu().numpy().astype(int)
            y_np = y.detach().cpu().numpy().astype(int)

            vanilla_flat, labels_flat = flatten_by_mode(
                vanilla_np,
                seq_len_np,
                y_np,
                mode=args.snippet_mode,
                sample_k=args.snippet_sample_k,
                seed=args.seed + batch_idx,
            )

            trained_flat, labels_flat_2 = flatten_by_mode(
                trained_np,
                seq_len_np,
                y_np,
                mode=args.snippet_mode,
                sample_k=args.snippet_sample_k,
                seed=args.seed + batch_idx,
            )

            if not np.array_equal(labels_flat, labels_flat_2):
                raise RuntimeError("Label mismatch between vanilla and trained embeddings.")

            vanilla_list.append(vanilla_flat)
            trained_list.append(trained_flat)
            label_list.append(labels_flat)

    vanilla = np.concatenate(vanilla_list, axis=0).astype(np.float32)
    trained = np.concatenate(trained_list, axis=0).astype(np.float32)
    labels = np.concatenate(label_list, axis=0).astype(np.int64)

    return vanilla, trained, labels


def poincare_dist_matrix_np(x, eps=1e-6):
    """
    Pairwise Poincaré distance for curvature -1.
    x: [N, D], ||x|| < 1
    """
    x = torch.tensor(x, dtype=torch.float32)
    x2 = (x * x).sum(dim=1, keepdim=True)
    diff2 = torch.cdist(x, x, p=2).pow(2)

    denom = (1.0 - x2).clamp_min(eps) @ (1.0 - x2).clamp_min(eps).T
    z = 1.0 + 2.0 * diff2 / denom
    z = z.clamp_min(1.0 + 1e-6)

    return torch.acosh(z)


def binary_search_perplexity(dist2, perplexity=30.0, max_iter=50):
    """
    t-SNE-style P matrix from hyperbolic squared distances.
    """
    n = dist2.shape[0]
    target_entropy = np.log(perplexity)

    P = torch.zeros_like(dist2)

    for i in range(n):
        beta_min = None
        beta_max = None
        beta = torch.tensor(1.0, device=dist2.device)

        d = torch.cat([dist2[i, :i], dist2[i, i + 1:]])

        for _ in range(max_iter):
            p = torch.exp(-d * beta)
            sum_p = p.sum().clamp_min(1e-12)
            p = p / sum_p

            entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
            diff = entropy - target_entropy

            if torch.abs(diff) < 1e-4:
                break

            if diff > 0:
                beta_min = beta.clone()
                beta = beta * 2.0 if beta_max is None else (beta + beta_max) / 2.0
            else:
                beta_max = beta.clone()
                beta = beta / 2.0 if beta_min is None else (beta + beta_min) / 2.0

        row = torch.zeros(n, device=dist2.device)
        row[:i] = p[:i]
        row[i + 1:] = p[i:]
        P[i] = row

    P = (P + P.T) / (2.0 * n)
    P = P / P.sum().clamp_min(1e-12)

    return P


def make_poincare_from_param(u, eps=1e-5):
    """
    Unconstrained R^2 parameter -> Poincaré disk.
    """
    norm = torch.linalg.norm(u, dim=1, keepdim=True).clamp_min(1e-8)
    radius = torch.tanh(norm) * (1.0 - eps)
    return radius * u / norm


def cosne_like(x, seed=9, perplexity=30, steps=800, lr=0.05, origin_weight=0.15):
    """
    Simplified CO-SNE-like optimization:
    - high-D similarity from hyperbolic Gaussian
    - low-D similarity from hyperbolic Cauchy
    - origin-distance preservation loss
    """
    torch.manual_seed(seed)

    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]

    X = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        D = poincare_dist_matrix_np(x)
        D2 = D.pow(2)

        P = binary_search_perplexity(
            D2,
            perplexity=min(perplexity, max(5, (n - 1) // 3)),
        )

        # Preserve absolute Poincaré radius instead of only standardized ranking.
        # This avoids pushing trained embeddings artificially to the boundary.
        rho_x = torch.linalg.norm(
            X,
            dim=1,
        ).clamp(min=0.0, max=1.0 - 1e-6)

    u = torch.randn(n, 2) * 1e-3
    u.requires_grad_(True)

    opt = torch.optim.Adam([u], lr=lr)

    for step in range(steps):
        Y = make_poincare_from_param(u)

        Dy = poincare_dist_matrix_np(Y.detach().cpu().numpy()).to(u.device)
        # Recompute with torch variable for gradient
        y2 = (Y * Y).sum(dim=1, keepdim=True)
        diff2 = torch.cdist(Y, Y).pow(2)
        denom = (1.0 - y2).clamp_min(1e-6) @ (1.0 - y2).clamp_min(1e-6).T
        z = (1.0 + 2.0 * diff2 / denom).clamp_min(1.0 + 1e-6)
        Dy = torch.acosh(z)

        Q = 1.0 / (1.0 + Dy.pow(2))
        Q.fill_diagonal_(0.0)
        Q = Q / Q.sum().clamp_min(1e-12)

        kl = (P * (torch.log(P.clamp_min(1e-12)) - torch.log(Q.clamp_min(1e-12)))).sum()

        rho_y = torch.linalg.norm(
            Y,
            dim=1,
        ).clamp(min=0.0, max=1.0 - 1e-6)

        origin_loss = torch.mean((rho_y - rho_x) ** 2)

        loss = kl + origin_weight * origin_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step={step:04d} loss={loss.item():.6f} kl={kl.item():.6f} origin={origin_loss.item():.6f}")

    Y = make_poincare_from_param(u).detach().cpu().numpy()
    return Y


def subsample_if_needed(vanilla, trained, labels, max_points, seed=9):
    n = labels.shape[0]

    if max_points is None or max_points <= 0 or n <= max_points:
        return vanilla, trained, labels

    rng = np.random.default_rng(seed)

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    n0 = max_points // 2
    n1 = max_points - n0

    pick0 = rng.choice(idx0, size=min(n0, len(idx0)), replace=False)
    pick1 = rng.choice(idx1, size=min(n1, len(idx1)), replace=False)

    idx = np.sort(np.concatenate([pick0, pick1]))

    return vanilla[idx], trained[idx], labels[idx]


def plot_disk(ax, y2d, labels, title):
    labels = np.asarray(labels).astype(int)

    color0 = "#2A9D8F"  # Non-depressed
    color1 = "#E76F51"  # Depressed

    colors = np.where(labels == 0, color0, color1)

    rng = np.random.default_rng(9)
    order = rng.permutation(len(labels))

    ax.scatter(
        y2d[order, 0],
        y2d[order, 1],
        s=8,
        alpha=0.78,
        c=colors[order],
        edgecolors="none",
        rasterized=True,
    )

    circle = plt.Circle((0, 0), 1.0, color="black", fill=False, linewidth=1.0)
    ax.add_patch(circle)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_title(title, fontsize=14, fontfamily="serif")


def main():
    args = parse_args()

    device = (
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and int(args.cuda) >= 0
        else "cpu"
    )

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

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    print("device:", device)
    print("folds:", folds)
    print("num samples:", len(dataset))

    model = load_lorentz_model_from_ckpt(args.hyper_ckpt, device)
    attach_lorentz_hooks(model)

    vanilla, trained, labels = collect_poincare_embeddings(model, loader, device, args)

    print("vanilla poincare:", vanilla.shape)
    print("trained poincare:", trained.shape)
    print("labels:", labels.shape, np.bincount(labels))

    vanilla, trained, labels = subsample_if_needed(
        vanilla,
        trained,
        labels,
        max_points=args.max_points,
        seed=args.seed,
    )

    print("after subsample:")
    print("vanilla:", vanilla.shape)
    print("trained:", trained.shape)
    print("labels:", labels.shape, np.bincount(labels))

    print("Running CO-SNE-like projection for vanilla...")
    y_vanilla = cosne_like(
        vanilla,
        seed=args.seed,
        steps=args.cosne_steps,
        lr=args.cosne_lr,
        origin_weight=args.origin_weight,
    )

    print("Running CO-SNE-like projection for trained...")
    y_trained = cosne_like(
        trained,
        seed=args.seed,
        steps=args.cosne_steps,
        lr=args.cosne_lr,
        origin_weight=args.origin_weight,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_prefix.with_suffix(".npz"),
        labels=labels,
        vanilla_poincare=vanilla,
        trained_poincare=trained,
        y_vanilla=y_vanilla,
        y_trained=y_trained,
        trained_branch=args.trained_branch,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))

    plot_disk(
        axes[0],
        y_vanilla,
        labels,
        "Vanilla Embeddings",
    )

    plot_disk(
        axes[1],
        y_trained,
        labels,
        "Trained Embeddings",
    )

    plt.tight_layout()

    for suffix in [".png", ".pdf", ".svg"]:
        path = out_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("saved:", path)


if __name__ == "__main__":
    main()