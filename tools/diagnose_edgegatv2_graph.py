import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import option
from dataset_lmvd import LMVDDataset
from model import Model


DEFAULT_SEEDS = [9, 42, 2024, 7, 123]


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Diagnose how Edge-GATv2 modifies the LMVD feature graph."
    )
    parser.add_argument(
        "--manifest-path",
        default="LMVD/processed_811/manifest.csv",
    )
    parser.add_argument(
        "--stats-path",
        default="LMVD/processed_811/lmvd_stats.npz",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./ckpt_lmvd",
    )
    parser.add_argument(
        "--fold",
        default="valid",
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of off-diagonal neighbors used for top-k overlap.",
    )
    parser.add_argument(
        "--change-tol",
        type=float,
        default=1e-5,
        help="Absolute adjacency-change tolerance for up/down fractions.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--out-dir",
        default="results/edgegatv2_graph_diagnosis",
    )
    return parser.parse_args()


def build_model_args(cuda_id):
    argv = [
        "--visual-dim", "465",
        "--audio-dim", "128",
        "--visual-proj-dim", "128",
        "--audio-proj-dim", "128",
        "--feat-dim", "256",
        "--max-seqlen", "300",
        "--batch-size", "2",
        "--fusion", "detour_adapted",
        "--pooling", "topk_mean",
        "--pool-alpha", "0.5",
        "--topk-divisor", "16",
        "--adj-mode", "soft_threshold",
        "--adj-threshold", "0.8",
        "--feature-adj-refiner", "gatv2",
        "--edge-gatv2-hidden", "32",
        "--edge-gatv2-dropout", "0.1",
        "--edge-gatv2-delta-scale", "1.0",
        "--edge-gatv2-use-temporal", "0",
        "--graph-branch", "both",
        "--feature-branch-weight", "1.0",
        "--temporal-branch-weight", "1.0",
        "--cuda", str(cuda_id),
    ]

    args = option.parser.parse_args(argv)
    args.device = (
        f"cuda:{cuda_id}"
        if torch.cuda.is_available() and cuda_id >= 0
        else "cpu"
    )
    return args


def checkpoint_path(checkpoint_dir, seed):
    return Path(checkpoint_dir) / (
        "hypervd_lmvd_seq300_detour_a05_"
        f"edgegatv2_notemp_seed{seed}_best.pkl"
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def compute_seq_len(x):
    seq_len = torch.sum(
        torch.max(torch.abs(x), dim=2)[0] > 0,
        dim=1,
    )
    return torch.clamp(seq_len, min=1)


def safe_entropy(p, eps=1e-12):
    return -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)


def offdiag_topk_overlap(base, refined, k):
    """
    base/refined: [L, L], row-normalized.
    Excludes self loops and returns mean row-wise Jaccard overlap.
    """
    length = base.shape[0]
    if length <= 1:
        return 1.0

    k = max(1, min(int(k), length - 1))
    eye = torch.eye(length, dtype=torch.bool, device=base.device)

    base_scores = base.masked_fill(eye, float("-inf"))
    refined_scores = refined.masked_fill(eye, float("-inf"))

    base_idx = torch.topk(base_scores, k=k, dim=-1).indices
    refined_idx = torch.topk(refined_scores, k=k, dim=-1).indices

    overlaps = []
    for row in range(length):
        base_set = set(base_idx[row].detach().cpu().tolist())
        refined_set = set(refined_idx[row].detach().cpu().tolist())
        union = base_set | refined_set
        overlaps.append(len(base_set & refined_set) / max(len(union), 1))

    return float(np.mean(overlaps))


def sample_graph_metrics(
    base,
    refined,
    scaled_delta,
    topk,
    change_tol,
):
    """All tensors are valid [L, L] graph slices."""
    length = base.shape[0]
    eps = 1e-12

    diff = refined - base
    abs_diff = diff.abs()

    base_entropy = safe_entropy(base)
    refined_entropy = safe_entropy(refined)

    normalizer = np.log(max(length, 2))
    base_entropy_norm = base_entropy / normalizer
    refined_entropy_norm = refined_entropy / normalizer

    row_tv = 0.5 * abs_diff.sum(dim=-1)

    diag_idx = torch.arange(length, device=base.device)
    base_self = base[diag_idx, diag_idx]
    refined_self = refined[diag_idx, diag_idx]

    if length > 1:
        pos = torch.arange(length, device=base.device, dtype=base.dtype)
        temporal_distance = torch.abs(pos[:, None] - pos[None, :]) / float(length - 1)
    else:
        temporal_distance = torch.zeros_like(base)

    base_expected_distance = (base * temporal_distance).sum(dim=-1)
    refined_expected_distance = (refined * temporal_distance).sum(dim=-1)

    base_indegree = base.sum(dim=0)
    refined_indegree = refined.sum(dim=0)

    base_indegree_cv = (
        base_indegree.std(unbiased=False) / base_indegree.mean().clamp_min(eps)
    )
    refined_indegree_cv = (
        refined_indegree.std(unbiased=False)
        / refined_indegree.mean().clamp_min(eps)
    )

    up = diff > change_tol
    down = diff < -change_tol
    unchanged = ~(up | down)

    metrics = {
        "seq_len": int(length),
        "mean_abs_adj_change": float(abs_diff.mean().item()),
        "max_abs_adj_change": float(abs_diff.max().item()),
        "mean_row_total_variation": float(row_tv.mean().item()),
        "frac_edges_up": float(up.float().mean().item()),
        "frac_edges_down": float(down.float().mean().item()),
        "frac_edges_unchanged": float(unchanged.float().mean().item()),
        "mean_abs_scaled_delta": float(scaled_delta.abs().mean().item()),
        "std_scaled_delta": float(scaled_delta.std(unbiased=False).item()),
        "base_entropy": float(base_entropy.mean().item()),
        "refined_entropy": float(refined_entropy.mean().item()),
        "entropy_change": float((refined_entropy - base_entropy).mean().item()),
        "base_normalized_entropy": float(base_entropy_norm.mean().item()),
        "refined_normalized_entropy": float(refined_entropy_norm.mean().item()),
        "base_effective_neighbors": float(torch.exp(base_entropy).mean().item()),
        "refined_effective_neighbors": float(torch.exp(refined_entropy).mean().item()),
        "base_max_edge_weight": float(base.max(dim=-1).values.mean().item()),
        "refined_max_edge_weight": float(refined.max(dim=-1).values.mean().item()),
        "base_self_loop_mass": float(base_self.mean().item()),
        "refined_self_loop_mass": float(refined_self.mean().item()),
        "base_expected_temporal_distance": float(base_expected_distance.mean().item()),
        "refined_expected_temporal_distance": float(
            refined_expected_distance.mean().item()
        ),
        "temporal_distance_change": float(
            (refined_expected_distance - base_expected_distance).mean().item()
        ),
        "base_indegree_cv": float(base_indegree_cv.item()),
        "refined_indegree_cv": float(refined_indegree_cv.item()),
        "topk_edge_jaccard": offdiag_topk_overlap(base, refined, topk),
    }

    return metrics


def summarize(df, group_cols):
    metric_cols = [
        col
        for col in df.columns
        if col not in {
            "seed",
            "sid",
            "label",
            "fold",
            "checkpoint_epoch",
            "score_weight_norm",
        }
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    rows = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        base_row = dict(zip(group_cols, keys))
        base_row["n"] = len(part)

        for metric in metric_cols:
            values = part[metric].to_numpy(dtype=np.float64)
            base_row[f"{metric}_mean"] = float(values.mean())
            base_row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

        rows.append(base_row)

    return pd.DataFrame(rows)


def print_core_summary(summary_df, title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    core = [
        "mean_abs_adj_change_mean",
        "mean_row_total_variation_mean",
        "frac_edges_up_mean",
        "frac_edges_down_mean",
        "topk_edge_jaccard_mean",
        "base_normalized_entropy_mean",
        "refined_normalized_entropy_mean",
        "base_effective_neighbors_mean",
        "refined_effective_neighbors_mean",
        "base_max_edge_weight_mean",
        "refined_max_edge_weight_mean",
        "base_self_loop_mass_mean",
        "refined_self_loop_mass_mean",
        "temporal_distance_change_mean",
        "base_indegree_cv_mean",
        "refined_indegree_cv_mean",
    ]

    available = [c for c in core if c in summary_df.columns]
    display_cols = [c for c in ["seed", "label", "n"] if c in summary_df.columns] + available

    print(summary_df[display_cols].to_string(index=False))


def main():
    cli = parse_cli()
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = build_model_args(cli.cuda)

    dataset = LMVDDataset(
        manifest_path=cli.manifest_path,
        fold=cli.fold,
        max_seqlen=args.max_seqlen,
        random_crop=False,
        stats_path=cli.stats_path,
        norm_clip=10.0,
    )

    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    all_rows = []

    for seed in cli.seeds:
        ckpt_path = checkpoint_path(cli.checkpoint_dir, seed)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ckpt = load_checkpoint(ckpt_path, args.device)
        model = Model(args).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()

        score_weight_norm = float(model.edge_gatv2.score.weight.detach().norm().item())
        checkpoint_epoch = int(ckpt.get("epoch", -1))

        print()
        print("#" * 110)
        print(
            f"seed={seed} | checkpoint={ckpt_path} | "
            f"epoch={checkpoint_epoch} | score_weight_norm={score_weight_norm:.6f}"
        )
        print("#" * 110)

        offset = 0

        with torch.no_grad():
            for batch_index, (x, y) in enumerate(loader):
                batch_size = x.shape[0]
                sample_meta = dataset.samples[offset : offset + batch_size]
                offset += batch_size

                seq_len = compute_seq_len(x)
                x = x.float().to(args.device)
                seq_len_device = seq_len.to(args.device)

                _, _, graph_info = model(
                    x,
                    seq_len_device,
                    return_graph_info=True,
                )

                for local_index in range(batch_size):
                    valid_len = int(seq_len[local_index].item())

                    base = graph_info["base_adj"][
                        local_index, :valid_len, :valid_len
                    ]
                    refined = graph_info["refined_adj"][
                        local_index, :valid_len, :valid_len
                    ]
                    scaled_delta = graph_info["scaled_delta"][
                        local_index, :valid_len, :valid_len
                    ]

                    metrics = sample_graph_metrics(
                        base=base,
                        refined=refined,
                        scaled_delta=scaled_delta,
                        topk=cli.topk,
                        change_tol=cli.change_tol,
                    )

                    sid, _, label, _ = sample_meta[local_index]

                    row = {
                        "seed": seed,
                        "sid": sid,
                        "label": int(label),
                        "fold": cli.fold,
                        "checkpoint_epoch": checkpoint_epoch,
                        "score_weight_norm": score_weight_norm,
                        **metrics,
                    }
                    all_rows.append(row)

                if (batch_index + 1) % 20 == 0 or offset == len(dataset):
                    print(
                        f"seed={seed}: processed {offset}/{len(dataset)} samples"
                    )

    per_sample = pd.DataFrame(all_rows)
    per_seed = summarize(per_sample, ["seed"])
    per_seed_class = summarize(per_sample, ["seed", "label"])
    class_summary = summarize(per_sample, ["label"])
    overall_summary = summarize(per_sample.assign(group="all"), ["group"])

    per_sample_path = out_dir / f"edgegatv2_graph_{cli.fold}_per_sample.csv"
    per_seed_path = out_dir / f"edgegatv2_graph_{cli.fold}_per_seed.csv"
    per_seed_class_path = out_dir / f"edgegatv2_graph_{cli.fold}_per_seed_class.csv"
    class_summary_path = out_dir / f"edgegatv2_graph_{cli.fold}_class_summary.csv"
    overall_path = out_dir / f"edgegatv2_graph_{cli.fold}_overall.csv"

    per_sample.to_csv(per_sample_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    per_seed_class.to_csv(per_seed_class_path, index=False)
    class_summary.to_csv(class_summary_path, index=False)
    overall_summary.to_csv(overall_path, index=False)

    print_core_summary(per_seed, "PER-SEED GRAPH REFINEMENT SUMMARY")
    print_core_summary(class_summary, "CLASS-CONDITIONAL GRAPH REFINEMENT SUMMARY")
    print_core_summary(overall_summary, "OVERALL GRAPH REFINEMENT SUMMARY")

    print()
    print("Saved:")
    for path in [
        per_sample_path,
        per_seed_path,
        per_seed_class_path,
        class_summary_path,
        overall_path,
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()
