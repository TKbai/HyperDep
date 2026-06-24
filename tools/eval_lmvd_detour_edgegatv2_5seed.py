from pathlib import Path

import numpy as np
import pandas as pd
import torch
import option

from torch.utils.data import DataLoader

from dataset_lmvd import LMVDDataset
from model import Model
from eval_dvlog import collect_dvlog_outputs, compute_binary_metrics


SEEDS = [9, 42, 2024, 7, 123]

METRICS = [
    "acc",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "auc",
    "auprc",
]


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path

    raise FileNotFoundError(
        "No checkpoint found in candidates:\n"
        + "\n".join(str(p) for p in paths)
    )


def baseline_checkpoint(seed):
    return first_existing([
        # New seed7/123 naming.
        f"./ckpt_lmvd/"
        f"hypervd_lmvd_seq300_detour_a05_"
        f"selauprc_thrf1weighted_seed{seed}_best.pkl",

        # Existing seed9/42/2024 naming.
        f"./ckpt_lmvd/"
        f"hypervd_lmvd_seq300_detour_a05_"
        f"selauprc_thrf1pos_seed{seed}_best.pkl",
    ])


def gatv2_checkpoint(seed):
    return first_existing([
        f"./ckpt_lmvd/"
        f"hypervd_lmvd_seq300_detour_a05_"
        f"edgegatv2_seed{seed}_best.pkl",
    ])


def build_args(use_gatv2):
    argv = [
        "--visual-dim", "465",
        "--audio-dim", "128",
        "--visual-proj-dim", "128",
        "--audio-proj-dim", "128",
        "--feat-dim", "256",

        "--max-seqlen", "300",
        "--batch-size", "4",

        "--fusion", "detour_adapted",

        "--pooling", "topk_mean",
        "--pool-alpha", "0.5",
        "--topk-divisor", "16",

        "--adj-mode", "soft_threshold",
        "--adj-threshold", "0.8",

        "--graph-branch", "both",
        "--feature-branch-weight", "1.0",
        "--temporal-branch-weight", "1.0",

        "--cuda", "0",
    ]

    if use_gatv2:
        argv.extend([
            "--feature-adj-refiner", "gatv2",
            "--edge-gatv2-hidden", "32",
            "--edge-gatv2-dropout", "0.1",
            "--edge-gatv2-delta-scale", "1.0",
            "--edge-gatv2-use-temporal", "1",
        ])

    args = option.parser.parse_args(argv)

    args.device = (
        "cuda:0"
        if torch.cuda.is_available() and int(args.cuda) >= 0
        else "cpu"
    )

    args.manifest_path = "LMVD/processed_811/manifest.csv"
    args.stats_path = "LMVD/processed_811/lmvd_stats.npz"
    args.lmvd_norm_clip = 10.0

    return args


def build_loader(args, fold):
    data = LMVDDataset(
        manifest_path=args.manifest_path,
        fold=fold,
        max_seqlen=args.max_seqlen,
        random_crop=False,
        stats_path=args.stats_path,
        norm_clip=args.lmvd_norm_clip,
    )

    return DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )


def make_thresholds(y_prob):
    y_prob = np.asarray(y_prob, dtype=np.float64)

    dense_tiny = np.linspace(1e-8, 1e-5, 200)
    dense_low = np.linspace(1e-5, 0.0500, 500)
    normal = np.linspace(0.055, 0.950, 180)

    unique_probs = np.unique(y_prob)

    if len(unique_probs) >= 2:
        midpoints = (
            unique_probs[:-1] + unique_probs[1:]
        ) / 2.0
    else:
        midpoints = unique_probs

    thresholds = np.concatenate([
        dense_tiny,
        dense_low,
        normal,
        unique_probs,
        midpoints,
    ])

    return np.unique(
        np.clip(thresholds, 1e-10, 0.999999)
    )


def search_f1_weighted_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for threshold in make_thresholds(y_prob):
        metrics = compute_binary_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
        )

        score = metrics["f1_weighted"]

        if np.isnan(score):
            continue

        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_score, best_metrics


def load_model(args, checkpoint_path):
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location=args.device,
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            checkpoint_path,
            map_location=args.device,
        )

    model = Model(args).to(args.device)

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model, ckpt


def evaluate_one(
    model_name,
    seed,
    checkpoint_path,
    use_gatv2,
):
    args = build_args(use_gatv2=use_gatv2)

    val_loader = build_loader(args, fold="valid")
    test_loader = build_loader(args, fold="test")

    model, ckpt = load_model(
        args=args,
        checkpoint_path=checkpoint_path,
    )

    y_val, p_val = collect_dvlog_outputs(
        dataloader=val_loader,
        model=model,
        args=args,
    )

    threshold, val_score, _ = search_f1_weighted_threshold(
        y_true=y_val,
        y_prob=p_val,
    )

    y_test, p_test = collect_dvlog_outputs(
        dataloader=test_loader,
        model=model,
        args=args,
    )

    test_metrics = compute_binary_metrics(
        y_true=y_test,
        y_prob=p_test,
        threshold=threshold,
    )

    row = {
        "model": model_name,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt.get("epoch", -1),
        "val_threshold": threshold,
        "val_f1_weighted": val_score,
    }

    for key in METRICS:
        row[key] = float(test_metrics[key])

    row["fp"] = int(test_metrics["fp"])
    row["fn"] = int(test_metrics["fn"])

    return row


def print_row(row):
    print(
        f"{row['model']:12s} "
        f"seed={row['seed']:4d} | "
        f"thr={row['val_threshold']:.7g} | "
        f"acc={row['acc']:.4f} | "
        f"prec+={row['precision_pos']:.4f} | "
        f"rec+={row['recall_pos']:.4f} | "
        f"f1+={row['f1_pos']:.4f} | "
        f"auc={row['auc']:.4f} | "
        f"auprc={row['auprc']:.4f} | "
        f"fp/fn={row['fp']}/{row['fn']}"
    )


def print_summary(df):
    print()
    print("=" * 100)
    print("FIVE-SEED SUMMARY: mean ± sample std")
    print("=" * 100)

    for model_name in ["baseline", "edge_gatv2"]:
        part = df[df["model"] == model_name]

        print()
        print(model_name)

        for key in METRICS:
            values = part[key].to_numpy(dtype=np.float64)
            print(
                f"  {key:14s}: "
                f"{values.mean():.4f} ± "
                f"{values.std(ddof=1):.4f}"
            )


def print_paired_delta(df):
    print()
    print("=" * 100)
    print("PAIRED DELTA: Edge-GATv2 - Baseline")
    print("=" * 100)

    baseline = (
        df[df["model"] == "baseline"]
        .set_index("seed")
        .sort_index()
    )

    gatv2 = (
        df[df["model"] == "edge_gatv2"]
        .set_index("seed")
        .sort_index()
    )

    common_seeds = sorted(
        set(baseline.index) & set(gatv2.index)
    )

    for seed in common_seeds:
        print()
        print(f"seed={seed}")

        for key in METRICS:
            delta = (
                gatv2.loc[seed, key]
                - baseline.loc[seed, key]
            )

            print(f"  delta_{key:14s}: {delta:+.4f}")

    print()
    print("Mean paired delta:")

    for key in METRICS:
        deltas = np.asarray([
            gatv2.loc[seed, key]
            - baseline.loc[seed, key]
            for seed in common_seeds
        ])

        print(
            f"  delta_{key:14s}: "
            f"{deltas.mean():+.4f} ± "
            f"{deltas.std(ddof=1):.4f}"
        )

    auprc_wins = sum(
        gatv2.loc[seed, "auprc"]
        > baseline.loc[seed, "auprc"]
        for seed in common_seeds
    )

    auc_wins = sum(
        gatv2.loc[seed, "auc"]
        > baseline.loc[seed, "auc"]
        for seed in common_seeds
    )

    f1_wins = sum(
        gatv2.loc[seed, "f1_pos"]
        > baseline.loc[seed, "f1_pos"]
        for seed in common_seeds
    )

    print()
    print(
        f"AUPRC paired wins: {auprc_wins}/{len(common_seeds)}"
    )
    print(
        f"AUC paired wins  : {auc_wins}/{len(common_seeds)}"
    )
    print(
        f"F1-pos wins      : {f1_wins}/{len(common_seeds)}"
    )


def main():
    rows = []

    for seed in SEEDS:
        baseline_path = baseline_checkpoint(seed)
        gatv2_path = gatv2_checkpoint(seed)

        baseline_row = evaluate_one(
            model_name="baseline",
            seed=seed,
            checkpoint_path=baseline_path,
            use_gatv2=False,
        )

        gatv2_row = evaluate_one(
            model_name="edge_gatv2",
            seed=seed,
            checkpoint_path=gatv2_path,
            use_gatv2=True,
        )

        rows.extend([
            baseline_row,
            gatv2_row,
        ])

        print()
        print_row(baseline_row)
        print_row(gatv2_row)

    df = pd.DataFrame(rows)

    out_path = Path(
        "results/lmvd_detour_vs_edgegatv2_"
        "5seed_f1weighted.csv"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(out_path, index=False)

    print_summary(df)
    print_paired_delta(df)

    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
