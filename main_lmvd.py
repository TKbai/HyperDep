import os
import random
import numpy as np
import torch

from torch.utils.data import DataLoader

import option
from model import Model
from train import train
from dataset_lmvd import LMVDDataset
from eval_dvlog import collect_dvlog_outputs, compute_binary_metrics, print_metrics


def add_argument_if_absent(parser, *flags, **kwargs):
    """
    Add argparse option only if it has not already been defined in option.py.
    """
    for flag in flags:
        if flag in parser._option_string_actions:
            return
    parser.add_argument(*flags, **kwargs)


# ---------------------------------------------------------------------
# LMVD-specific args
# ---------------------------------------------------------------------
add_argument_if_absent(
    option.parser,
    "--manifest-path",
    default="LMVD/processed_811/manifest.csv",
    help="LMVD processed manifest path",
)

add_argument_if_absent(
    option.parser,
    "--stats-path",
    default="LMVD/processed_811/lmvd_stats.npz",
    help="LMVD normalization stats path",
)

add_argument_if_absent(
    option.parser,
    "--save-dir",
    default="./ckpt_lmvd",
    help="directory to save LMVD checkpoints",
)

add_argument_if_absent(
    option.parser,
    "--train-fold",
    default="train",
    help="train fold name in manifest",
)

add_argument_if_absent(
    option.parser,
    "--val-fold",
    default="valid",
    help="validation fold name in manifest",
)

add_argument_if_absent(
    option.parser,
    "--test-fold",
    default="test",
    help="test fold name in manifest",
)

add_argument_if_absent(
    option.parser,
    "--lmvd-norm-clip",
    type=float,
    default=10.0,
    help="clip normalized LMVD features to [-clip, clip]",
)

add_argument_if_absent(
    option.parser,
    "--metric",
    default="f1",
    choices=[
        "f1",
        "f1_weighted",
        "f1_pos",
        "f1_macro",
        "balanced_acc",
        "auc",
        "auprc",
        "acc",
    ],
    help="metric used for checkpoint selection",
)


add_argument_if_absent(
    option.parser,
    "--selection-metric",
    default=None,
    choices=[
        "f1",
        "f1_weighted",
        "f1_pos",
        "f1_macro",
        "balanced_acc",
        "auc",
        "auprc",
        "acc",
    ],
    help="metric used for checkpoint selection",
)

add_argument_if_absent(
    option.parser,
    "--threshold-metric",
    default=None,
    choices=[
        "f1",
        "f1_weighted",
        "f1_pos",
        "f1_macro",
        "balanced_acc",
        "auc",
        "auprc",
        "acc",
    ],
    help="metric used for threshold search on validation set",
)

add_argument_if_absent(
    option.parser,
    "--weight-decay",
    type=float,
    default=0.0,
    help="weight decay",
)

add_argument_if_absent(
    option.parser,
    "--grad-clip",
    type=float,
    default=5.0,
    help="gradient clipping max norm",
)


def set_seed(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_thresholds(y_prob):
    """
    HyperVD-style probabilities are often very small,
    so we search densely in the low-threshold region.
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)

    dense_tiny = np.linspace(1e-8, 1e-5, 200)
    dense_low = np.linspace(1e-5, 0.0500, 500)
    normal = np.linspace(0.055, 0.950, 180)

    unique_probs = np.unique(y_prob)

    if len(unique_probs) >= 2:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
    else:
        midpoints = unique_probs

    thresholds = np.concatenate(
        [
            dense_tiny,
            dense_low,
            normal,
            unique_probs,
            midpoints,
        ]
    )

    thresholds = np.unique(np.clip(thresholds, 1e-10, 0.999999))

    return thresholds


def metric_value(metrics, metric_name):
    """
    metric_name='f1' is treated as weighted F1 if compute_binary_metrics
    does not expose a separate key named 'f1'.
    """
    if metric_name in metrics:
        return metrics[metric_name]

    if metric_name == "f1" and "f1_weighted" in metrics:
        return metrics["f1_weighted"]

    raise KeyError(
        f"Metric {metric_name} not found. Available keys: {list(metrics.keys())}"
    )


def search_best_threshold_from_outputs(
    y_true,
    y_prob,
    threshold_metric="f1",
):
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for threshold in make_thresholds(y_prob):
        metrics = compute_binary_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
        )

        score = metric_value(metrics, threshold_metric)

        if np.isnan(score):
            continue

        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_score, best_metrics


def evaluate_validation_for_checkpoint(
    dataloader,
    model,
    args,
):
    y_true, y_prob = collect_dvlog_outputs(
        dataloader=dataloader,
        model=model,
        args=args,
    )

    # 1. Search threshold using threshold_metric.
    best_threshold, threshold_score, threshold_metrics = search_best_threshold_from_outputs(
        y_true=y_true,
        y_prob=y_prob,
        threshold_metric=args.threshold_metric,
    )

    # 2. Select checkpoint using selection_metric.
    # For auc/auprc, this value is threshold-independent.
    selection_score = metric_value(
        threshold_metrics,
        args.selection_metric,
    )

    return best_threshold, threshold_score, selection_score, threshold_metrics


def evaluate_with_threshold(dataloader, model, args, threshold):
    y_true, y_prob = collect_dvlog_outputs(
        dataloader=dataloader,
        model=model,
        args=args,
    )

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=float(threshold),
    )

    return metrics


def build_loader(args, fold, shuffle, random_crop):
    dataset = LMVDDataset(
        manifest_path=args.manifest_path,
        fold=fold,
        max_seqlen=args.max_seqlen,
        random_crop=random_crop,
        stats_path=args.stats_path,
        norm_clip=args.lmvd_norm_clip,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=(args.device.startswith("cuda")),
        drop_last=False,
    )

    return dataset, loader


def load_best_checkpoint(model, ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    return ckpt


def main():
    args = option.parser.parse_args()

    # Robust type fixes for old option.py definitions.
    args.cuda = int(args.cuda)
    args.batch_size = int(args.batch_size)
    args.max_epoch = int(args.max_epoch)
    args.workers = int(getattr(args, "workers", 4))
    args.seed = int(getattr(args, "seed", 9))
    args.lr = float(args.lr)
    args.weight_decay = float(getattr(args, "weight_decay", 0.0))

    if getattr(args, "selection_metric", None) is None:
        args.selection_metric = args.metric

    if getattr(args, "threshold_metric", None) is None:
        args.threshold_metric = args.metric

    args.device = (
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_path = os.path.join(
        args.save_dir,
        f"{args.model_name}_best.pkl",
    )

    print("=" * 80)
    print("LMVD HyperVD baseline")
    print("=" * 80)
    print("manifest_path:", args.manifest_path)
    print("stats_path   :", args.stats_path)
    print("save_dir     :", args.save_dir)
    print("model_name   :", args.model_name)
    print("device       :", args.device)
    print("seed         :", args.seed)
    print()
    print("visual_dim   :", args.visual_dim)
    print("audio_dim    :", args.audio_dim)
    print("feat_dim     :", args.feat_dim)
    print("max_seqlen   :", args.max_seqlen)
    print("batch_size   :", args.batch_size)
    print("lr           :", args.lr)
    print("dropout      :", args.dropout)
    print("max_epoch    :", args.max_epoch)
    print("metric legacy      :", args.metric)
    print("selection_metric   :", args.selection_metric)
    print("threshold_metric   :", args.threshold_metric)
    print()
    print("fusion       :", getattr(args, "fusion", "unknown"))
    print("pooling      :", getattr(args, "pooling", "unknown"))
    print("pool_alpha   :", getattr(args, "pool_alpha", "unknown"))
    print("topk_divisor :", getattr(args, "topk_divisor", "unknown"))
    print("adj_mode     :", getattr(args, "adj_mode", "unknown"))
    print("adj_threshold:", getattr(args, "adj_threshold", "unknown"))
    print("graph_branch :", getattr(args, "graph_branch", "unknown"))
    print("=" * 80)

    train_data, train_loader = build_loader(
        args=args,
        fold=args.train_fold,
        shuffle=True,
        random_crop=True,
    )

    val_data, val_loader = build_loader(
        args=args,
        fold=args.val_fold,
        shuffle=False,
        random_crop=False,
    )

    test_data, test_loader = build_loader(
        args=args,
        fold=args.test_fold,
        shuffle=False,
        random_crop=False,
    )

    print()
    print("Dataset summary:")
    print("train samples:", len(train_data))
    print("valid samples:", len(val_data))
    print("test samples :", len(test_data))

    model = Model(args).to(args.device)

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = -1.0
    best_epoch = -1
    best_threshold = 0.5

    for epoch in range(1, args.max_epoch + 1):
        print()
        print("=" * 10, f"Epoch {epoch}/{args.max_epoch}", "=" * 10)

        train_loss, _, _, _ = train(
            train_loader,
            model,
            optimizer,
            args,
            criterion,
        )

        val_threshold, val_threshold_score, val_selection_score, val_metrics = evaluate_validation_for_checkpoint(
            dataloader=val_loader,
            model=model,
            args=args,
        )

        print("train_loss          :", f"{train_loss:.6f}")
        print("val_threshold       :", f"{val_threshold:.8f}")
        print(
            f"val_threshold_{args.threshold_metric:<8}:",
            f"{val_threshold_score:.6f}",
        )
        print(
            f"val_selection_{args.selection_metric:<8}:",
            f"{val_selection_score:.6f}",
        )

        if val_metrics is not None:
            print(
                "val_summary  : "
                f"acc={val_metrics['acc']:.4f}, "
                f"f1={metric_value(val_metrics, 'f1'):.4f}, "
                f"auc={val_metrics['auc']:.4f}, "
                f"auprc={val_metrics['auprc']:.4f}"
            )

        if val_selection_score > best_score:
            best_score = val_selection_score
            best_epoch = epoch
            best_threshold = val_threshold

            torch.save(
                {
                    "epoch": best_epoch,
                    "best_score": best_score,
                    "best_threshold": best_threshold,
                    "selection_metric": args.selection_metric,
                    "threshold_metric": args.threshold_metric,
                    "threshold_score": val_threshold_score,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )

            print("saved best model to:", ckpt_path)

    print()
    print("=" * 10, "Training Finished", "=" * 10)
    print("best_epoch    :", best_epoch)
    print("best_score    :", best_score)
    print("best_threshold:", best_threshold)

    # Load best model for final test evaluation.
    ckpt = load_best_checkpoint(
        model=model,
        ckpt_path=ckpt_path,
        device=args.device,
    )

    test_metrics = evaluate_with_threshold(
        dataloader=test_loader,
        model=model,
        args=args,
        threshold=best_threshold,
    )

    print()
    print("=" * 10, "Test Result", "=" * 10)
    print_metrics(test_metrics, prefix="")

    print()
    print("elapsed training done.")
    print("best checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
