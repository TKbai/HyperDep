import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

import option
from model import Model
from dataset_dvlog_mw import DVlogMultiWindowDataset
from eval_dvlog import collect_dvlog_outputs, compute_binary_metrics, print_metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="dvlog-dataset")
    parser.add_argument("--stats-path", default="dvlog_stats.npz")
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--fusion", default="concat_proj")
    parser.add_argument("--max-seqlen", type=int, default=200)
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument(
        "--window-agg",
        default="mean",
        choices=["mean", "max", "top2_mean", "top3_mean"],
    )

    parser.add_argument(
        "--pooling",
        default="topk_mean",
        choices=["topk", "mean", "topk_mean"],
    )
    parser.add_argument("--pool-alpha", type=float, default=0.3)
    parser.add_argument("--topk-divisor", type=int, default=16)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def build_model_args(cli_args):
    args = option.parser.parse_args([
        "--data-root", cli_args.data_root,
        "--stats-path", cli_args.stats_path,
        "--max-seqlen", str(cli_args.max_seqlen),
        "--batch-size", str(cli_args.batch_size),
        "--fusion", cli_args.fusion,
        "--pooling", cli_args.pooling,
        "--pool-alpha", str(cli_args.pool_alpha),
        "--topk-divisor", str(cli_args.topk_divisor),
        "--cuda", str(cli_args.cuda),
    ])

    args.device = (
        "cuda:" + str(args.cuda)
        if torch.cuda.is_available() and int(args.cuda) >= 0
        else "cpu"
    )

    args.eval_num_windows = cli_args.num_windows
    args.window_agg = cli_args.window_agg

    return args


def build_loader(args, fold, num_windows, num_workers):
    dataset = DVlogMultiWindowDataset(
        root=args.data_root,
        fold=fold,
        gender=args.gender,
        max_seqlen=args.max_seqlen,
        num_windows=num_windows,
        random_windows=False,
        stats_path=args.stats_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataset, loader


def load_model(args, checkpoint_path):
    try:
        ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=args.device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("checkpoint epoch:", ckpt.get("epoch", "unknown"))
        print("checkpoint best_score:", ckpt.get("best_score", "unknown"))
        print("checkpoint best_threshold:", ckpt.get("best_threshold", "unknown"))
    else:
        state_dict = ckpt

    model = Model(args).to(args.device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("loaded checkpoint:", checkpoint_path)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("first missing keys:", missing[:5])

    if len(unexpected) > 0:
        print("first unexpected keys:", unexpected[:5])

    model.eval()
    return model


def make_thresholds(y_prob):
    y_prob = np.asarray(y_prob, dtype=np.float64)

    dense_tiny = np.linspace(1e-8, 1e-5, 200)
    dense_low = np.linspace(1e-5, 0.0500, 500)
    normal = np.linspace(0.055, 0.950, 180)

    unique_probs = np.unique(y_prob)

    if len(unique_probs) >= 2:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
    else:
        midpoints = unique_probs

    thresholds = np.concatenate([
        dense_tiny,
        dense_low,
        normal,
        unique_probs,
        midpoints,
    ])

    thresholds = np.unique(np.clip(thresholds, 1e-10, 0.999999))

    return thresholds


def search_threshold(
    y_true,
    y_prob,
    optimize="f1_pos",
    min_recall=None,
    min_specificity=None,
    min_precision=None,
):
    best = {
        "score": -1.0,
        "threshold": None,
        "metrics": None,
    }

    for threshold in make_thresholds(y_prob):
        metrics = compute_binary_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
        )

        if min_recall is not None and metrics["recall_pos"] < min_recall:
            continue

        if min_specificity is not None and metrics["specificity"] < min_specificity:
            continue

        if min_precision is not None and metrics["precision_pos"] < min_precision:
            continue

        if optimize not in metrics:
            raise KeyError(
                f"Metric {optimize} not found. Available metrics: {list(metrics.keys())}"
            )

        score = metrics[optimize]

        if np.isnan(score):
            continue

        if score > best["score"]:
            best["score"] = score
            best["threshold"] = float(threshold)
            best["metrics"] = metrics

    return best


def run_setting(setting, y_val, p_val, y_test, p_test):
    print()
    print("=" * 100)
    print("setting:", setting["name"])
    print("=" * 100)

    best = search_threshold(
        y_true=y_val,
        y_prob=p_val,
        optimize=setting["optimize"],
        min_recall=setting.get("min_recall"),
        min_specificity=setting.get("min_specificity"),
        min_precision=setting.get("min_precision"),
    )

    if best["metrics"] is None:
        print("No valid threshold found.")
        return None

    test_metrics = compute_binary_metrics(
        y_true=y_test,
        y_prob=p_test,
        threshold=best["threshold"],
    )

    print("best val threshold:", best["threshold"])
    print("best val score    :", best["score"])

    print_metrics(best["metrics"], prefix="\nVALID:")
    print_metrics(test_metrics, prefix="\nTEST:")

    return {
        "name": setting["name"],
        "threshold": best["threshold"],
        "val_metrics": best["metrics"],
        "test_metrics": test_metrics,
    }


def main():
    cli_args = parse_args()
    args = build_model_args(cli_args)

    print("========== Multi-window threshold diagnosis ==========")
    print("checkpoint   :", cli_args.checkpoint)
    print("fusion       :", cli_args.fusion)
    print("max_seqlen   :", cli_args.max_seqlen)
    print("num_windows  :", cli_args.num_windows)
    print("window_agg   :", cli_args.window_agg)
    print("pooling      :", cli_args.pooling)
    print("pool_alpha   :", cli_args.pool_alpha)
    print("topk_divisor :", cli_args.topk_divisor)
    print("batch_size   :", cli_args.batch_size)
    print("device       :", args.device)
    print("======================================================")

    model = load_model(args, cli_args.checkpoint)

    val_dataset, val_loader = build_loader(
        args=args,
        fold=args.val_fold,
        num_windows=cli_args.num_windows,
        num_workers=cli_args.num_workers,
    )

    test_dataset, test_loader = build_loader(
        args=args,
        fold=args.test_fold,
        num_windows=cli_args.num_windows,
        num_workers=cli_args.num_workers,
    )

    print()
    print("val samples :", len(val_dataset))
    print("test samples:", len(test_dataset))

    print()
    print("Collecting validation predictions...")
    y_val, p_val = collect_dvlog_outputs(
        dataloader=val_loader,
        model=model,
        args=args,
    )

    print("Collecting test predictions...")
    y_test, p_test = collect_dvlog_outputs(
        dataloader=test_loader,
        model=model,
        args=args,
    )

    print()
    print("Validation positive ratio:", float(np.mean(y_val)))
    print("Test positive ratio      :", float(np.mean(y_test)))

    print()
    print("Validation probability range:")
    print("min   :", float(np.min(p_val)))
    print("max   :", float(np.max(p_val)))
    print("mean  :", float(np.mean(p_val)))
    print("median:", float(np.median(p_val)))

    print()
    print("Test probability range:")
    print("min   :", float(np.min(p_test)))
    print("max   :", float(np.max(p_test)))
    print("mean  :", float(np.mean(p_test)))
    print("median:", float(np.median(p_test)))

    settings = [
        {
            "name": "1) optimize weighted F1",
            "optimize": "f1",
        },
        {
            "name": "2) optimize positive F1",
            "optimize": "f1_pos",
        },
        {
            "name": "3) recall_pos >= 0.80, optimize positive F1",
            "optimize": "f1_pos",
            "min_recall": 0.80,
        },
        {
            "name": "4) specificity >= 0.50, optimize positive F1",
            "optimize": "f1_pos",
            "min_specificity": 0.50,
        },
        {
            "name": "5) recall_pos >= 0.80 and specificity >= 0.50, optimize positive F1",
            "optimize": "f1_pos",
            "min_recall": 0.80,
            "min_specificity": 0.50,
        },
        {
            "name": "6) recall_pos >= 0.80, optimize specificity",
            "optimize": "specificity",
            "min_recall": 0.80,
        },
        {
            "name": "7) precision_pos >= 0.70, optimize positive F1",
            "optimize": "f1_pos",
            "min_precision": 0.70,
        },
        {
            "name": "8) recall_pos >= 0.80 and precision_pos >= 0.68, optimize positive F1",
            "optimize": "f1_pos",
            "min_recall": 0.80,
            "min_precision": 0.68,
        },
    ]

    all_results = []

    for setting in settings:
        result = run_setting(
            setting=setting,
            y_val=y_val,
            p_val=p_val,
            y_test=y_test,
            p_test=p_test,
        )

        if result is not None:
            all_results.append(result)

    print()
    print("=" * 100)
    print("SUMMARY ON TEST")
    print("=" * 100)

    print(
        f"{'Setting':65s} "
        f"{'Thr':>10s} "
        f"{'Acc':>8s} "
        f"{'F1+':>8s} "
        f"{'Prec+':>8s} "
        f"{'Rec+':>8s} "
        f"{'Spec':>8s} "
        f"{'AUC':>8s} "
        f"{'AUPRC':>8s} "
        f"{'FP/FN':>10s}"
    )

    for r in all_results:
        m = r["test_metrics"]
        name = r["name"][:65]
        fp_fn = f"{m['fp']}/{m['fn']}"

        print(
            f"{name:65s} "
            f"{r['threshold']:10.6g} "
            f"{m['acc']:8.4f} "
            f"{m['f1_pos']:8.4f} "
            f"{m['precision_pos']:8.4f} "
            f"{m['recall_pos']:8.4f} "
            f"{m['specificity']:8.4f} "
            f"{m['auc']:8.4f} "
            f"{m['auprc']:8.4f} "
            f"{fp_fn:>10s}"
        )

    print()
    print("multi-window threshold diagnosis finished.")


if __name__ == "__main__":
    main()
