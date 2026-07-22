import argparse
import os
import numpy as np
import torch

from torch.utils.data import DataLoader

from model import Model
from dataset_lmvd import LMVDDataset
from eval_dvlog import collect_dvlog_outputs, compute_binary_metrics


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

    thresholds = np.concatenate(
        [
            dense_tiny,
            dense_low,
            normal,
            unique_probs,
            midpoints,
        ]
    )

    return np.unique(np.clip(thresholds, 1e-10, 0.999999))


def metric_value(metrics, name):
    if name in metrics:
        return metrics[name]

    if name == "f1" and "f1_weighted" in metrics:
        return metrics["f1_weighted"]

    raise KeyError(
        f"Metric {name} not found. Available keys: {list(metrics.keys())}"
    )


def best_threshold(y_true, y_prob, metric):
    best_t = 0.5
    best_score = -1.0
    best_metrics = None

    for t in make_thresholds(y_prob):
        m = compute_binary_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(t),
        )

        s = metric_value(m, metric)

        if np.isnan(s):
            continue

        if s > best_score:
            best_score = float(s)
            best_t = float(t)
            best_metrics = m

    return best_t, best_score, best_metrics


def build_loader(args, fold, batch_size=16):
    ds = LMVDDataset(
        manifest_path=args.manifest_path,
        fold=fold,
        max_seqlen=args.max_seqlen,
        random_crop=False,
        stats_path=args.stats_path,
        norm_clip=args.lmvd_norm_clip,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=str(args.device).startswith("cuda"),
        drop_last=False,
    )


def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    args = argparse.Namespace(**ckpt["args"])
    args.device = device

    # Backward compatibility for old checkpoints.
    if not hasattr(args, "temporal_gamma"):
        args.temporal_gamma = 2.718281828459045

    if not hasattr(args, "workers"):
        args.workers = 0

    model = Model(args).to(device)

    result = model.load_state_dict(
        ckpt["model_state_dict"],
        strict=False,
    )

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    model.eval()

    return model, args, ckpt, missing, unexpected


def eval_one_checkpoint(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print("=" * 120)
        print("MISSING:", ckpt_path)
        print("=" * 120)
        return None

    model, args, ckpt, missing, unexpected = load_model_from_ckpt(
        ckpt_path,
        device,
    )

    valid_loader = build_loader(args, args.val_fold, batch_size=16)
    test_loader = build_loader(args, args.test_fold, batch_size=16)

    yv, pv = collect_dvlog_outputs(
        dataloader=valid_loader,
        model=model,
        args=args,
    )

    yt, pt = collect_dvlog_outputs(
        dataloader=test_loader,
        model=model,
        args=args,
    )

    print("=" * 120)
    print("checkpoint:", ckpt_path)
    print("manifold:", getattr(args, "manifold", "unknown"))
    print("feature_adj_refiner:", getattr(args, "feature_adj_refiner", "none"))
    print("temporal_gamma:", getattr(args, "temporal_gamma", "missing"))
    print("selection_metric:", ckpt.get("selection_metric", "unknown"))
    print("saved threshold_metric:", ckpt.get("threshold_metric", "unknown"))
    print("saved best_threshold:", ckpt.get("best_threshold", "unknown"))
    print("saved best_score:", ckpt.get("best_score", "unknown"))
    print("epoch:", ckpt.get("epoch", "unknown"))
    print("missing_keys:", missing)
    print("unexpected_keys:", unexpected)
    print("=" * 120)

    if len(unexpected) > 0:
        print("WARNING: unexpected keys exist. Check model/ckpt mismatch.")
    if len(missing) > 0:
        print("WARNING: missing keys exist. Check model/ckpt mismatch.")

    rows = []

    # First: official saved validation threshold.
    saved_t = float(ckpt.get("best_threshold", 0.5))
    saved_m = compute_binary_metrics(
        y_true=yt,
        y_prob=pt,
        threshold=saved_t,
    )

    rows.append(
        {
            "threshold_metric": "saved_ckpt",
            "val_t": saved_t,
            "val_score": float("nan"),
            "test_acc": saved_m["acc"],
            "test_f1pos": saved_m["f1_pos"],
            "test_f1macro": saved_m["f1_macro"],
            "test_auc": saved_m["auc"],
            "test_auprc": saved_m["auprc"],
            "test_spec": saved_m["specificity"],
            "test_rec": saved_m["recall_pos"],
            "tn": saved_m["tn"],
            "fp": saved_m["fp"],
            "fn": saved_m["fn"],
            "tp": saved_m["tp"],
        }
    )

    # Then: recompute validation thresholds using different threshold metrics.
    for metric in [
        "f1_pos",
        "f1_macro",
        "balanced_acc",
        "f1_weighted",
        "acc",
    ]:
        val_t, val_score, _ = best_threshold(
            y_true=yv,
            y_prob=pv,
            metric=metric,
        )

        test_m = compute_binary_metrics(
            y_true=yt,
            y_prob=pt,
            threshold=val_t,
        )

        rows.append(
            {
                "threshold_metric": metric,
                "val_t": val_t,
                "val_score": val_score,
                "test_acc": test_m["acc"],
                "test_f1pos": test_m["f1_pos"],
                "test_f1macro": test_m["f1_macro"],
                "test_auc": test_m["auc"],
                "test_auprc": test_m["auprc"],
                "test_spec": test_m["specificity"],
                "test_rec": test_m["recall_pos"],
                "tn": test_m["tn"],
                "fp": test_m["fp"],
                "fn": test_m["fn"],
                "tp": test_m["tp"],
            }
        )

    print()
    print(
        f"{'threshold_metric':18s} "
        f"{'val_t':>9s} "
        f"{'val_score':>10s} "
        f"{'acc':>8s} "
        f"{'f1pos':>8s} "
        f"{'f1mac':>8s} "
        f"{'auc':>8s} "
        f"{'auprc':>8s} "
        f"{'spec':>8s} "
        f"{'rec':>8s} "
        f"{'tn':>4s} "
        f"{'fp':>4s} "
        f"{'fn':>4s} "
        f"{'tp':>4s}"
    )
    print("-" * 130)

    for r in rows:
        print(
            f"{r['threshold_metric']:18s} "
            f"{r['val_t']:9.4f} "
            f"{r['val_score']:10.4f} "
            f"{r['test_acc']:8.4f} "
            f"{r['test_f1pos']:8.4f} "
            f"{r['test_f1macro']:8.4f} "
            f"{r['test_auc']:8.4f} "
            f"{r['test_auprc']:8.4f} "
            f"{r['test_spec']:8.4f} "
            f"{r['test_rec']:8.4f} "
            f"{r['tn']:4d} "
            f"{r['fp']:4d} "
            f"{r['fn']:4d} "
            f"{r['tp']:4d}"
        )

    print()

    # Return compact rows for summary table.
    compact = []

    for r in rows:
        if r["threshold_metric"] in ["saved_ckpt", "f1_macro", "balanced_acc"]:
            compact.append(
                {
                    "ckpt": os.path.basename(ckpt_path),
                    "manifold": getattr(args, "manifold", "unknown"),
                    "refiner": getattr(args, "feature_adj_refiner", "none"),
                    **r,
                }
            )

    return compact


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ckpts = [
        # Main clean comparisons.
        "./ckpt_lmvd/hypervd_lmvd_seq300_detour_a05_euclidean_maskfix_gammaE_seed9_best.pkl",
        "./ckpt_lmvd/hypervd_lmvd_seq300_detour_a05_selauprc_thrf1pos_maskfix_gammaE_seed9_best.pkl",
        "./ckpt_lmvd/hypervd_lmvd_seq300_detour_a05_edgegatv2_maskfix_gammaE_seed9_best.pkl",

        # Diagnostic / legacy reference.
        "./ckpt_lmvd/hypervd_lmvd_seq300_detour_a05_edgegatv2_seed9_best.pkl",
    ]

    all_compact = []

    for ckpt_path in ckpts:
        compact = eval_one_checkpoint(ckpt_path, device)
        if compact is not None:
            all_compact.extend(compact)

    print()
    print("#" * 120)
    print("COMPACT SUMMARY: saved_ckpt / f1_macro / balanced_acc only")
    print("#" * 120)

    print(
        f"{'model':64s} "
        f"{'manifold':>10s} "
        f"{'refiner':>8s} "
        f"{'thr_metric':>13s} "
        f"{'val_t':>8s} "
        f"{'acc':>7s} "
        f"{'f1pos':>7s} "
        f"{'auc':>7s} "
        f"{'auprc':>7s} "
        f"{'spec':>7s} "
        f"{'rec':>7s}"
    )
    print("-" * 150)

    for r in all_compact:
        print(
            f"{r['ckpt'][:64]:64s} "
            f"{r['manifold']:>10s} "
            f"{r['refiner']:>8s} "
            f"{r['threshold_metric']:>13s} "
            f"{r['val_t']:8.4f} "
            f"{r['test_acc']:7.4f} "
            f"{r['test_f1pos']:7.4f} "
            f"{r['test_auc']:7.4f} "
            f"{r['test_auprc']:7.4f} "
            f"{r['test_spec']:7.4f} "
            f"{r['test_rec']:7.4f}"
        )


if __name__ == "__main__":
    main()