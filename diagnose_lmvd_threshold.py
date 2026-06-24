import numpy as np
import torch
import option

from torch.utils.data import DataLoader
from dataset_lmvd import LMVDDataset
from model import Model
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


def get_metric(metrics, name):
    if name in metrics:
        return metrics[name]

    if name == "f1" and "f1_weighted" in metrics:
        return metrics["f1_weighted"]

    raise KeyError(f"{name} not found. Available: {list(metrics.keys())}")


def search_threshold(
    y_true,
    y_prob,
    optimize="f1_pos",
    min_precision=None,
    min_recall=None,
    min_acc=None,
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

        if min_precision is not None and metrics["precision_pos"] < min_precision:
            continue

        if min_recall is not None and metrics["recall_pos"] < min_recall:
            continue

        if min_acc is not None and metrics["acc"] < min_acc:
            continue

        score = get_metric(metrics, optimize)

        if np.isnan(score):
            continue

        if score > best["score"]:
            best["score"] = float(score)
            best["threshold"] = float(threshold)
            best["metrics"] = metrics

    return best


def build_args():
    args = option.parser.parse_args([
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
    ])

    args.device = "cuda:0" if torch.cuda.is_available() and int(args.cuda) >= 0 else "cpu"

    args.manifest_path = "LMVD/processed_811/manifest.csv"
    args.stats_path = "LMVD/processed_811/lmvd_stats.npz"
    args.lmvd_norm_clip = 10.0
    args.val_fold = "valid"
    args.test_fold = "test"

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

    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return loader


def load_model(args, ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=args.device)

    model = Model(args).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    print("loaded:", ckpt_path)
    print("ckpt epoch:", ckpt.get("epoch", "unknown"))
    print("ckpt best_score:", ckpt.get("best_score", "unknown"))
    print("ckpt stored_threshold:", ckpt.get("best_threshold", "unknown"))

    return model


def print_one_line(seed, setting_name, threshold, m):
    print(
        f"{seed:>6s} | "
        f"{setting_name:45s} | "
        f"thr={threshold:9.6g} | "
        f"acc={m['acc']:.4f} | "
        f"prec+={m['precision_pos']:.4f} | "
        f"rec+={m['recall_pos']:.4f} | "
        f"f1+={m['f1_pos']:.4f} | "
        f"auc={m['auc']:.4f} | "
        f"auprc={m['auprc']:.4f} | "
        f"fp/fn={m['fp']}/{m['fn']}"
    )


def main():
    args = build_args()

    ckpts = {
        "seed9": "./ckpt_lmvd/hypervd_lmvd_seq300_detour_a05_selauprc_thrf1pos_seed9_best.pkl",
        "seed42": "./ckpt_lmvd/hypervd_lmvd_seq300_detour_a05_selauprc_thrf1pos_seed42_best.pkl",
        "seed2024": "./ckpt_lmvd/hypervd_lmvd_seq300_detour_a05_selauprc_thrf1pos_seed2024_best.pkl",
    }

    settings = [
        {
            "name": "optimize f1_pos",
            "optimize": "f1_pos",
        },
        {
            "name": "optimize weighted_f1",
            "optimize": "f1_weighted",
        },
        {
            "name": "optimize acc",
            "optimize": "acc",
        },
        {
            "name": "precision>=0.70 optimize f1_pos",
            "optimize": "f1_pos",
            "min_precision": 0.70,
        },
        {
            "name": "precision>=0.72 optimize f1_pos",
            "optimize": "f1_pos",
            "min_precision": 0.72,
        },
        {
            "name": "recall>=0.78 precision>=0.70 optimize f1_pos",
            "optimize": "f1_pos",
            "min_recall": 0.78,
            "min_precision": 0.70,
        },
        {
            "name": "recall>=0.80 optimize f1_pos",
            "optimize": "f1_pos",
            "min_recall": 0.80,
        },
    ]

    val_loader = build_loader(args, args.val_fold)
    test_loader = build_loader(args, args.test_fold)

    print("=" * 120)
    print("LMVD threshold diagnosis")
    print("=" * 120)

    all_rows = []

    for seed, ckpt_path in ckpts.items():
        print()
        print("#" * 120)
        print(seed)
        print("#" * 120)

        model = load_model(args, ckpt_path)

        y_val, p_val = collect_dvlog_outputs(
            dataloader=val_loader,
            model=model,
            args=args,
        )

        y_test, p_test = collect_dvlog_outputs(
            dataloader=test_loader,
            model=model,
            args=args,
        )

        for setting in settings:
            best = search_threshold(
                y_true=y_val,
                y_prob=p_val,
                optimize=setting["optimize"],
                min_precision=setting.get("min_precision"),
                min_recall=setting.get("min_recall"),
                min_acc=setting.get("min_acc"),
            )

            if best["metrics"] is None:
                print(seed, setting["name"], "No valid threshold found.")
                continue

            test_metrics = compute_binary_metrics(
                y_true=y_test,
                y_prob=p_test,
                threshold=best["threshold"],
            )

            print_one_line(
                seed=seed,
                setting_name=setting["name"],
                threshold=best["threshold"],
                m=test_metrics,
            )

            all_rows.append(
                {
                    "seed": seed,
                    "setting": setting["name"],
                    "threshold": best["threshold"],
                    "acc": test_metrics["acc"],
                    "precision_pos": test_metrics["precision_pos"],
                    "recall_pos": test_metrics["recall_pos"],
                    "f1_pos": test_metrics["f1_pos"],
                    "auc": test_metrics["auc"],
                    "auprc": test_metrics["auprc"],
                    "fp": test_metrics["fp"],
                    "fn": test_metrics["fn"],
                }
            )

    print()
    print("=" * 120)
    print("Mean by setting")
    print("=" * 120)

    setting_names = []
    for row in all_rows:
        if row["setting"] not in setting_names:
            setting_names.append(row["setting"])

    for setting_name in setting_names:
        rows = [r for r in all_rows if r["setting"] == setting_name]

        if len(rows) == 0:
            continue

        print()
        print(setting_name)
        for key in ["acc", "precision_pos", "recall_pos", "f1_pos", "auc", "auprc"]:
            vals = np.asarray([r[key] for r in rows], dtype=np.float64)
            print(f"  {key:14s}: {vals.mean():.4f} ± {vals.std(ddof=1) if len(vals)>1 else 0.0:.4f}")

    print()
    print("diagnosis finished.")


if __name__ == "__main__":
    main()
