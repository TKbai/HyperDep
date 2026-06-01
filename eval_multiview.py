import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

import option
from model import Model
from dataset_dvlog import DVlogDataset
from eval_dvlog import compute_binary_metrics, print_metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="dvlog-dataset")
    parser.add_argument("--stats-path", default="dvlog_stats.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fusion", default="concat_proj")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--pooling", default="topk_mean")
    parser.add_argument("--pool-alpha", type=float, default=0.3)
    parser.add_argument("--topk-divisor", type=int, default=16)

    return parser.parse_args()


def build_model_args(cli_args):
    args = option.parser.parse_args([
        "--data-root", cli_args.data_root,
        "--stats-path", cli_args.stats_path,
        "--max-seqlen", str(cli_args.window_size),
        "--batch-size", str(cli_args.batch_size),
        "--fusion", cli_args.fusion,
        "--pooling", cli_args.pooling,
        "--pool-alpha", str(cli_args.pool_alpha),
        "--topk-divisor", str(cli_args.topk_divisor),
        "--cuda", str(cli_args.cuda),
    ])

    args.device = "cuda:" + str(args.cuda) if torch.cuda.is_available() and args.cuda >= 0 else "cpu"

    return args


def load_model(args, checkpoint_path):
    try:
        ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=args.device)

    model = Model(args).to(args.device)

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("loaded checkpoint:", checkpoint_path)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    model.eval()
    return model


def load_stats(stats_path):
    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    return mean, std


def make_windows(feat, window_size=200, num_windows=5):
    """
    feat: [T, C]
    return:
        windows: [N, window_size, C]
        seq_lens: [N]
    """

    T, C = feat.shape

    if T <= window_size:
        out = np.zeros((1, window_size, C), dtype=np.float32)
        out[0, :T] = feat
        seq_lens = np.array([max(T, 1)], dtype=np.int64)
        return out, seq_lens

    max_start = T - window_size

    starts = np.linspace(0, max_start, num_windows)
    starts = np.round(starts).astype(np.int64)
    starts = np.unique(starts)

    windows = []
    seq_lens = []

    for s in starts:
        w = feat[s:s + window_size]
        windows.append(w)
        seq_lens.append(window_size)

    windows = np.stack(windows, axis=0).astype(np.float32)
    seq_lens = np.asarray(seq_lens, dtype=np.int64)

    return windows, seq_lens


@torch.no_grad()
def predict_windows(model, args, windows, seq_lens, batch_size=16):
    probs = []

    for start in range(0, len(windows), batch_size):
        end = start + batch_size

        x = torch.from_numpy(windows[start:end]).float().to(args.device)
        l = torch.from_numpy(seq_lens[start:end]).long().to(args.device)

        video_prob, _ = model(x, l)
        probs.extend(video_prob.view(-1).detach().cpu().numpy().tolist())

    return np.asarray(probs, dtype=np.float64)


def aggregate_window_probs(window_probs, mode):
    window_probs = np.asarray(window_probs, dtype=np.float64)

    if mode == "mean":
        return float(window_probs.mean())

    if mode == "max":
        return float(window_probs.max())

    if mode == "median":
        return float(np.median(window_probs))

    if mode == "top2_mean":
        k = min(2, len(window_probs))
        return float(np.sort(window_probs)[-k:].mean())

    if mode == "top3_mean":
        k = min(3, len(window_probs))
        return float(np.sort(window_probs)[-k:].mean())

    raise ValueError(f"Unknown aggregation mode: {mode}")


def predict_dataset_multiview(dataset, model, args, mean, std, num_windows, agg_mode):
    y_true = []
    y_prob = []

    for sid, visual_path, acoustic_path, label in dataset.samples:
        visual = np.load(visual_path).astype(np.float32)
        acoustic = np.load(acoustic_path).astype(np.float32)

        t = min(visual.shape[0], acoustic.shape[0])
        visual = visual[:t]
        acoustic = acoustic[:t]

        feat = np.concatenate([visual, acoustic], axis=1).astype(np.float32)
        feat = (feat - mean) / std

        windows, seq_lens = make_windows(
            feat,
            window_size=args.max_seqlen,
            num_windows=num_windows,
        )

        window_probs = predict_windows(
            model=model,
            args=args,
            windows=windows,
            seq_lens=seq_lens,
            batch_size=args.batch_size,
        )

        prob = aggregate_window_probs(window_probs, agg_mode)

        y_true.append(int(label))
        y_prob.append(prob)

    return np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float64)


def make_thresholds(y_prob):
    dense_low = np.linspace(0.000001, 0.0500, 300)
    normal = np.linspace(0.055, 0.950, 180)

    unique_probs = np.unique(y_prob)
    if len(unique_probs) >= 2:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
    else:
        midpoints = unique_probs

    thresholds = np.concatenate([dense_low, normal, unique_probs, midpoints])
    thresholds = np.unique(np.clip(thresholds, 1e-8, 0.999999))

    return thresholds


def search_threshold(y_true, y_prob, metric="f1"):
    best = {
        "score": -1.0,
        "threshold": 0.5,
        "metrics": None,
    }

    for threshold in make_thresholds(y_prob):
        metrics = compute_binary_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
        )

        if metric not in metrics:
            raise KeyError(f"Metric {metric} not found in metrics dict.")

        score = metrics[metric]

        if np.isnan(score):
            continue

        if score > best["score"]:
            best["score"] = score
            best["threshold"] = float(threshold)
            best["metrics"] = metrics

    return best


def main():
    cli_args = parse_args()
    args = build_model_args(cli_args)

    print("========== Multi-window Evaluation ==========")
    print("checkpoint :", cli_args.checkpoint)
    print("fusion     :", cli_args.fusion)
    print("window_size:", cli_args.window_size)
    print("num_windows:", cli_args.num_windows)
    print("metric     :", cli_args.metric)
    print("device     :", args.device)

    model = load_model(args, cli_args.checkpoint)
    mean, std = load_stats(cli_args.stats_path)

    val_data = DVlogDataset(
        root=args.data_root,
        fold=args.val_fold,
        gender=args.gender,
        max_seqlen=args.max_seqlen,
        random_crop=False,
        stats_path=args.stats_path,
    )

    test_data = DVlogDataset(
        root=args.data_root,
        fold=args.test_fold,
        gender=args.gender,
        max_seqlen=args.max_seqlen,
        random_crop=False,
        stats_path=args.stats_path,
    )

    agg_modes = ["mean", "max", "median", "top2_mean", "top3_mean"]

    best_overall = None

    for agg_mode in agg_modes:
        print()
        print("=" * 80)
        print("Aggregation:", agg_mode)

        y_val, p_val = predict_dataset_multiview(
            dataset=val_data,
            model=model,
            args=args,
            mean=mean,
            std=std,
            num_windows=cli_args.num_windows,
            agg_mode=agg_mode,
        )

        best = search_threshold(
            y_true=y_val,
            y_prob=p_val,
            metric=cli_args.metric,
        )

        print("best val threshold:", best["threshold"])
        print("best val score:", best["score"])
        print_metrics(best["metrics"], prefix="VALID:")

        if best_overall is None or best["score"] > best_overall["score"]:
            best_overall = {
                "agg_mode": agg_mode,
                "threshold": best["threshold"],
                "score": best["score"],
                "val_metrics": best["metrics"],
            }

    print()
    print("=" * 80)
    print("Best validation aggregation:", best_overall["agg_mode"])
    print("Best validation threshold:", best_overall["threshold"])
    print("Best validation score:", best_overall["score"])

    y_test, p_test = predict_dataset_multiview(
        dataset=test_data,
        model=model,
        args=args,
        mean=mean,
        std=std,
        num_windows=cli_args.num_windows,
        agg_mode=best_overall["agg_mode"],
    )

    test_metrics = compute_binary_metrics(
        y_true=y_test,
        y_prob=p_test,
        threshold=best_overall["threshold"],
    )

    print_metrics(best_overall["val_metrics"], prefix="\nBEST VALID:")
    print_metrics(test_metrics, prefix="\nTEST with best validation aggregation/threshold:")

    print()
    print("multi-window evaluation finished.")


if __name__ == "__main__":
    main()