import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def _get_device(args):
    if hasattr(args, "device"):
        return args.device
    return "cuda:0" if torch.cuda.is_available() and int(args.cuda) >= 0 else "cpu"


@torch.no_grad()
def collect_dvlog_outputs(dataloader, model, args, max_batches=None):
    """
    Collect video-level probabilities and labels.

    Supports:
        inputs: [B, T, C]
        inputs: [B, K, T, C]
    """

    from window_utils import aggregate_window_probs_tensor

    device = _get_device(args)
    model.eval()

    probs_all = []
    labels_all = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        labels = labels.float().view(-1).to(device, non_blocking=True)

        # ==================================================
        # Case 1: standard input [B, T, C]
        # ==================================================
        if inputs.dim() == 3:
            seq_len = torch.sum(
                torch.max(torch.abs(inputs), dim=2)[0] > 0,
                dim=1,
            )
            seq_len = torch.clamp(seq_len, min=1)

            max_len = int(seq_len.max().item())
            max_len = max(1, max_len)

            inputs = inputs[:, :max_len, :]

            inputs = inputs.float().to(device, non_blocking=True)
            seq_len = seq_len.to(device)

            video_prob, frame_prob = model(inputs, seq_len)
            video_prob = video_prob.view(-1)

        # ==================================================
        # Case 2: multi-window input [B, K, T, C]
        # ==================================================
        elif inputs.dim() == 4:
            b, k, t, c = inputs.shape

            inputs = inputs.view(b * k, t, c)

            seq_len = torch.sum(
                torch.max(torch.abs(inputs), dim=2)[0] > 0,
                dim=1,
            )
            seq_len = torch.clamp(seq_len, min=1)

            max_len = int(seq_len.max().item())
            max_len = max(1, max_len)

            inputs = inputs[:, :max_len, :]

            inputs = inputs.float().to(device, non_blocking=True)
            seq_len = seq_len.to(device)

            window_agg = getattr(args, "window_agg", "mean")

            if window_agg in ["learn_attn", "learn_attn_conf"]:
                window_prob, frame_prob, window_emb = model(
                    inputs,
                    seq_len,
                    return_embedding=True,
                )
            else:
                window_prob, frame_prob = model(inputs, seq_len)
                window_emb = None

            window_prob = window_prob.view(b, k)

            if window_emb is not None:
                window_emb = window_emb.view(b, k, -1)

            if window_agg == "learn_attn_conf":
                attn_layer = getattr(model, "window_attn_conf", None)
            else:
                attn_layer = getattr(model, "window_attn", None)

            video_prob = aggregate_window_probs_tensor(
                window_prob,
                mode=window_agg,
                window_emb=window_emb,
                attn_layer=attn_layer,
                attn_temperature=getattr(args, "window_attn_temperature", 1.0),
                attn_logit_bias=getattr(args, "window_attn_logit_bias", 0.5),
            )

        else:
            raise ValueError(
                f"Expected inputs shape [B,T,C] or [B,K,T,C], got {inputs.shape}"
            )

        if torch.isnan(video_prob).any() or torch.isinf(video_prob).any():
            raise FloatingPointError(
                "NaN or Inf detected in model output during evaluation."
            )

        probs_all.extend(video_prob.detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())

    y_prob = np.asarray(probs_all, dtype=np.float64)
    y_true = np.asarray(labels_all, dtype=np.int64)

    return y_true, y_prob


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute D-Vlog binary classification metrics.

    label:
        0 = non-depression
        1 = depression
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    # Confusion matrix: rows = true, cols = pred
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / max(tn + fp, 1)
    sensitivity = recall_pos

    # AUC / AUPRC need both classes to exist.
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        auprc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

    metrics = {
        # aliases for later model selection
        "acc": acc,
        "f1": f1_w,
        "auc": auc,
        "auprc": auprc,

        # detailed metrics
        "balanced_acc": balanced_acc,
        "precision": precision_w,
        "recall": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


@torch.no_grad()
def evaluate_dvlog(
    dataloader,
    model,
    args,
    threshold=0.5,
    max_batches=None,
):
    """
    Full evaluation function.

    Returns:
        metrics dict
    """

    y_true, y_prob = collect_dvlog_outputs(
        dataloader=dataloader,
        model=model,
        args=args,
        max_batches=max_batches,
    )

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )

    return metrics


@torch.no_grad()
@torch.no_grad()
def find_best_threshold(
    dataloader,
    model,
    args,
    metric="f1",
    max_batches=None,
):
    """
    Search the best threshold on validation set.

    This version is adapted for HyperVD-DVlog because the model
    probabilities are often concentrated in a very small range,
    e.g. 0.000 ~ 0.100.

    metric choices:
        "f1"           -> weighted F1
        "acc"          -> accuracy
        "balanced_acc" -> balanced accuracy
        "f1_pos"       -> positive-class F1 for depression class
    """

    y_true, y_prob = collect_dvlog_outputs(
        dataloader=dataloader,
        model=model,
        args=args,
        max_batches=max_batches,
    )

    # ------------------------------------------------------
    # 1. Dense low-threshold grid
    # Important because current HyperVD-DVlog probabilities
    # are mostly far below 0.05.
    # ------------------------------------------------------
    dense_low = np.linspace(0.0001, 0.0500, 200)

    # ------------------------------------------------------
    # 2. Normal threshold grid
    # Keep this for later models whose probabilities become
    # better calibrated.
    # ------------------------------------------------------
    normal_grid = np.linspace(0.055, 0.950, 180)

    # ------------------------------------------------------
    # 3. Data-driven thresholds
    # Add model's actual probability values and midpoints.
    # This avoids missing the true best threshold when the
    # validation set is small.
    # ------------------------------------------------------
    unique_probs = np.unique(y_prob)

    if len(unique_probs) >= 2:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
    else:
        midpoints = unique_probs

    thresholds = np.concatenate(
        [
            dense_low,
            normal_grid,
            unique_probs,
            midpoints,
        ]
    )

    thresholds = np.unique(np.clip(thresholds, 1e-6, 0.999999))

    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for threshold in thresholds:
        metrics = compute_binary_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
        )

        if metric not in metrics:
            raise KeyError(
                f"Metric {metric} not found. Available metrics: {list(metrics.keys())}"
            )

        score = metrics[metric]

        if np.isnan(score):
            continue

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_score, best_metrics


def print_metrics(metrics, prefix=""):
    """
    Pretty print metrics.
    """

    if prefix:
        print(prefix)

    keys = [
        "threshold",
        "acc",
        "balanced_acc",
        "precision",
        "recall",
        "f1_weighted",
        "f1_macro",
        "precision_pos",
        "recall_pos",
        "f1_pos",
        "auc",
        "auprc",
        "sensitivity",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    for k in keys:
        if k not in metrics:
            continue

        v = metrics[k]

        if isinstance(v, float):
            if np.isnan(v):
                print(f"{k:16s}: nan")
            else:
                print(f"{k:16s}: {v:.4f}")
        else:
            print(f"{k:16s}: {v}")