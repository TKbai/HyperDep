import torch


def aggregate_window_probs_tensor(window_probs, mode="mean"):
    """
    window_probs: [B, K]
    return: [B]

    mode:
        mean
        max
        top2_mean
        top3_mean
    """

    if window_probs.dim() != 2:
        raise ValueError(
            f"Expected window_probs shape [B, K], got {window_probs.shape}"
        )

    if mode == "mean":
        return window_probs.mean(dim=1)

    if mode == "max":
        return window_probs.max(dim=1)[0]

    if mode == "top2_mean":
        k = min(2, window_probs.shape[1])
        return torch.topk(window_probs, k=k, dim=1).values.mean(dim=1)

    if mode == "top3_mean":
        k = min(3, window_probs.shape[1])
        return torch.topk(window_probs, k=k, dim=1).values.mean(dim=1)

    raise ValueError(f"Unknown window aggregation mode: {mode}")
