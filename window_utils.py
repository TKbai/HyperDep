import torch


def _safe_logit(prob, eps=1e-6):
    """
    Convert probability to logit safely.

    prob: [B, K]
    return: [B, K]
    """

    prob = torch.clamp(prob, min=eps, max=1.0 - eps)
    return torch.log(prob / (1.0 - prob))


def aggregate_window_probs_tensor(window_probs, mode="mean"):
    """
    Aggregate window-level probabilities into video-level probability.

    window_probs: [B, K]
    return: [B]

    Supported modes:
        mean
        max
        top2_mean
        top3_mean

        logit_mean
        logit_top2_mean
        attn_logit
        noisy_or
    """

    if window_probs.dim() != 2:
        raise ValueError(
            f"Expected window_probs shape [B, K], got {window_probs.shape}"
        )

    # Avoid numerical issues when converting probability to logit.
    window_probs = torch.clamp(window_probs, min=1e-6, max=1.0 - 1e-6)

    # --------------------------------------------------
    # Existing probability-level aggregation
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Logit-level aggregation
    # --------------------------------------------------
    if mode == "logit_mean":
        logits = _safe_logit(window_probs)
        video_logit = logits.mean(dim=1)
        return torch.sigmoid(video_logit)

    if mode == "logit_top2_mean":
        logits = _safe_logit(window_probs)
        k = min(2, logits.shape[1])
        video_logit = torch.topk(logits, k=k, dim=1).values.mean(dim=1)
        return torch.sigmoid(video_logit)

    # --------------------------------------------------
    # Parameter-free window-level attention MIL
    # --------------------------------------------------
    if mode == "attn_logit":
        logits = _safe_logit(window_probs)

        # Higher-logit windows get larger attention weights.
        # This is attention MIL without extra learnable parameters.
        attn = torch.softmax(logits, dim=1)

        video_logit = torch.sum(attn * logits, dim=1)
        return torch.sigmoid(video_logit)

    # --------------------------------------------------
    # Noisy-OR MIL
    # --------------------------------------------------
    if mode == "noisy_or":
        # p(video positive) = 1 - product_i(1 - p_i)
        video_prob = 1.0 - torch.prod(1.0 - window_probs, dim=1)
        return torch.clamp(video_prob, min=1e-6, max=1.0 - 1e-6)

    raise ValueError(f"Unknown window aggregation mode: {mode}")