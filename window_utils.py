import torch


def _safe_logit(prob, eps=1e-6):
    """
    Convert probability to logit safely.

    prob: [B, K]
    return: [B, K]
    """

    prob = torch.clamp(prob, min=eps, max=1.0 - eps)
    return torch.log(prob / (1.0 - prob))


def aggregate_window_probs_tensor(
    window_probs,
    mode="mean",
    window_emb=None,
    attn_layer=None,
    attn_temperature=1.0,
    attn_logit_bias=0.5,
    return_attn=False,
):
    """
    Aggregate window-level probabilities into video-level probability.

    window_probs: [B, K]
    window_emb  : [B, K, D], needed by learn_attn / learn_attn_conf

    return:
        [B]
        or ([B], [B, K]) if return_attn=True
    """

    if window_probs.dim() != 2:
        raise ValueError(
            f"Expected window_probs shape [B, K], got {window_probs.shape}"
        )

    window_probs = torch.clamp(window_probs, min=1e-6, max=1.0 - 1e-6)

    # --------------------------------------------------
    # Probability-level aggregation
    # --------------------------------------------------
    if mode == "mean":
        out = window_probs.mean(dim=1)

    elif mode == "max":
        out = window_probs.max(dim=1)[0]

    elif mode == "top2_mean":
        k = min(2, window_probs.shape[1])
        out = torch.topk(window_probs, k=k, dim=1).values.mean(dim=1)

    elif mode == "top3_mean":
        k = min(3, window_probs.shape[1])
        out = torch.topk(window_probs, k=k, dim=1).values.mean(dim=1)

    # --------------------------------------------------
    # Logit-level aggregation
    # --------------------------------------------------
    elif mode == "logit_mean":
        logits = _safe_logit(window_probs)
        video_logit = logits.mean(dim=1)
        out = torch.sigmoid(video_logit)

    elif mode == "logit_top2_mean":
        logits = _safe_logit(window_probs)
        k = min(2, logits.shape[1])
        video_logit = torch.topk(logits, k=k, dim=1).values.mean(dim=1)
        out = torch.sigmoid(video_logit)

    # --------------------------------------------------
    # Parameter-free attention by window logit
    # --------------------------------------------------
    elif mode == "attn_logit":
        logits = _safe_logit(window_probs)
        attn = torch.softmax(logits, dim=1)
        video_logit = torch.sum(attn * logits, dim=1)
        out = torch.sigmoid(video_logit)

    # --------------------------------------------------
    # Noisy-OR MIL
    # --------------------------------------------------
    elif mode == "noisy_or":
        out = 1.0 - torch.prod(1.0 - window_probs, dim=1)
        out = torch.clamp(out, min=1e-6, max=1.0 - 1e-6)

    # --------------------------------------------------
    # Learnable attention using only window embedding
    # --------------------------------------------------
    elif mode == "learn_attn":
        if window_emb is None:
            raise ValueError("window_emb is required when mode='learn_attn'.")

        if attn_layer is None:
            raise ValueError("attn_layer is required when mode='learn_attn'.")

        if window_emb.dim() != 3:
            raise ValueError(
                f"Expected window_emb shape [B, K, D], got {window_emb.shape}"
            )

        temperature = max(float(attn_temperature), 1e-6)

        attn_logits = attn_layer(window_emb).squeeze(-1)  # [B, K]
        attn = torch.softmax(attn_logits / temperature, dim=1)

        out = torch.sum(attn * window_probs, dim=1)

        if return_attn:
            return out, attn

    # --------------------------------------------------
    # Confidence-aware learnable attention
    # --------------------------------------------------
    elif mode == "learn_attn_conf":
        if window_emb is None:
            raise ValueError("window_emb is required when mode='learn_attn_conf'.")

        if attn_layer is None:
            raise ValueError("attn_layer is required when mode='learn_attn_conf'.")

        if window_emb.dim() != 3:
            raise ValueError(
                f"Expected window_emb shape [B, K, D], got {window_emb.shape}"
            )

        if window_emb.shape[:2] != window_probs.shape:
            raise ValueError(
                f"window_emb first two dims {window_emb.shape[:2]} "
                f"must match window_probs shape {window_probs.shape}"
            )

        temperature = max(float(attn_temperature), 1e-6)

        logits = _safe_logit(window_probs)  # [B, K]
        centered_logits = logits - logits.mean(dim=1, keepdim=True)

        # Confidence features:
        # prob, raw logit, relative logit within the same vlog.
        conf_feat = torch.stack(
            [
                window_probs,
                logits,
                centered_logits,
            ],
            dim=-1,
        )  # [B, K, 3]

        # Detach confidence features to avoid the model gaming attention
        # through probability values. Window probability still receives
        # gradients through the final weighted sum.
        conf_feat = conf_feat.detach()
        centered_logits_detached = centered_logits.detach()

        attn_input = torch.cat(
            [
                window_emb,
                conf_feat,
            ],
            dim=-1,
        )  # [B, K, D + 3]

        attn_logits = attn_layer(attn_input).squeeze(-1)  # [B, K]

        # Confidence prior: higher relative window logit gets higher attention.
        # The MLP learns a correction on top of this prior.
        attn_logits = attn_logits + float(attn_logit_bias) * centered_logits_detached

        attn = torch.softmax(attn_logits / temperature, dim=1)

        out = torch.sum(attn * window_probs, dim=1)

        if return_attn:
            return out, attn

    else:
        raise ValueError(f"Unknown window aggregation mode: {mode}")

    if return_attn:
        return out, None

    return out