import torch
from tqdm import tqdm

from window_utils import aggregate_window_probs_tensor


def _compute_seq_len(inputs):
    """
    inputs: [B, T, C]
    return: [B]
    """
    seq_len = torch.sum(
        torch.max(torch.abs(inputs), dim=2)[0] > 0,
        dim=1,
    )

    seq_len = torch.clamp(seq_len, min=1)

    return seq_len


def _compute_loss_with_aux(model, criterion, labels, args, base_loss, batch_size=None, num_windows=None):
    """
    Add auxiliary unimodal losses if enabled.

    For normal training:
        aux_prob: [B]

    For multi-window training:
        aux_prob: [B*K], reshape to [B, K] and aggregate.
    """

    loss = base_loss

    aux_loss_weight = float(getattr(args, "aux_loss_weight", 0.0))

    if aux_loss_weight <= 0:
        return loss

    if not hasattr(model, "aux_outputs"):
        return loss

    aux_losses = []

    for aux_name, aux_prob in model.aux_outputs.items():
        aux_prob = aux_prob.view(-1)

        if batch_size is not None and num_windows is not None:
            aux_prob = aux_prob.view(batch_size, num_windows)
            aux_agg_mode = getattr(args, "window_agg", "mean")

            # Auxiliary branches do not have window embeddings.
            # Use mean aggregation for aux loss when main aggregation is learn_attn.
            if aux_agg_mode == "learn_attn":
                aux_agg_mode = "mean"

            aux_prob = aggregate_window_probs_tensor(
                aux_prob,
                mode=aux_agg_mode,
            )

        if aux_prob.shape != labels.shape:
            raise ValueError(
                f"aux output {aux_name} shape {aux_prob.shape} "
                f"does not match labels shape {labels.shape}"
            )

        aux_losses.append(criterion(aux_prob, labels))

    if len(aux_losses) > 0:
        aux_loss = torch.stack(aux_losses).mean()
        loss = loss + aux_loss_weight * aux_loss

    return loss


def train(dataloader, model, optimizer, args, criterion, max_batches=None):
    """
    D-Vlog training function for HyperVD.

    Supports:
        inputs: [B, T, C]
        inputs: [B, K, T, C]
    """

    model.train()

    total_loss = 0.0
    num_batches = 0

    for i, (inputs, labels) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Training",
    ):
        if max_batches is not None and i >= max_batches:
            break

        labels = labels.float().view(-1).to(args.device, non_blocking=True)

        # ==================================================
        # Case 1: standard single-window input [B, T, C]
        # ==================================================
        if inputs.dim() == 3:
            with torch.no_grad():
                seq_len = _compute_seq_len(inputs)
                max_len = int(seq_len.max().item())
                max_len = max(1, max_len)

            inputs = inputs[:, :max_len, :]

            inputs = inputs.float().to(args.device, non_blocking=True)
            seq_len = seq_len.to(args.device)

            video_prob, frame_prob = model(inputs, seq_len)
            video_prob = video_prob.view(-1)

            if video_prob.shape != labels.shape:
                raise ValueError(
                    f"video_prob shape {video_prob.shape} does not match "
                    f"labels shape {labels.shape}"
                )

            loss = criterion(video_prob, labels)
            loss = _compute_loss_with_aux(
                model=model,
                criterion=criterion,
                labels=labels,
                args=args,
                base_loss=loss,
            )

        # ==================================================
        # Case 2: multi-window input [B, K, T, C]
        # ==================================================
        elif inputs.dim() == 4:
            b, k, t, c = inputs.shape

            if labels.shape[0] != b:
                raise ValueError(
                    f"labels batch size {labels.shape[0]} does not match "
                    f"inputs batch size {b}"
                )

            inputs = inputs.view(b * k, t, c)

            with torch.no_grad():
                seq_len = _compute_seq_len(inputs)
                max_len = int(seq_len.max().item())
                max_len = max(1, max_len)

            inputs = inputs[:, :max_len, :]

            inputs = inputs.float().to(args.device, non_blocking=True)
            seq_len = seq_len.to(args.device)

            window_agg = getattr(args, "window_agg", "mean")

            if window_agg == "learn_attn":
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

            video_prob = aggregate_window_probs_tensor(
                window_prob,
                mode=window_agg,
                window_emb=window_emb,
                attn_layer=getattr(model, "window_attn", None),
                attn_temperature=getattr(args, "window_attn_temperature", 1.0),
            )

            if video_prob.shape != labels.shape:
                raise ValueError(
                    f"video_prob shape {video_prob.shape} does not match "
                    f"labels shape {labels.shape}"
                )

            loss = criterion(video_prob, labels)
            loss = _compute_loss_with_aux(
                model=model,
                criterion=criterion,
                labels=labels,
                args=args,
                base_loss=loss,
                batch_size=b,
                num_windows=k,
            )

        else:
            raise ValueError(
                f"Expected inputs shape [B,T,C] or [B,K,T,C], got {inputs.shape}"
            )

        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError(
                f"Invalid loss detected: {loss.item()}. "
                f"Check normalization, learning rate, or hyperbolic distance."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_clip = getattr(args, "grad_clip", 5.0)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.detach().item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, 0, 0, 0