import torch
from tqdm import tqdm


def train(dataloader, model, optimizer, args, criterion, max_batches=None):
    """
    D-Vlog training function for HyperVD.

    dataloader returns:
        inputs: [B, T, 161]
        labels: [B] or [B, 1]

    model returns:
        mil_logits: [B], already passed through sigmoid
        frame_logits: [B, T, 1]
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

        if inputs.dim() != 3:
            raise ValueError(f"Expected inputs shape [B, T, C], got {inputs.shape}")

        # --------------------------------------------------
        # Compute valid sequence length before moving to GPU.
        # Padding positions should be all-zero features.
        # --------------------------------------------------
        with torch.no_grad():
            seq_len = torch.sum(
                torch.max(torch.abs(inputs), dim=2)[0] > 0,
                dim=1,
            )

            max_len = int(seq_len.max().item())
            max_len = max(1, max_len)

        # Trim useless padded tail to reduce memory cost.
        inputs = inputs[:, :max_len, :]

        inputs = inputs.float().to(args.device, non_blocking=True)
        labels = labels.float().view(-1).to(args.device, non_blocking=True)
        seq_len = seq_len.to(args.device)

        mil_logits, frame_logits = model(inputs, seq_len)

        mil_logits = mil_logits.view(-1)

        if mil_logits.shape != labels.shape:
            raise ValueError(
                f"mil_logits shape {mil_logits.shape} does not match "
                f"labels shape {labels.shape}"
            )

        loss = criterion(mil_logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError(
                f"Invalid loss detected: {loss.item()}. "
                f"Check normalization, learning rate, or hyperbolic distance."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Hyperbolic models can be numerically sensitive.
        # Gradient clipping makes the first migration more stable.
        grad_clip = getattr(args, "grad_clip", 5.0)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.detach().item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, 0, 0, 0