import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEdgeGATv2Adj(nn.Module):
    """
    GATv2-style dynamic refinement of an existing feature adjacency.

    The original HyperVD feature adjacency is used as a prior:

        refined_logits_ij = log(base_adj_ij + eps) + scale * delta_ij

    where delta_ij is produced by a GATv2-style query-conditioned score.
    The final score layer is zero-initialized, so the module starts exactly
    from the original adjacency and learns only a residual correction.

    Inputs:
        x        : [B, T, D] Euclidean fused snippet features
        base_adj : [B, T, T] row-normalized baseline feature adjacency
        seq_len  : [B] valid sequence lengths, or None

    Output:
        refined_adj: [B, T, T], row-normalized over valid nodes
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        delta_scale: float = 1.0,
        use_temporal_edge: bool = True,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_scale = float(delta_scale)
        self.use_temporal_edge = bool(use_temporal_edge)

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.src_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.dst_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)

        self.temporal_proj = None
        if self.use_temporal_edge:
            self.temporal_proj = nn.Linear(1, self.hidden_dim, bias=False)

        self.act = nn.LeakyReLU(negative_slope=negative_slope)
        self.score = nn.Linear(self.hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Exact baseline initialization: delta_ij = 0 for all edges.
        nn.init.zeros_(self.score.weight)

    def _valid_pair_mask(self, batch_size, seq_len, max_len, device):
        if seq_len is None:
            return torch.ones(
                batch_size,
                max_len,
                max_len,
                dtype=torch.bool,
                device=device,
            )

        valid_len = seq_len.to(device=device).long()
        valid_len = torch.clamp(valid_len, min=1, max=max_len)

        positions = torch.arange(max_len, device=device).unsqueeze(0)
        valid_node = positions < valid_len.unsqueeze(1)  # [B, T]

        return valid_node.unsqueeze(2) & valid_node.unsqueeze(1)

    def _normalized_temporal_distance(self, max_len, device, dtype):
        pos = torch.arange(max_len, device=device, dtype=dtype)
        distance = torch.abs(pos[:, None] - pos[None, :])
        denominator = max(max_len - 1, 1)
        return (distance / float(denominator)).unsqueeze(-1)  # [T, T, 1]

    def forward(self, x, base_adj, seq_len=None, return_details=False):
        if x.dim() != 3:
            raise ValueError(f"Expected x [B, T, D], got {x.shape}")
        if base_adj.dim() != 3:
            raise ValueError(
                f"Expected base_adj [B, T, T], got {base_adj.shape}"
            )

        batch_size, max_len, input_dim = x.shape

        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected x dim {self.input_dim}, got {input_dim}"
            )
        if base_adj.shape != (batch_size, max_len, max_len):
            raise ValueError(
                "base_adj shape must match x temporal dimensions: "
                f"x={x.shape}, base_adj={base_adj.shape}"
            )

        h = self.input_norm(x)
        src = self.src_proj(h)  # [B, T, H]
        dst = self.dst_proj(h)  # [B, T, H]

        # GATv2-style dynamic pair representation.
        pair_hidden = src.unsqueeze(2) + dst.unsqueeze(1)  # [B, T, T, H]

        if self.temporal_proj is not None:
            temporal_distance = self._normalized_temporal_distance(
                max_len=max_len,
                device=x.device,
                dtype=x.dtype,
            )
            temporal_hidden = self.temporal_proj(temporal_distance)
            pair_hidden = pair_hidden + temporal_hidden.unsqueeze(0)

        delta = self.score(self.act(pair_hidden)).squeeze(-1)  # [B, T, T]
        delta = self.dropout(delta)

        eps = 1e-8 if base_adj.dtype == torch.float64 else 1e-6
        base_logits = torch.log(torch.clamp(base_adj, min=eps))
        scaled_delta = self.delta_scale * delta
        logits = base_logits + scaled_delta

        pair_mask = self._valid_pair_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            max_len=max_len,
            device=x.device,
        )

        logits = logits.masked_fill(~pair_mask, -1e9)
        refined_adj = F.softmax(logits, dim=-1)

        # Padded rows/columns must remain exactly zero.
        refined_adj = refined_adj * pair_mask.to(refined_adj.dtype)

        # Re-normalize valid rows after masking/dropout effects.
        denominator = refined_adj.sum(dim=-1, keepdim=True).clamp_min(eps)
        refined_adj = refined_adj / denominator
        refined_adj = refined_adj * pair_mask.to(refined_adj.dtype)

        if return_details:
            return refined_adj, {
                "delta": delta,
                "scaled_delta": scaled_delta,
                "pair_mask": pair_mask,
            }

        return refined_adj
