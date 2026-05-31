from models.base_models import *
from layers.hyp_layers import *
from geoopt import ManifoldParameter
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch.nn.functional as F
import math
from torch.nn.modules.module import Module
from torch import FloatTensor
from torch.nn.parameter import Parameter
import manifolds

class HypClassifier(nn.Module):
    """
    Hyperbolic Classifier
    """

    def __init__(self, args):
        super(HypClassifier, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim * 2
        self.output_dim = args.num_classes
        self.use_bias = args.bias
        self.cls = ManifoldParameter(self.manifold.random_normal((args.num_classes, self.input_dim), std=1./math.sqrt(self.input_dim)), manifold=self.manifold)
        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.num_classes))

    def forward(self, x):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias


class DistanceAdj(Module):

    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.sigma = Parameter(FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen, args):
        # To support batch operations
        self.arith = np.arange(max_seqlen).reshape(-1, 1)
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
        self.dist = torch.from_numpy(squareform(dist)).to(args.device)
        self.dist = torch.exp(-self.dist / torch.exp(torch.tensor(1.)))
        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1).to(args.device)
        return self.dist



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.manifold = getattr(manifolds, args.manifold)()

        # =====================================================
        # D-Vlog input setting:
        # raw input = visual 136 + acoustic 25 = 161
        # projected input for HyperVD = visual 128 + audio 128 = 256
        # =====================================================
        self.visual_dim = int(args.visual_dim)
        self.audio_dim = int(args.audio_dim)
        self.input_dim = self.visual_dim + self.audio_dim

        self.visual_proj_dim = int(args.visual_proj_dim)
        self.audio_proj_dim = int(args.audio_proj_dim)
        self.raw_feat_dim = self.visual_proj_dim + self.audio_proj_dim

        self.fusion = getattr(args, "fusion", "concat_proj")

        if self.fusion not in ["concat_proj", "detour_adapted", "gated_scalar", "residual_gate"]:
            raise ValueError(f"Unknown fusion type: {self.fusion}")

        # args.feat_dim should be 256 before adding Lorentz time axis.
        # For Lorentz/Hyperboloid, expm() will add one time-axis dimension.
        base_feat_dim = int(args.feat_dim)

        if self.manifold.name in ["Lorentz", "Hyperboloid"]:
            if base_feat_dim == self.raw_feat_dim:
                args.feat_dim = base_feat_dim + 1
            elif base_feat_dim == self.raw_feat_dim + 1:
                # Already added once. Keep it.
                pass
            else:
                raise ValueError(
                    f"args.feat_dim should be {self.raw_feat_dim} or "
                    f"{self.raw_feat_dim + 1}, but got {base_feat_dim}."
                )
        else:
            if base_feat_dim != self.raw_feat_dim:
                raise ValueError(
                    f"For non-Lorentz manifold, args.feat_dim should be "
                    f"{self.raw_feat_dim}, but got {base_feat_dim}."
                )

        self.disAdj = DistanceAdj()

        # =====================================================
        # Visual projection
        #
        # concat_proj:
        #   visual: 136 -> 256 -> 128
        #   audio : 25  -> 64  -> 128
        #
        # detour_adapted:
        #   visual: 136 -> 256 -> 128
        #   audio : 25  -> 128
        #
        # detour_adapted is closer to HyperVD's detour fusion:
        # visual branch is learned more strongly, audio branch is kept shallower.
        # =====================================================

        self.v_conv1 = nn.Conv1d(
            in_channels=self.visual_dim,
            out_channels=256,
            kernel_size=1,
            padding=0,
        )

        self.v_conv2 = nn.Conv1d(
            in_channels=256,
            out_channels=self.visual_proj_dim,
            kernel_size=1,
            padding=0,
        )

        self.a_conv1 = None
        self.a_conv2 = None
        self.a_proj = None

        if self.fusion in ["concat_proj", "gated_scalar", "residual_gate"]:
            # Current baseline: symmetric two-step audio projection
            # Also used by gated_scalar fusion.
            self.a_conv1 = nn.Conv1d(
                in_channels=self.audio_dim,
                out_channels=64,
                kernel_size=1,
                padding=0,
            )

            self.a_conv2 = nn.Conv1d(
                in_channels=64,
                out_channels=self.audio_proj_dim,
                kernel_size=1,
                padding=0,
            )

        elif self.fusion == "detour_adapted":
            # HyperVD-style adapted detour:
            # audio is only linearly projected to 128 dim,
            # without extra nonlinearity/dropout.
            self.a_proj = nn.Conv1d(
                in_channels=self.audio_dim,
                out_channels=self.audio_proj_dim,
                kernel_size=1,
                padding=0,
            )

        # -----------------------------------------------------
        # Scalar modality reliability gate.
        # Used only when fusion == "gated_scalar".
        # gate_t controls frame-level visual/audio reliability.
        # -----------------------------------------------------
        self.gate_mlp = None

        if self.fusion in ["gated_scalar", "residual_gate"]:
            if self.visual_proj_dim != self.audio_proj_dim:
                raise ValueError(
                    "gated_scalar requires visual_proj_dim == audio_proj_dim, "
                    f"but got {self.visual_proj_dim} and {self.audio_proj_dim}"
                )

            gate_in_dim = self.visual_proj_dim * 3  # [v, a, |v-a|]

            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_in_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

            # Initialize gate around 0.5.
            # With x = concat(2*g*v, 2*(1-g)*a),
            # g=0.5 makes the module start from normal concat behavior.
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)
        self.gate_gamma = float(getattr(args, "gate_gamma", 0.25))

        self.v_aux_cls = nn.Linear(self.visual_proj_dim, 1)
        self.a_aux_cls = nn.Linear(self.audio_proj_dim, 1)
        self.aux_outputs = {}

        self.HFSGCN = FHyperGCN(args)
        self.HTRGCN = FHyperGCN(args)

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.HyperCLS = HypClassifier(args)

    def forward(self, inputs, seq_len):
        """
        inputs: [B, T, 161]
                first 136 dims are visual features,
                last 25 dims are acoustic features.

        seq_len: [B], valid sequence length after padding.
        """

        if inputs.dim() != 3:
            raise ValueError(
                f"Expected inputs with shape [B, T, C], but got {inputs.shape}."
            )

        if inputs.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim} "
                f"= visual_dim {self.visual_dim} + audio_dim {self.audio_dim}, "
                f"but got {inputs.size(-1)}."
            )

        # -----------------------------------------------------
        # Split D-Vlog features
        # -----------------------------------------------------
        xv = inputs[:, :, :self.visual_dim]  # [B, T, 136]
        xa = inputs[:, :, self.visual_dim:self.visual_dim + self.audio_dim]  # [B, T, 25]

        # -----------------------------------------------------
        # Visual branch
        # [B, T, 136] -> [B, 136, T] -> [B, 128, T] -> [B, T, 128]
        # -----------------------------------------------------
        xv = xv.permute(0, 2, 1)
        xv = self.relu(self.v_conv1(xv))
        xv = self.dropout(xv)
        xv = self.relu(self.v_conv2(xv))
        xv = self.dropout(xv)
        xv = xv.permute(0, 2, 1)

        # -----------------------------------------------------
        # Acoustic branch
        # concat_proj:
        #   [B, T, 25] -> [B, 25, T] -> [B, 64, T] -> [B, 128, T] -> [B, T, 128]
        #
        # detour_adapted:
        #   [B, T, 25] -> [B, 25, T] -> [B, 128, T] -> [B, T, 128]
        # -----------------------------------------------------
        xa = xa.permute(0, 2, 1)

        if self.fusion in ["concat_proj", "gated_scalar", "residual_gate"]:
            xa = self.relu(self.a_conv1(xa))
            xa = self.dropout(xa)
            xa = self.relu(self.a_conv2(xa))
            xa = self.dropout(xa)

        elif self.fusion == "detour_adapted":
            xa = self.a_proj(xa)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")

        xa = xa.permute(0, 2, 1)


        self.aux_outputs = {}

        aux_loss_weight = float(getattr(self.args, "aux_loss_weight", 0.0))
        aux_modalities = getattr(self.args, "aux_modalities", "both")

        if self.training and aux_loss_weight > 0 and aux_modalities != "none":

            if aux_modalities in ["visual", "both"]:
                v_frame_logits = self.v_aux_cls(xv)
                self.aux_outputs["visual"] = self.clas(v_frame_logits, seq_len)

            if aux_modalities in ["audio", "both"]:
                a_frame_logits = self.a_aux_cls(xa)
                self.aux_outputs["audio"] = self.clas(a_frame_logits, seq_len)

        # -----------------------------------------------------
        # Fuse visual and acoustic features
        # -----------------------------------------------------
        if self.fusion in ["gated_scalar", "residual_gate"]:
            gate_input = torch.cat(
                [
                    xv,
                    xa,
                    torch.abs(xv - xa),
                ],
                dim=-1,
            )

            gate = torch.sigmoid(self.gate_mlp(gate_input))  # [B, T, 1]

            if self.fusion == "gated_scalar":
                # Raw gate, kept for ablation.
                xv_gated = 2.0 * gate * xv
                xa_gated = 2.0 * (1.0 - gate) * xa

            elif self.fusion == "residual_gate":
                # Conservative residual gate.
                # delta is restricted to [-gamma, gamma].
                gamma = max(0.0, min(float(self.gate_gamma), 1.0))
                delta = gamma * (2.0 * gate - 1.0)

                xv_gated = (1.0 + delta) * xv
                xa_gated = (1.0 - delta) * xa

            x = torch.cat((xv_gated, xa_gated), dim=-1)

        else:
            x = torch.cat((xv, xa), dim=-1)


        if x.size(-1) != self.raw_feat_dim:
            raise ValueError(
                f"Expected fused feature dim {self.raw_feat_dim}, "
                f"but got {x.size(-1)}."
            )

        # Temporal-distance graph
        disadj = self.disAdj(x.shape[0], x.shape[1], self.args).to(x.device)

        # Map Euclidean features to hyperbolic space
        proj_x = self.expm(x)

        # Feature-similarity graph
        adj = self.adj(proj_x, seq_len)

        # Two hyperbolic GCN branches
        x1 = self.relu(self.HFSGCN.encode(proj_x, adj))
        x1 = self.dropout(x1)

        x2 = self.relu(self.HTRGCN.encode(proj_x, disadj))
        x2 = self.dropout(x2)

        out_x = torch.cat((x1, x2), dim=2)

        # Frame/snippet-level logits
        frame_prob = self.HyperCLS(out_x)

        # Video-level probability after MIL pooling
        mil_logits = self.clas(frame_prob, seq_len)

        return mil_logits, frame_prob

    def expm(self, x):
        """
        Map Euclidean tangent vectors to Lorentz/Hyperboloid manifold.
        Input x: [B, T, D]
        Output:  [B, T, D + 1] when using Lorentz/Hyperboloid
        """
        if self.manifold.name in ["Lorentz", "Hyperboloid"]:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, :, 0:1], x], dim=-1)

            if self.manifold.name == "Lorentz":
                x = self.manifold.expmap0(x)

            return x

        return x

    def adj(self, x, seq_len):
        """
        Build feature-similarity graph in hyperbolic space.
        x: [B, T, D]
        """
        soft = nn.Softmax(dim=1)

        x2 = self.lorentz_similarity(x, x, self.manifold.k)
        x2 = torch.exp(-x2)

        output = torch.zeros_like(x2)

        if seq_len is None:
            for i in range(x.shape[0]):
                adj2 = x2[i]
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(x.shape[0]):
                valid_len = seq_len[i]
                if torch.is_tensor(valid_len):
                    valid_len = int(valid_len.detach().cpu().item())
                else:
                    valid_len = int(valid_len)

                valid_len = max(1, min(valid_len, x.shape[1]))

                adj2 = x2[i, :valid_len, :valid_len]
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)

                output[i, :valid_len, :valid_len] = adj2

        return output

    def clas(self, logits, seq_len):
        """
        MIL pooling from snippet-level logits to video-level probability.

        Supported pooling:
            topk       : original HyperVD pooling
            mean       : average over all valid snippets
            topk_mean  : alpha * mean + (1 - alpha) * topk

        For D-Vlog depression detection, topk_mean is often more suitable
        than pure topk because depressive cues are usually diffuse rather
        than short burst events.
        """

        # logits: [B, T, 1] -> [B, T]
        logits = logits.squeeze(-1)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        pooling = getattr(self.args, "pooling", "topk")
        pool_alpha = float(getattr(self.args, "pool_alpha", 0.5))
        topk_divisor = int(getattr(self.args, "topk_divisor", 16))

        pool_alpha = max(0.0, min(1.0, pool_alpha))
        topk_divisor = max(1, topk_divisor)

        instance_logits = []

        for i in range(logits.shape[0]):
            if seq_len is None:
                valid_len = logits.shape[1]
            else:
                valid_len = seq_len[i]

                if torch.is_tensor(valid_len):
                    valid_len = int(valid_len.detach().cpu().item())
                else:
                    valid_len = int(valid_len)

                valid_len = max(1, min(valid_len, logits.shape[1]))

            valid_scores = logits[i, :valid_len]

            mean_score = torch.mean(valid_scores)

            k = valid_len // topk_divisor + 1
            k = max(1, min(k, valid_len))

            topk_score, _ = torch.topk(
                valid_scores,
                k=k,
                largest=True,
            )
            topk_score = torch.mean(topk_score)

            if pooling == "topk":
                video_score = topk_score

            elif pooling == "mean":
                video_score = mean_score

            elif pooling == "topk_mean":
                video_score = pool_alpha * mean_score + (1.0 - pool_alpha) * topk_score

            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")

            instance_logits.append(video_score.view(1))

        instance_logits = torch.cat(instance_logits, dim=0)

        # Keep the same behavior as before:
        # model output is probability, criterion uses BCELoss.
        instance_prob = torch.sigmoid(instance_logits)

        return instance_prob

    def lorentz_similarity(self, x: torch.Tensor, y: torch.Tensor, k) -> torch.Tensor:
        """
        Lorentz distance matrix.

        x: [B, T, D]
        y: [B, T, D]
        return: [B, T, T]
        """

        eps = 1e-6 if x.dtype == torch.float32 else 1e-8

        # Lorentz metric diag(-1, 1, ..., 1)
        metric = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        metric[0] = -1.0

        temp = x * metric
        xy_inner = -(temp @ y.transpose(-1, -2))

        xy_inner = torch.clamp(xy_inner, min=1.0 + eps)

        sqrt_k = k ** 0.5
        dist = sqrt_k * self.arccosh(xy_inner / k)
        dist = torch.clamp(dist, min=eps, max=200)

        return dist

    def arccosh(self, x):
        """
        Numerically stable arccosh.
        """
        eps = 1e-6 if x.dtype == torch.float32 else 1e-8
        x = torch.clamp(x, min=1.0 + eps)
        return torch.log(x + torch.sqrt(torch.pow(x, 2) - 1.0))

