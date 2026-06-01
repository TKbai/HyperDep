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
        self.adj_threshold = float(getattr(args, "adj_threshold", 0.8))
        self.adj_mode = getattr(args, "adj_mode", "soft_threshold")
        self.adj_topk = int(getattr(args, "adj_topk", 20))

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

        if self.fusion not in [
            "concat_proj",
            "detour_adapted",
            "gated_scalar",
            "residual_gate",
            "dri_fusion",
        ]:
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

        if self.fusion in ["concat_proj", "gated_scalar", "residual_gate", "dri_fusion"]:
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
        # DRI-Fusion: Depression-aware Residual Interaction Fusion
        #
        # v: visual projected feature [B, T, 128]
        # a: audio projected feature  [B, T, 128]
        #
        # base = concat(v, a)
        # interaction = MLP([v, a, |v-a|, v*a])
        # x = base + gamma * interaction
        #
        # Zero-init the last layer so DRI starts from the original
        # concat baseline.
        # -----------------------------------------------------
        self.dri_mlp = None
        self.dri_gamma = float(getattr(args, "dri_gamma", 0.1))

        if self.fusion == "dri_fusion":
            if self.visual_proj_dim != self.audio_proj_dim:
                raise ValueError(
                    "dri_fusion requires visual_proj_dim == audio_proj_dim, "
                    f"but got {self.visual_proj_dim} and {self.audio_proj_dim}"
                )

            dri_in_dim = self.visual_proj_dim * 4
            dri_hidden_dim = int(getattr(args, "dri_hidden_dim", self.raw_feat_dim))
            dri_dropout = float(getattr(args, "dri_dropout", 0.1))

            self.dri_mlp = nn.Sequential(
                nn.Linear(dri_in_dim, dri_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dri_dropout),
                nn.Linear(dri_hidden_dim, self.raw_feat_dim),
            )

            # Make DRI-Fusion start as exact concat baseline.
            nn.init.zeros_(self.dri_mlp[-1].weight)
            nn.init.zeros_(self.dri_mlp[-1].bias)

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

        # -----------------------------------------------------
        # Lightweight temporal residual convolution.
        # Applied before expmap, while x is still Euclidean.
        # -----------------------------------------------------
        self.use_temporal_conv = int(getattr(args, "use_temporal_conv", 0)) == 1
        self.temporal_conv_scale = float(getattr(args, "temporal_conv_scale", 0.3))

        self.temp_dwconv = None
        self.temp_pwconv = None
        self.temp_act = None

        if self.use_temporal_conv:
            temporal_kernel = int(getattr(args, "temporal_kernel", 5))

            if temporal_kernel % 2 == 0:
                raise ValueError(
                    f"temporal_kernel should be odd, got {temporal_kernel}"
                )

            self.temp_dwconv = nn.Conv1d(
                in_channels=self.raw_feat_dim,
                out_channels=self.raw_feat_dim,
                kernel_size=temporal_kernel,
                padding=temporal_kernel // 2,
                groups=self.raw_feat_dim,
                bias=True,
            )

            self.temp_pwconv = nn.Conv1d(
                in_channels=self.raw_feat_dim,
                out_channels=self.raw_feat_dim,
                kernel_size=1,
                padding=0,
                bias=True,
            )

            self.temp_act = nn.LeakyReLU()

            self.temp_dropout = nn.Dropout(
                float(getattr(args, "temporal_dropout", 0.1))
            )

            # Zero-init last conv so temporal branch starts as no-op.
            nn.init.zeros_(self.temp_pwconv.weight)
            nn.init.zeros_(self.temp_pwconv.bias)

        self.HFSGCN = FHyperGCN(args)
        self.HTRGCN = FHyperGCN(args)

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.HyperCLS = HypClassifier(args)

        # -----------------------------------------------------
        # Late Logit Residual Fusion
        #
        # Main HyperVD branch remains unchanged.
        # Visual/audio branches only provide small residual logits
        # at final video-level prediction.
        #
        # final_logit = main_logit + beta_v * visual_logit + beta_a * audio_logit
        #
        # beta_v and beta_a are zero-initialized, so the model starts
        # exactly as the current concat baseline.
        # -----------------------------------------------------
        self.use_late_logit_fusion = int(getattr(args, "use_late_logit_fusion", 0)) == 1
        self.late_logit_beta_scale = float(getattr(args, "late_logit_beta_scale", 0.2))

        self.late_v_cls = None
        self.late_a_cls = None
        self.raw_beta_v = None
        self.raw_beta_a = None
        self.late_logit_dropout = None

        if self.use_late_logit_fusion:
            self.late_logit_dropout = nn.Dropout(
                float(getattr(args, "late_logit_dropout", 0.0))
            )

            self.late_v_cls = nn.Linear(self.visual_proj_dim, 1)
            self.late_a_cls = nn.Linear(self.audio_proj_dim, 1)

            self.raw_beta_v = nn.Parameter(torch.zeros(1))
            self.raw_beta_a = nn.Parameter(torch.zeros(1))

        # -----------------------------------------------------
        # Learnable window-level attention MIL.
        # Used only when window_agg == "learn_attn".
        # Input dim is args.dim * 2, same as out_x / HyperCLS input.
        # -----------------------------------------------------
        self.window_emb_dim = int(args.dim) * 2

        self.window_attn = nn.Sequential(
            nn.Linear(
                self.window_emb_dim,
                int(getattr(args, "window_attn_hidden", 64)),
            ),
            nn.Tanh(),
            nn.Dropout(float(getattr(args, "window_attn_dropout", 0.1))),
            nn.Linear(
                int(getattr(args, "window_attn_hidden", 64)),
                1,
            ),
        )

        # -----------------------------------------------------
        # Confidence-aware learnable window attention.
        # Input = window_emb + [prob, logit, centered_logit]
        # -----------------------------------------------------
        self.window_attn_conf = nn.Sequential(
            nn.Linear(
                self.window_emb_dim + 3,
                int(getattr(args, "window_attn_hidden", 64)),
            ),
            nn.Tanh(),
            nn.Dropout(float(getattr(args, "window_attn_dropout", 0.1))),
            nn.Linear(
                int(getattr(args, "window_attn_hidden", 64)),
                1,
            ),
        )

        # Zero-init final layer.
        # Attention starts from the confidence prior controlled by
        # --window-attn-logit-bias.
        nn.init.zeros_(self.window_attn_conf[-1].weight)
        nn.init.zeros_(self.window_attn_conf[-1].bias)

        # Important:
        # zero-init makes learn_attn start from uniform attention,
        # approximately equivalent to mean aggregation at the beginning.
        nn.init.zeros_(self.window_attn[-1].weight)
        nn.init.zeros_(self.window_attn[-1].bias)

    def masked_mean_pool(self, x, seq_len):
        """
        x: [B, T, D]
        seq_len: [B]
        return: [B, D]
        """

        if seq_len is None:
            return x.mean(dim=1)

        valid_len = seq_len.long()
        valid_len = torch.clamp(valid_len, min=1, max=x.shape[1])

        mask = (
            torch.arange(x.shape[1], device=x.device)
            .unsqueeze(0)
            < valid_len.unsqueeze(1)
        )

        mask = mask.unsqueeze(-1).to(x.dtype)  # [B, T, 1]

        pooled = (x * mask).sum(dim=1) / valid_len.unsqueeze(-1).to(x.dtype)

        return pooled

    def forward(self, inputs, seq_len, return_embedding=False):
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

        if self.fusion in ["concat_proj", "gated_scalar", "residual_gate", "dri_fusion"]:
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
        # Late unimodal video-level probabilities
        # Used only by Late Logit Residual Fusion.
        # These do not affect the hyperbolic graph features.
        # -----------------------------------------------------
        late_v_prob = None
        late_a_prob = None

        if self.use_late_logit_fusion:
            v_frame_logit = self.late_v_cls(
                self.late_logit_dropout(xv)
            )  # [B, T, 1]

            a_frame_logit = self.late_a_cls(
                self.late_logit_dropout(xa)
            )  # [B, T, 1]

            late_v_prob = self.clas(v_frame_logit, seq_len)  # [B]
            late_a_prob = self.clas(a_frame_logit, seq_len)  # [B]

        # -----------------------------------------------------
        # Fuse visual and acoustic features
        # -----------------------------------------------------
        if self.fusion == "dri_fusion":
            # Base concat feature: same as the strongest baseline.
            base = torch.cat((xv, xa), dim=-1)  # [B, T, 256]

            # Audio-visual interaction features.
            interaction_input = torch.cat(
                [
                    xv,
                    xa,
                    torch.abs(xv - xa),
                    xv * xa,
                ],
                dim=-1,
            )  # [B, T, 512]

            interaction = self.dri_mlp(interaction_input)  # [B, T, 256]

            x = base + self.dri_gamma * interaction

        elif self.fusion in ["gated_scalar", "residual_gate"]:
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
                xv_gated = 2.0 * gate * xv
                xa_gated = 2.0 * (1.0 - gate) * xa

            elif self.fusion == "residual_gate":
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

        if self.use_temporal_conv:
            x = self.temporal_residual_conv(x, seq_len)

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

        graph_branch = getattr(self.args, "graph_branch", "both")

        if graph_branch == "feature_only":
            x2 = torch.zeros_like(x2)

        elif graph_branch == "temporal_only":
            x1 = torch.zeros_like(x1)

        elif graph_branch == "both":
            pass

        else:
            raise ValueError(f"Unknown graph_branch: {graph_branch}")

        feature_branch_weight = float(getattr(self.args, "feature_branch_weight", 1.0))
        temporal_branch_weight = float(getattr(self.args, "temporal_branch_weight", 1.0))

        out_x = torch.cat((x1, x2), dim=2)

        # Window-level representation for learnable window attention.
        # Shape: [B, args.dim * 2]
        window_emb = self.masked_mean_pool(out_x, seq_len)

        frame_prob = self.HyperCLS(out_x)

        # Main HyperVD video-level probability
        mil_logits = self.clas(frame_prob, seq_len)

        # -----------------------------------------------------
        # Late Logit Residual Fusion
        # -----------------------------------------------------
        if self.use_late_logit_fusion and late_v_prob is not None and late_a_prob is not None:
            main_logit = self.prob_to_logit(mil_logits)
            v_logit = self.prob_to_logit(late_v_prob)
            a_logit = self.prob_to_logit(late_a_prob)

            beta_v = self.late_logit_beta_scale * torch.tanh(self.raw_beta_v)
            beta_a = self.late_logit_beta_scale * torch.tanh(self.raw_beta_a)

            final_logit = main_logit + beta_v * v_logit + beta_a * a_logit
            mil_logits = torch.sigmoid(final_logit)

        if return_embedding:
            return mil_logits, frame_prob, window_emb

        return mil_logits, frame_prob

    def temporal_residual_conv(self, x, seq_len):
        """
        Local temporal context before hyperbolic expmap.

        x: [B, T, D]
        seq_len: [B]
        """

        residual = x

        xt = x.transpose(1, 2)  # [B, D, T]
        xt = self.temp_dwconv(xt)
        xt = self.temp_act(xt)
        xt = self.temp_pwconv(xt)
        xt = self.temp_dropout(xt)
        xt = xt.transpose(1, 2)  # [B, T, D]

        x = residual + self.temporal_conv_scale * xt

        # Keep padded positions zero.
        if seq_len is not None:
            valid_len = seq_len.long()
            valid_len = torch.clamp(valid_len, min=1, max=x.shape[1])

            mask = (
                torch.arange(x.shape[1], device=x.device)
                .unsqueeze(0)
                < valid_len.unsqueeze(1)
            )

            x = x * mask.unsqueeze(-1).to(x.dtype)

        return x

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
    
    def normalize_feature_adj(self, adj_sim):
        """
        Build normalized feature-similarity adjacency.

        adj_sim: [L, L], similarity matrix, larger means more similar.

        Modes:
            soft_threshold:
                Original HyperVD-style behavior.
                Values below threshold are set to 0, then softmax.

            hard_threshold:
                Values below threshold are masked out before softmax.

            topk:
                For each snippet, keep top-k most similar snippets.
        """

        if adj_sim.dim() != 2:
            raise ValueError(f"Expected adj_sim shape [L, L], got {adj_sim.shape}")

        l = adj_sim.shape[0]

        if l <= 0:
            return adj_sim

        if self.adj_mode == "soft_threshold":
            adj = F.threshold(adj_sim, self.adj_threshold, 0.0)
            adj = F.softmax(adj, dim=1)
            return adj

        elif self.adj_mode == "hard_threshold":
            mask = adj_sim > self.adj_threshold

            # Always keep self-loop to avoid empty rows.
            eye = torch.eye(l, device=adj_sim.device, dtype=torch.bool)
            mask = mask | eye

            masked_adj = adj_sim.masked_fill(~mask, -1e9)
            adj = F.softmax(masked_adj, dim=1)
            return adj

        elif self.adj_mode == "topk":
            k = max(1, min(self.adj_topk, l))

            values, indices = torch.topk(
                adj_sim,
                k=k,
                dim=1,
                largest=True,
            )

            masked_adj = torch.full_like(adj_sim, -1e9)
            masked_adj.scatter_(dim=1, index=indices, src=values)

            adj = F.softmax(masked_adj, dim=1)
            return adj

        else:
            raise ValueError(f"Unknown adj_mode: {self.adj_mode}")

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
                adj2 = self.normalize_feature_adj(adj2)
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
                adj2 = self.normalize_feature_adj(adj2)

                output[i, :valid_len, :valid_len] = adj2

        return output

    def prob_to_logit(self, prob, eps=1e-6):
        """
        Convert probability to logit safely.
        prob: [B]
        """
        prob = torch.clamp(prob, min=eps, max=1.0 - eps)
        return torch.log(prob / (1.0 - prob))

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

