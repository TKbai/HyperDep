import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def none_or_float(v):
    if v is None:
        return None
    if isinstance(v, str) and v.lower() == "none":
        return None
    return float(v)


parser = argparse.ArgumentParser(description="HyperVD-DVlog")


# =========================================================
# 1. 原 HyperVD 参数：先保留，避免其他文件 import 时出错
# =========================================================
parser.add_argument("--rgb-list", default="list/rgb.list", help="list of rgb features")
parser.add_argument("--flow-list", default="list/flow.list", help="list of flow features")
parser.add_argument("--audio-list", default="list/audio.list", help="list of audio features")

parser.add_argument("--test-rgb-list", default="list/rgb_test.list", help="list of test rgb features")
parser.add_argument("--test-flow-list", default="list/flow_test.list", help="list of test flow features")
parser.add_argument("--test-audio-list", default="list/audio_test.list", help="list of test audio features")

parser.add_argument("--dataset-name", default="D-Vlog", help="dataset name")
parser.add_argument("--gt", default="list/gt.npy", help="file of ground truth")


# =========================================================
# 2. D-Vlog 数据相关参数
# =========================================================
parser.add_argument("--data-root", default="dvlog-dataset", help="root path of D-Vlog dataset")
parser.add_argument("--train-fold", default="train", help="train fold name in labels.csv")
parser.add_argument("--val-fold", default="valid", help="validation fold name in labels.csv")
parser.add_argument("--test-fold", default="test", help="test fold name in labels.csv")

parser.add_argument(
    "--gender",
    default="both",
    choices=["both", "male", "female", "m", "f"],
    help="use all / male / female samples",
)

parser.add_argument(
    "--stats-path",
    default="dvlog_stats.npz",
    help="path of train-set normalization statistics",
)


# =========================================================
# 3. D-Vlog 特征维度
# =========================================================
# D-Vlog: visual 136 + acoustic 25 = 161
parser.add_argument("--visual-dim", type=int, default=136, help="dimension of D-Vlog visual feature")
parser.add_argument("--audio-dim", type=int, default=25, help="dimension of D-Vlog acoustic feature")
parser.add_argument("--input-dim", type=int, default=161, help="raw input feature dimension")

# 后续在 model.py 里会把 visual 和 audio 分别投影到 128 维
parser.add_argument("--visual-proj-dim", type=int, default=128, help="projected visual feature dimension")
parser.add_argument("--audio-proj-dim", type=int, default=128, help="projected audio feature dimension")

# visual_proj_dim + audio_proj_dim = 256
# 这个维度会送入后面的 HyperVD / HGCN 模块
parser.add_argument("--feat-dim", type=int, default=256, help="input size of feature for HGCN")


# =========================================================
# 4. 训练参数
# =========================================================
parser.add_argument("--cuda", type=int, default=0, help="which cuda device to use, -1 for cpu training")

parser.add_argument(
    "--modality",
    default="MIX2",
    help="input modality type: AUDIO, RGB, FLOW, MIX1, MIX2, MIX3, MIX_ALL",
)

# D-Vlog 先用小学习率，双曲模型更稳
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")

# 原 HyperVD 默认 128 对 D-Vlog 太大，先用 8
parser.add_argument("--batch-size", type=int, default=8, help="batch size")

parser.add_argument("--workers", type=int, default=4, help="number of workers in dataloader")
parser.add_argument("--model-name", default="hypervd_dvlog", help="name to save model")
parser.add_argument("--save-dir", default="./ckpt_dvlog", help="directory to save checkpoints")
parser.add_argument("--pretrained-ckpt", default=None, help="ckpt for pretrained model")

parser.add_argument("--dropout", type=float, default=0.6, help="dropout rate")


# =========================================================
# 5. 双曲 / HGCN 模型参数
# =========================================================
parser.add_argument(
    "--model",
    default="HyboNet",
    help="encoder type: Shallow, MLP, HNN, GCN, GAT, HGCN, HyboNet",
)

parser.add_argument(
    "--manifold",
    default="Lorentz",
    help="manifold type: Euclidean, Hyperboloid, PoincareBall, Lorentz",
)

parser.add_argument(
    "--c",
    type=none_or_float,
    default=None,
    help="hyperbolic radius / curvature, None means trainable curvature",
)

parser.add_argument("--num-layers", type=int, default=2, help="number of HGCN layers")
parser.add_argument("--act", default="leaky_relu", help="activation function")
parser.add_argument("--dim", type=int, default=32, help="embedding dimension")
parser.add_argument("--bias", type=int, default=1, choices=[0, 1], help="whether to use bias")
parser.add_argument("--use-att", type=int, default=0, choices=[0, 1], help="whether to use hyperbolic attention")
parser.add_argument("--local-agg", type=int, default=0, choices=[0, 1], help="whether to use local tangent aggregation")
parser.add_argument("--tie_weight", type=str2bool, default=True, help="whether to tie transformation matrices")


# =========================================================
# 6. 分类任务参数
# =========================================================
parser.add_argument("--num-classes", type=int, default=1, help="binary classification output dim")

# 先用 200 跑通 debug；正式实验后面再命令行改成 596
parser.add_argument("--max-seqlen", type=int, default=200, help="maximum sequence length during training")

parser.add_argument("--max-epoch", type=int, default=50, help="maximum training epoch")
parser.add_argument("--seed", type=int, default=9, help="random seed")

parser.add_argument(
    "--metric",
    default="f1",
    choices=["f1", "auc", "auprc", "acc"],
    help="metric used for model selection",
)

parser.add_argument(
    "--pooling",
    default="topk",
    choices=["topk", "mean", "topk_mean"],
    help="MIL pooling strategy: topk, mean, or topk_mean",
)



parser.add_argument(
    "--fusion",
    default="concat_proj",
    choices=["concat_proj", "detour_adapted", "gated_scalar", "residual_gate"],
    help="fusion type: concat_proj, detour_adapted, gated_scalar, or residual_gate",
)

parser.add_argument(
    "--gate-gamma",
    type=float,
    default=0.25,
    help="maximum residual gate strength for residual_gate fusion",
)


parser.add_argument(
    "--pool-alpha",
    type=float,
    default=0.5,
    help="alpha for topk_mean pooling: alpha * mean + (1 - alpha) * topk",
)

parser.add_argument(
    "--topk-divisor",
    type=int,
    default=16,
    help="k = valid_len // topk_divisor + 1 for top-k MIL pooling",
)

parser.add_argument("--eval-threshold", type=float, default=0.5, help="default classification threshold")
parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")