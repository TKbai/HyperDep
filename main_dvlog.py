import os
import copy
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import option
from model import Model
from dataset_dvlog import DVlogDataset
from train import train
from eval_dvlog import evaluate_dvlog, find_best_threshold, print_metrics
from dataset_dvlog_mw import DVlogMultiWindowDataset


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(args):
    if int(args.cuda) >= 0 and torch.cuda.is_available():
        return "cuda:" + str(args.cuda)
    return "cpu"


def build_loader(args, fold, shuffle, random_crop):
    is_train = fold == args.train_fold

    if is_train:
        num_windows = int(getattr(args, "train_num_windows", 1))
    else:
        num_windows = int(getattr(args, "eval_num_windows", 1))

    if num_windows > 1:
        dataset = DVlogMultiWindowDataset(
            root=args.data_root,
            fold=fold,
            gender=args.gender,
            max_seqlen=args.max_seqlen,
            num_windows=num_windows,
            random_windows=random_crop,
            stats_path=args.stats_path,
        )
    else:
        dataset = DVlogDataset(
            root=args.data_root,
            fold=fold,
            gender=args.gender,
            max_seqlen=args.max_seqlen,
            random_crop=random_crop,
            stats_path=args.stats_path,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, loader


def main():
    args = option.parser.parse_args()

    setup_seed(args.seed)

    args.device = get_device(args)

    os.makedirs(args.save_dir, exist_ok=True)

    print("========== HyperVD-DVlog Training ==========")
    print("device      :", args.device)
    print("data_root   :", args.data_root)
    print("stats_path  :", args.stats_path)
    print("train_fold  :", args.train_fold)
    print("val_fold    :", args.val_fold)
    print("test_fold   :", args.test_fold)
    print("max_seqlen  :", args.max_seqlen)
    print("batch_size  :", args.batch_size)
    print("lr          :", args.lr)
    print("metric      :", args.metric)
    print("fusion      :", args.fusion)
    print("train_windows:", args.train_num_windows)
    print("eval_windows :", args.eval_num_windows)
    print("window_agg   :", args.window_agg)
    print("adj_threshold:", args.adj_threshold)
    print("save_dir    :", args.save_dir)
    print("model_name  :", args.model_name)
    print("============================================")

    train_data, train_loader = build_loader(
        args=args,
        fold=args.train_fold,
        shuffle=True,
        random_crop=True,
    )

    val_data, val_loader = build_loader(
        args=args,
        fold=args.val_fold,
        shuffle=False,
        random_crop=False,
    )

    test_data, test_loader = build_loader(
        args=args,
        fold=args.test_fold,
        shuffle=False,
        random_crop=False,
    )

    print()
    print("Dataset size:")
    print("train:", len(train_data))
    print("val  :", len(val_data))
    print("test :", len(test_data))

    model = Model(args).to(args.device)

    criterion = torch.nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_state = copy.deepcopy(model.state_dict())

    save_path = os.path.join(args.save_dir, args.model_name + "_best.pkl")

    start_time = time.time()

    for epoch in range(1, args.max_epoch + 1):
        print()
        print(f"========== Epoch {epoch}/{args.max_epoch} ==========")

        train_loss, _, _, _ = train(
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            args=args,
            criterion=criterion,
        )

        # 用 validation set 搜索最佳阈值
        threshold_metric = "f1"
        val_threshold, val_threshold_score, _ = find_best_threshold(
            dataloader=val_loader,
            model=model,
            args=args,
            metric=threshold_metric,
        )

        val_metrics = evaluate_dvlog(
            dataloader=val_loader,
            model=model,
            args=args,
            threshold=val_threshold,
        )

        current_score = val_metrics[args.metric]

        print(f"train_loss     : {train_loss:.6f}")
        print(f"val_threshold  : {val_threshold:.6f}")
        print(f"val_{args.metric:10s}: {current_score:.6f}")
        print(
            "val summary    : "
            f"acc={val_metrics['acc']:.4f}, "
            f"f1={val_metrics['f1_weighted']:.4f}, "
            f"auc={val_metrics['auc']:.4f}, "
            f"auprc={val_metrics['auprc']:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_threshold = val_threshold
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "best_score": best_score,
                    "best_threshold": best_threshold,
                    "args": vars(args),
                },
                save_path,
            )

            print(f"saved best model to: {save_path}")

    print()
    print("========== Training Finished ==========")
    print("best_epoch    :", best_epoch)
    print("best_score    :", best_score)
    print("best_threshold:", best_threshold)

    model.load_state_dict(best_state)

    test_metrics = evaluate_dvlog(
        dataloader=test_loader,
        model=model,
        args=args,
        threshold=best_threshold,
    )

    print()
    print_metrics(test_metrics, prefix="========== Test Result ==========")

    elapsed = time.time() - start_time
    print()
    print(f"elapsed time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()