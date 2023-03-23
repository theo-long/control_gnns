import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Union
from training import train_eval, TrainConfig
from models import GraphMLP, GCN
from data import (
    generate_dataloaders,
    get_dataset,
    get_test_val_train_split,
    get_test_val_train_mask,
)
from utils import get_device, parse_callable_string, BasicLogger, CSVLogger

import torchmetrics
from torch.nn.functional import cross_entropy
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("-s", "--split", default=None, type=int)

    parser.add_argument(
        "--control_type", default="null", type=str, choices=["null", "gcn", "mp"]
    )
    parser.add_argument(
        "--control_edges", default="adj", type=str, choices=["adj", "dense_subset"]
    )
    parser.add_argument(
        "--control_metric",
        default="b_centrality",
        type=str,
        choices=["degree", "b_centrality", "pr_centrality", "curvature"],
    )
    parser.add_argument("--control_k", default="1", type=str)
    parser.add_argument("--control_self_adj", action="store_true")

    parser.add_argument("-t", "--time_inv", action="store_true", default=False)
    parser.add_argument("-l", "--linear", action="store_true", default=False)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--conv_depth", default=2, type=int)
    parser.add_argument(
        "--norm", default="layernorm", choices=[None, "batchnorm", "layernorm"]
    )
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--control_init", default=None, type=float)

    args = parser.parse_args()

    dataset, is_node_classifier = get_dataset(
        args.dataset,
        args.control_type,
        args.control_edges,
        args.control_metric,
        parse_callable_string(args.control_k),
        args.control_self_adj,
    )

    if args.norm == "batchnorm":
        norm = lambda channels: torch.nn.BatchNorm1d(momentum=args.bn_momentum)
    elif args.norm == "layernorm":
        norm = torch.nn.LayerNorm
    elif args.norm is None:
        norm = None
    else:
        raise ValueError("Norm must be None, layernorm or batchnorm")

    if args.model == "mlp":
        model_factory = lambda dropout_rate: GraphMLP(
            input_dim=dataset[0].x.shape[1],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            dropout_rate=dropout_rate,
            is_node_classifier=is_node_classifier,
            norm=norm,
        )
    elif args.model == "gcn":
        model_factory = lambda dropout_rate: GCN(
            input_dim=dataset[0].x.shape[1],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            conv_depth=args.conv_depth,
            dropout_rate=dropout_rate,  # passed during hyperparameter tuning
            linear=args.linear,
            time_inv=args.time_inv,
            control_type=args.control_type,
            is_node_classifier=is_node_classifier,
            norm=norm,
            control_init=args.control_init,
        )
    else:
        raise ValueError(f"Model name {args.model} not recognized")

    device = get_device()
    accuracy_function = torchmetrics.Accuracy(
        "multiclass", num_classes=dataset.num_classes
    ).to(device)

    if args.split is None:
        splits = range(5)
    else:
        splits = args.split

    run_logger = CSVLogger(args.name+'_run.csv', [
        "lr",
        "weight_decay",
        "dropout_rate",
        "split",
        "best_val_loss",
        "test_loss",
        "test_metric",
        ])

    epoch_logger = CSVLogger(args.name+'_epoch.csv', [
            "lr",
            "weight_decay",
            "dropout_rate",
            "split",
            "epoch",
            "train_loss",
            "val_loss",
            "train_metric",
            "val_metric"])

    for lr in [1e-4, 1e-3, 1e-2]:
        for weight_decay in [0.0, 1e-6, 1e-5]:
            for dropout_rate in [0.0, 0.2, 0.4]:

                mean_stats = {"best_val_loss": [],
                    "test_loss": [],
                    "test_metric": []}

                for split in splits:

                    log_const = {
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "dropout_rate": dropout_rate,
                            "split": split
                            }

                    epoch_logger.update_const(log_const)

                    if is_node_classifier:
                        train_loader, val_loader, test_loader = dataset, dataset, dataset
                        train_mask, val_mask, test_mask = get_test_val_train_mask(
                            dataset, split=split
                        )
                    else:
                        train_loader, val_loader, test_loader = generate_dataloaders(
                            dataset, args.dataset, args.batch_size, split=split
                        )
                        train_mask, val_mask, test_mask = None, None, None

                    training_config = TrainConfig(
                        lr=lr,
                        batch_size=1 if is_node_classifier else train_loader.batch_size,
                        epochs=args.epochs,
                        weight_decay=weight_decay
                    )

                    if args.save_models:
                        save_path=f'models/{args.name}_lr{lr}_wd{weight_decay}_do{dropout_rate}_s{split}.pt'
                    else:
                        save_path=None

                    model = model_factory(dropout_rate=dropout_rate)
                    run_stats = train_eval(
                        model,
                        training_config,
                        train_loader,
                        val_loader,
                        test_loader,
                        loss_function=cross_entropy,
                        metric_function=accuracy_function,
                        logger = epoch_logger,
                        train_mask=train_mask,
                        val_mask=val_mask,
                        test_mask=test_mask,
                        save_path=save_path,
                    )

                    for key, value in run_stats.items():
                        mean_stats[key].append(value)

                    print(lr, weight_decay, dropout_rate, split, run_stats)

                    run_stats = {**run_stats, **log_const}
                    run_logger.log(run_stats)

                for key in mean_stats.keys():
                    mean_stats[key] = np.mean(mean_stats[key])

                print(lr, weight_decay, dropout_rate, "avg", mean_stats)

                log_const["split"] = "avg"
                mean_stats = {**mean_stats, **log_const}
                run_logger.log(mean_stats)

if __name__ == "__main__":
    main()
