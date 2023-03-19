import argparse
import wandb
import pandas as pd

import torch
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

from training import train_eval, TrainConfig, BasicLogger
from data import get_dataset, generate_dataloaders, get_test_val_train_mask
from models import GCN, GraphMLP
from utils import parse_callable_string, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("-n", "--num_runs", default=1, type=int)

    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--time_inv", action="store_true")

    parser.add_argument(
        "--control_type",
        default="null",
        type=str,
        choices=["null", "gcn", "mp", "random"],
    )
    parser.add_argument(
        "--random_control_method", default="edge", type=str, choices=["path", "edge"]
    )
    parser.add_argument("--random_control_rate", default=0.9, type=float)
    parser.add_argument(
        "--control_edges", default="adj", type=str, choices=["adj", "dense", "two_hop"]
    )
    parser.add_argument(
        "--control_metric",
        default="b_centrality",
        type=str,
        choices=["degree", "b_centrality", "pr_centrality"],
    )
    parser.add_argument("--control_k", default=lambda x: 1, type=parse_callable_string)
    parser.add_argument("--control_self_adj", action="store_true")

    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--conv_depth", default=2, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)

    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("-d", "--debug", action="store_true")

    parser.add_argument("-s", "--split", default=0, type=int)

    parser.add_argument(
        "--norm", default="layernorm", choices=["none", "batchnorm", "layernorm"]
    )
    parser.add_argument("--bn_momentum", default=0.1, type=float)
    parser.add_argument("--residual", action="store_true")

    args = parser.parse_args()

    training_config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    dataset, is_node_classifier = get_dataset(
        args.dataset,
        args.control_type,
        args.control_edges,
        args.control_metric,
        args.control_k,
        args.control_self_adj,
    )

    if is_node_classifier:
        train_loader, val_loader, test_loader = dataset, dataset, dataset
        train_mask, val_mask, test_mask = get_test_val_train_mask(
            dataset, split=args.split
        )
    else:
        train_loader, val_loader, test_loader = generate_dataloaders(
            dataset, args.dataset, args.batch_size, split=args.split
        )
        train_mask, val_mask, test_mask = None, None, None

    if args.norm == "batchnorm":
        norm = lambda channels: torch.nn.BatchNorm1d(
            channels, momentum=args.bn_momentum
        )
    elif args.norm == "layernorm":
        norm = torch.nn.LayerNorm
    elif args.norm == "none":
        norm = None
    else:
        raise ValueError("Norm must be None, layernorm or batchnorm")

    if args.model.lower() == "gcn":

        if args.control_type == "random":
            control_kwargs = {
                "method": args.random_control_method,
                "rate": args.random_control_rate,
            }
        else:
            control_kwargs = {}

        model_factory = lambda: GCN(
            input_dim=dataset[0].x.shape[1],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            conv_depth=args.conv_depth,
            dropout_rate=args.dropout,
            linear=args.linear,
            time_inv=args.time_inv,
            control_type=args.control_type,
            is_node_classifier=is_node_classifier,
            norm=norm,
            residual=args.residual,
            **control_kwargs,
        )

    elif args.model.lower() == "mlp":
        model_factory = lambda: GraphMLP(
            input_dim=dataset[0].x.shape[1],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout,
            is_node_classifier=is_node_classifier,
            norm=norm,
        )
    else:
        raise ValueError(f"Model name {args.model} not recognized")

    stats = []
    for run in range(args.num_runs):
        print(f"Starting run {run}", end="\r")
        if not args.debug:
            run = wandb.init(
                project="control_gnns", config=training_config, reinit=True
            )
            wandb.config.model = args.model
            wandb.config.hidden_dim = args.hidden_dim
            wandb.config.conv_depth = args.conv_depth
            wandb.config.dropout_rate = args.dropout
            logger = run
        else:
            logger = BasicLogger()

        device = get_device()
        accuracy_function = Accuracy("multiclass", num_classes=dataset.num_classes).to(
            device
        )
        model = model_factory()
        final_stats = train_eval(
            model,
            training_config,
            train_loader,
            val_loader,
            test_loader,
            loss_function=cross_entropy,
            metric_function=accuracy_function,
            logger=logger,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        stats.append(final_stats)

        if not args.debug:
            run.finish()

    stats_df = pd.DataFrame(stats)
    mean_stats = stats_df.mean().to_dict()
    std_stats = stats_df.std().to_dict()

    print(
        f"Training completed, mean final stats:\n {mean_stats} \n std_devs:{std_stats}"
    )


if __name__ == "__main__":
    main()
