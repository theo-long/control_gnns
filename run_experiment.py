from training import train_eval, TrainConfig, BasicLogger
from data import get_tu_dataset, generate_dataloaders, get_test_val_train_split
from models import GCN, GraphMLP
from control import CONTROL_DICT

import argparse
import wandb

import torch
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("-n", "--num_runs", default=1, type=int)

    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--time_inv", action="store_true")

    parser.add_argument("--control_type", default="null", type=str)
    parser.add_argument("--control_stat", default="degree", type=str)
    parser.add_argument("--control_k", default=1, type=int)
    parser.add_argument("--control_normalise", action="store_true")

    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_encoding_layers", default=2, type=int)
    parser.add_argument("--num_decoding_layers", default=2, type=int)
    parser.add_argument("--conv_depth", default=2, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()

    torch.random.manual_seed(0)

    training_config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    dataset = get_tu_dataset(args.dataset)
    splits = get_test_val_train_split(args.dataset, seed=0)
    train_loader, val_loader, test_loader = generate_dataloaders(
        dataset, splits, args.batch_size
    )

    if args.model.lower() == "gcn":

        control_factory = lambda: CONTROL_DICT[args.control_type](
            feature_dim=args.hidden_dim,
            node_stat=args.control_stat,
            k=args.control_k,
            normalise=args.control_normalise,
        )

        model_factory = lambda: GCN(
            input_dim=dataset[0].x.shape[1],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            conv_depth=args.conv_depth,
            num_decoding_layers=args.num_decoding_layers,
            num_encoding_layers=args.num_encoding_layers,
            control_factory=control_factory,
            dropout_rate=args.dropout,
            linear=args.linear,
            time_inv=args.time_inv,
        )
    elif args.model.lower() == "mlp":
        model_factory = lambda: GraphMLP(
            input_dim=dataset[0].x.shape[1],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            num_decoding_layers=args.num_decoding_layers,
            num_encoding_layers=args.num_encoding_layers,
            dropout_rate=args.dropout,
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
            wandb.config.num_encoding_layers = args.num_encoding_layers
            wandb.config.num_decoding_layers = args.num_decoding_layers
            wandb.config.dropout_rate = args.dropout
            logger = run
        else:
            logger = BasicLogger()

        accuracy_function = Accuracy("multiclass", num_classes=dataset.num_classes)
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
