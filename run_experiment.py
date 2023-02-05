from training import train_eval, TrainConfig, BasicLogger
from data import get_tu_dataset, generate_dataloaders
from models import BasicGCN

import argparse
import wandb
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()

    training_config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
    )

    dataset = get_tu_dataset(args.dataset)
    train_loader, val_loader, test_loader = generate_dataloaders(
        dataset, args.batch_size
    )

    if args.model.lower() == "gcn":
        model = BasicGCN(
            input_dim=dataset[0].x.shape[0],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    else:
        raise ValueError(f"Model name {args.model} not recognized")

    if not args.debug:
        wandb.init(project="control_gnns", config=TrainConfig)
        wandb.config.model = args.model
        wandb.config.hidden_dim = args.hidden_dim
        wandb.config.num_layers = args.num_layers
        wandb.config.dropout_rate = args.dropout
        logger = wandb
    else:
        logger = BasicLogger()

    accuracy_function = Accuracy("multiclass", num_classes=dataset.num_classes)
    train_eval(
        model,
        training_config,
        train_loader,
        val_loader,
        test_loader,
        loss_function=cross_entropy,
        metric_function=accuracy_function,
        logger=logger,
    )


if __name__ == "__main__":
    main()
