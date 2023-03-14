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
from control import CONTROL_DICT
from utils import get_device, parse_callable_string

import wandb
import torchmetrics
from torch.nn.functional import cross_entropy
import torch


@dataclass
class DiscreteParameter:
    values: List[Any]


@dataclass
class ContinuousParameter:
    max: float
    min: float
    distribution: str


@dataclass
class SweepConfiguration:
    method: str
    name: str
    metric: Dict
    parameters: Dict[str, Union[DiscreteParameter, ContinuousParameter]]

    def to_dict(self):
        return {
            "method": self.method,
            "name": self.name,
            "metric": self.metric,
            "parameters": {k: asdict(v) for k, v in self.parameters.items()},
        }


default_sweep = SweepConfiguration(
    method="bayes",
    name="default",
    metric={"name": "best_val_loss", "goal": "minimize"},
    parameters={
        "lr": ContinuousParameter(0.1, 0.00001, distribution="log_uniform_values"),
        "weight_decay": ContinuousParameter(
            0.01, 1e-8, distribution="log_uniform_values"
        ),
        "beta1": ContinuousParameter(0.99, 0.5, distribution="log_uniform_values"),
        "dropout_rate": ContinuousParameter(0.5, 0.0, distribution="uniform"),
    },
)


SWEEPS_DICT = {
    "mlp": default_sweep,
    "gcn": default_sweep,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-e", "--epochs", default=20, type=int)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--batch_size", default=128, type=int)

    parser.add_argument(
        "--control_type", default="null", type=str, choices=["null", "gcn", "mp"]
    )
    parser.add_argument(
        "--control_edges", default="adj", type=str, choices=["adj", "dense"]
    )
    parser.add_argument(
        "--control_metric",
        default="b_centrality",
        type=str,
        choices=["degree", "b_centrality"],
    )
    parser.add_argument("--control_k", default=lambda x: 1, type=parse_callable_string)
    parser.add_argument("--control_self_adj", action="store_true")

    parser.add_argument("-t", "--time_inv", action="store_true", default=False)
    parser.add_argument("-l", "--linear", action="store_true", default=False)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--conv_depth", default=2, type=int)

    parser.add_argument("--dataset", default="PROTEINS")

    parser.add_argument(
        "--norm", default=None, choices=[None, "batchnorm", "layernorm"]
    )

    args = parser.parse_args()

    sweep_configuration = SWEEPS_DICT[args.model]
    sweep_configuration.name = args.name
    sweep_id = wandb.sweep(sweep=sweep_configuration.to_dict(), project="control_gnns")

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
        train_mask, val_mask, test_mask = get_test_val_train_mask(dataset)
    else:
        train_loader, val_loader, test_loader = generate_dataloaders(
            dataset, args.dataset, args.batch_size
        )
        train_mask, val_mask, test_mask = None, None, None

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
        )
    else:
        raise ValueError(f"Model name {args.model} not recognized")

    device = get_device()
    accuracy_function = torchmetrics.Accuracy(
        "multiclass", num_classes=dataset.num_classes
    ).to(device)

    def single_training_run():
        run = wandb.init(project="control_gnns")
        hyperparameters = dict(wandb.config)
        training_config = TrainConfig(
            lr=hyperparameters.pop("lr"),
            batch_size=1 if is_node_classifier else train_loader.batch_size,
            epochs=args.epochs,
            weight_decay=hyperparameters.pop("weight_decay"),
            beta1=hyperparameters.pop("beta1"),
        )
        model = model_factory(**hyperparameters)
        final_stats = train_eval(
            model,
            training_config,
            train_loader,
            val_loader,
            test_loader,
            loss_function=cross_entropy,
            metric_function=accuracy_function,
            logger=wandb,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        return final_stats

    wandb.agent(sweep_id=sweep_id, function=single_training_run)


if __name__ == "__main__":
    main()
