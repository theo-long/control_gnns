import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Union
from training import train_eval, TrainConfig
from models import GraphMLP, GCN
from data import generate_dataloaders, get_tu_dataset, get_test_val_train_split
from control import CONTROL_DICT

import wandb
import torchmetrics
from torch.nn.functional import cross_entropy


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

    parser.add_argument("--control_type", default="null", type=str)
    parser.add_argument("--control_stat", default="degree", type=str)
    parser.add_argument("--control_k", default=1, type=int)
    parser.add_argument("--control_normalise", action="store_true")

    parser.add_argument("-t", "--time_inv", action="store_true", default=False)
    parser.add_argument("-l", "--linear", action="store_true", default=False)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--num_encoding_layers", default=2, type=int)
    parser.add_argument("--num_decoding_layers", default=2, type=int)
    parser.add_argument("--conv_depth", default=2, type=int)
    parser.add_argument("--dataset", default="PROTEINS")
    args = parser.parse_args()

    sweep_configuration = SWEEPS_DICT[args.model]
    sweep_configuration.name = args.name
    sweep_id = wandb.sweep(sweep=sweep_configuration.to_dict(), project="control_gnns")
    dataset = get_tu_dataset(args.dataset)
    splits = get_test_val_train_split(args.dataset, seed=0)
    train_loader, val_loader, test_loader = generate_dataloaders(
        dataset, splits, batch_size=128
    )

    if args.model == "mlp":
        model_factory = lambda dropout_rate: GraphMLP(
            input_dim=dataset[0].x.shape[1],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            num_decoding_layers=args.num_encoding_layers,
            num_encoding_layers=args.num_decoding_layers,
            dropout_rate=dropout_rate,  # passed during hyperparameter tuning
        )
    elif args.model == "gcn":
        control_factory = lambda: CONTROL_DICT[args.control_type](
            feature_dim=args.hidden_dim,
            node_stat=args.control_stat,
            k=args.control_k,
            normalise=args.control_normalise,
        )

        model_factory = lambda dropout_rate: GCN(
            input_dim=dataset[0].x.shape[1],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            conv_depth=args.conv_depth,
            control_factory=control_factory,
            num_decoding_layers=args.num_encoding_layers,
            num_encoding_layers=args.num_decoding_layers,
            dropout_rate=dropout_rate,  # passed during hyperparameter tuning
            linear=args.linear,
            time_inv=args.time_inv,
        )
    else:
        raise ValueError(f"Model name {args.model} not recognized")

    accuracy_function = torchmetrics.Accuracy(
        "multiclass", num_classes=dataset.num_classes
    )

    def single_training_run():
        run = wandb.init(project="control_gnns")
        hyperparameters = dict(wandb.config)
        training_config = TrainConfig(
            lr=hyperparameters.pop("lr"),
            batch_size=train_loader.batch_size,
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
        )
        return final_stats

    wandb.agent(sweep_id=sweep_id, function=single_training_run)


if __name__ == "__main__":
    main()
