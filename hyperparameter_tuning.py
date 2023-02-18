import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Union
from training import train_eval, TrainConfig
from models import GraphMLP, GCN
from data import generate_dataloaders, get_tu_dataset

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


mlp_sweep = SweepConfiguration(
    method="random",
    name="mlp",
    metric={"name": "val_loss", "goal": "minimize"},
    parameters={
        "lr": ContinuousParameter(0.1, 0.00001, distribution="log_uniform_values"),
        "weight_decay": ContinuousParameter(
            0.01, 1e-7, distribution="log_uniform_values"
        ),
        "beta1": ContinuousParameter(0.5, 0.99, distribution="log_uniform_values"),
        "dropout_rate": ContinuousParameter(0.5, 0.0, distribution="uniform"),
    },
)

gcn_sweep = SweepConfiguration(
    method="random",
    name="gcn",
    metric={"name": "val_loss", "goal": "minimize"},
    parameters={
        "lr": ContinuousParameter(0.1, 0.00001, distribution="log_uniform_values"),
        "weight_decay": ContinuousParameter(
            0.01, 1e-7, distribution="log_uniform_values"
        ),
        "beta1": ContinuousParameter(0.99, 0.5, distribution="log_uniform_values"),
        "dropout_rate": ContinuousParameter(0.5, 0.0, distribution="uniform"),
    },
)


SWEEPS_DICT = {
    "mlp": mlp_sweep,
    "gcn": gcn_sweep,
}


def training_run_factory(model_factory, epochs: int, dataset, batch_size=128):
    """Wrapper function to generate a function that trains a single model.
    Needed so we can dynamically generate a target function for different datasets.

    Args:
        model_factory: creates the model object
        epochs (int): number of epochs to train for
        dataset: dataset to train on
        batch_size (int, optional): batch size. Defaults to 128.
    """
    train_loader, val_loader, test_loader = generate_dataloaders(dataset, batch_size)

    def single_training_run():
        run = wandb.init(project="control_gnns")
        hyperparameters = dict(wandb.config)
        training_config = TrainConfig(
            lr=hyperparameters.pop("lr"),
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=hyperparameters.pop("weight_decay"),
            beta1=hyperparameters.pop("beta1"),
        )
        accuracy_function = torchmetrics.Accuracy(
            "multiclass", num_classes=dataset.num_classes
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

    return single_training_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=20)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-t", "--time_inv", action="store_true", default=False)
    parser.add_argument("-l", "--linear", action="store_true", default=False)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--num_encoding_layers", default=2, type=int)
    parser.add_argument("--num_decoding_layers", default=2, type=int)
    parser.add_argument("--num_conv_layers", default=2, type=int)
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("-n", "--num_runs", default=10)
    args = parser.parse_args()

    sweep_configuration = SWEEPS_DICT[args.model]
    sweep_id = wandb.sweep(sweep=sweep_configuration.to_dict(), project="control_gnns")
    dataset = get_tu_dataset(args.dataset)

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
        model_factory = lambda dropout_rate: GCN(
            input_dim=dataset[0].x.shape[1],
            output_dim=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            num_conv_layers=args.num_conv_layers,
            num_decoding_layers=args.num_encoding_layers,
            num_encoding_layers=args.num_decoding_layers,
            dropout_rate=dropout_rate,  # passed during hyperparameter tuning
            linear=args.linear,
            time_inv=args.time_inv,
        )
    else:
        raise ValueError(f"Model name {args.model} not recognized")

    training_function = training_run_factory(
        epochs=args.epochs, model_factory=model_factory, dataset=dataset
    )

    wandb.agent(sweep_id=sweep_id, function=training_function, count=args.num_runs)


if __name__ == "__main__":
    main()
