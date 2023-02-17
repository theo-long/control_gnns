import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Union

import wandb


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
        "dropout_rate": ContinuousParameter(0.5, 0.0, distribution="uniform"),
    },
)


SWEEPS_DICT = {"mlp": mlp_sweep}


def single_training_run():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("-m", "--max_runs", default=10)
    args = parser.parse_args()

    sweep_configuration = SWEEPS_DICT[args.name]
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="control_gnns")

    wandb.agent(sweep_id=sweep_id, function=single_training_run, count=args.max_runs)


if __name__ == "__main__":
    main()
