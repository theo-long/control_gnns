import random
import numpy as np

from training import train_eval, TrainConfig, BasicLogger
from data import (
    get_tu_dataset,
    generate_dataloaders,
    random_toy_graph,
    add_node_rankings,
)
from models import GCN, GraphMLP
from control import CONTROL_DICT

import argparse
import wandb

import torch
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

from torch_geometric.loader import DataLoader

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--time_inv", action="store_true")

    parser.add_argument("--control", default="null", type=str)
    parser.add_argument("--control_stat", default="degree", type=str)
    parser.add_argument("--control_k", default=1, type=int)
    parser.add_argument("--control_normalise", action="store_true")
    parser.add_argument("--control_alpha", default=-1.0, type=float)

    parser.add_argument("--hidden_dim", default=8, type=int)
    parser.add_argument("--num_encoding_layers", default=2, type=int)
    parser.add_argument("--num_decoding_layers", default=2, type=int)
    parser.add_argument("--num_conv_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    dataset = [add_node_rankings(random_toy_graph()) for _ in range(10)]

    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    model = GCN(
        input_dim=dataset[0].x.shape[1],
        output_dim=2,
        hidden_dim=args.hidden_dim,
        num_conv_layers=args.num_conv_layers,
        num_decoding_layers=args.num_decoding_layers,
        num_encoding_layers=args.num_encoding_layers,
        control_factory=CONTROL_DICT[args.control],
        control_stat=args.control_stat,
        control_k=args.control_k,
        control_normalise=args.control_normalise,
        control_alpha=args.control_alpha,
        dropout_rate=args.dropout,
        linear=args.linear,
        time_inv=args.time_inv,
    )

    for batch in dataloader:

        output = model(batch)


if __name__ == "__main__":
    main()