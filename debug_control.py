import argparse
import random
import numpy as np

from data import ToyDataset, RankingTransform, ControlTransform
from models import GCN, GraphMLP

import torch
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
from torch_geometric.loader import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--time_inv", action="store_true")

    parser.add_argument("--control_type", default="null", type=str)
    parser.add_argument("--control_edges", default="adj", type=str)
    parser.add_argument("--control_metric", default="degree", type=str)
    parser.add_argument("--control_k", default=1, type=int)

    parser.add_argument("--hidden_dim", default=8, type=int)
    parser.add_argument("--conv_depth", default=2, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    torch.set_printoptions(linewidth=320)

    if args.control_type != "null":
        transform = ControlTransform(
            args.control_edges, args.control_metric, args.control_k
        )
    else:
        transform = None

    pre_transform = RankingTransform()

    dataset = ToyDataset(transform, pre_transform)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = GCN(
        input_dim=dataset[0].x.shape[1],
        output_dim=2,
        hidden_dim=args.hidden_dim,
        conv_depth=args.conv_depth,
        dropout_rate=args.dropout,
        linear=args.linear,
        time_inv=args.time_inv,
        control_type=args.control_type
    )

    for batch in dataloader:
        output = model(batch)

if __name__ == "__main__":
    main()
