from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MLPBlock(nn.Module):
    """
    ReLU MLP with dropout, can be used as encoder or decoder
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super().__init__()

        assert num_layers >= 2, "an MLP needs at least 2 layers"

        self.dropout_rate = dropout_rate

        layers = [nn.Linear(input_dim, hidden_dim)]

        # only enters loop if num_layers >= 3
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # no relu or dropout after last linear layer
        x = self.layers[-1](x)

        return x


class GCNBlock(nn.Module):
    """
    A block of TIME VARYING GCN layers with option for control.
    Control defaults to NullControl which returns 0 (so has no effect)
    """

    def __init__(
        self,
        feature_dim: int,
        depth: int,
        control_factory: Callable,
        dropout_rate: float,
        linear: bool,
        time_inv: bool,
    ):
        super().__init__()

        self.depth = depth
        self.dropout_rate = dropout_rate
        self.linear = linear
        self.time_inv = time_inv

        if self.time_inv:
            self.conv = nn.ModuleList([GCNConv(feature_dim, feature_dim)])
            self.control = nn.ModuleList([control_factory()])

        else:
            self.conv = nn.ModuleList([GCNConv(feature_dim, feature_dim) for _ in range(depth)])
            self.control = nn.ModuleList([control_factory() for _ in range(depth)])

    def forward(self, x, edge_index, batch_index, node_rankings):

        for i in range(self.depth):

            # if module lists contain multiple modules, iterate over them
            # otherwise just return the only layer

            conv = self.conv[i % len(self.conv)]
            control = self.control[i % len(self.control)]

            x = conv(x, edge_index) + control(x, edge_index, batch_index, node_rankings)

            if not self.linear:
                x = F.relu(x)

            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x
