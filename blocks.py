from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from control import NullControl, ControlGCNConv


class MLPBlock(nn.Module):
    """
    ReLU MLP with dropout, can be used as encoder or decoder
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        num_layers: int = 2,
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
    a block of GCN layers with (optional) control
    has flags to make linear and / or time invariant
    """

    def __init__(
        self,
        feature_dim: int,
        depth: int,
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
            self.control = nn.ModuleList([ControlGCNConv(feature_dim, feature_dim)])

        else:
            self.conv = nn.ModuleList(
                [GCNConv(feature_dim, feature_dim) for _ in range(depth)]
            )
            self.control = nn.ModuleList([ControlGCNConv(feature_dim, feature_dim) for _ in range(depth)])

    def forward(self, x, edge_index, control_edge_index):

        # handles both time_inv = True and time_inv = False
        for i in range(self.depth):

            conv = self.conv[i % len(self.conv)]
            control = self.control[i % len(self.control)]

            x = conv(x, edge_index) + control(x, control_edge_index)

            # no dropout or (optional) relu after final conv
            if i != (self.depth - 1):
                if not self.linear:
                    x = F.relu(x)

                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x
