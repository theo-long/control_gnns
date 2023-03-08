from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from blocks import MLPBlock, GCNBlock


class GCN(nn.Module):
    """
    The original GCN architecture from Kipf and Welling, with additional node encoding/decoding MLP layers.
    + Optional arguments for linearity, time-invarince, and control modules
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        conv_depth: int,
        dropout_rate: float,
        linear: bool,
        time_inv: bool,
        use_control: bool,
    ):
        super().__init__()

        self.use_control = use_control

        self.encoder = MLPBlock(input_dim, hidden_dim, hidden_dim, dropout_rate)

        self.gcn_block = GCNBlock(
            hidden_dim, conv_depth, dropout_rate, linear, time_inv, use_control
        )

        self.decoder = MLPBlock(hidden_dim, output_dim, hidden_dim, dropout_rate)

    def forward(self, data):

        x = data.x

        x = self.encoder(x)

        if self.use_control:
            x = self.gcn_block(x, data.edge_index, data.control_edge_index)
        else:
            x = self.gcn_block(x, data.edge_index)

        x = global_add_pool(x, data.batch)

        x = self.decoder(x)

        return x


class GraphMLP(nn.Module):
    """
    An MLP Baseline which has no convolutional layers.
    It simply encodes the node features, applies graph sum pooling, then decodes the graph-level features.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.encoder = MLPBlock(input_dim, hidden_dim, hidden_dim, dropout_rate)

        self.decoder = MLPBlock(hidden_dim, output_dim, hidden_dim, dropout_rate)

    def forward(self, data):

        x = data.x

        x = self.encoder(x)

        x = global_add_pool(x, data.batch)

        x = self.decoder(x)

        return x
