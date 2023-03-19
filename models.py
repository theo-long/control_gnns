from typing import Callable, Optional

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
        control_type: str,
        is_node_classifier: bool,
        residual: bool,
        norm: Optional[Callable] = None,
        **control_kwargs,
    ):
        super().__init__()

        self.is_node_classifier = is_node_classifier

        self.control_type = control_type

        self.encoder = MLPBlock(
            input_dim, hidden_dim, hidden_dim, dropout_rate, norm=norm
        )

        if control_type == "mp":
            control_kwargs["norm"] = norm

        self.gcn_block = GCNBlock(
            hidden_dim,
            conv_depth,
            dropout_rate,
            linear,
            time_inv,
            residual,
            control_type,
            **control_kwargs,
        )

        self.decoder = MLPBlock(
            hidden_dim, output_dim, hidden_dim, dropout_rate, norm=norm
        )

    def forward(self, data):

        x = data.x

        x = self.encoder(x)

        if self.control_type == "null":
            x = self.gcn_block(x, data.edge_index)
        else:
            x = self.gcn_block(x, data.edge_index, data.control_edge_index)

        if not self.is_node_classifier:
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
        is_node_classifier: bool = False,
        norm: Optional[Callable] = None,
    ):
        super().__init__()

        self.is_node_classifier = is_node_classifier

        self.encoder = MLPBlock(input_dim, hidden_dim, hidden_dim, dropout_rate, norm)

        self.decoder = MLPBlock(hidden_dim, output_dim, hidden_dim, dropout_rate, norm)

    def forward(self, data):

        x = data.x

        x = self.encoder(x)

        if not self.is_node_classifier:
            x = global_add_pool(x, data.batch)

        x = self.decoder(x)

        return x
