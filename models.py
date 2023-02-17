import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from blocks import MLPBlock, GCNBlock, GCNBlockTimeInv


class GCN(nn.Module):
    """
    The original GCN architecture from Kipf and Welling, with additional node encoding/decoding MLP layers.
    + Optional arguments for linearity and time-invarince
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_conv_layers: int,
        num_encoding_layers: int,
        num_decoding_layers: int,
        dropout_rate: float,
        linear: bool = False,
        time_inv: bool = False,
    ):
        super().__init__()

        self.encoder = MLPBlock(
            input_dim, hidden_dim, hidden_dim, num_encoding_layers, dropout_rate
        )

        gcn_block_factory = GCNBlockTimeInv if time_inv else GCNBlock
        self.gcn_block = gcn_block_factory(
            hidden_dim, num_conv_layers, dropout_rate, linear
        )

        self.decoder = MLPBlock(
            hidden_dim, output_dim, hidden_dim, num_decoding_layers, dropout_rate
        )

    def forward(self, data):

        x = data.x

        x = self.encoder(x)

        x = self.gcn_block(x, data.edge_index, data.batch)

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
        num_encoding_layers: int,
        num_decoding_layers: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.dropout_rate = dropout_rate

        self.encoder = MLPBlock(
            input_dim, hidden_dim, hidden_dim, num_encoding_layers, dropout_rate
        )

        self.decoder = MLPBlock(
            hidden_dim, output_dim, hidden_dim, num_decoding_layers, dropout_rate
        )

    def forward(self, data):

        x = data.x

        x = self.encoder(x)

        x = global_add_pool(x, data.batch)

        x = self.decoder(x)

        return x
