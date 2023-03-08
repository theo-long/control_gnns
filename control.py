import torch
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter


class ControlGCNConv(nn.Module):
    """
    wraps a GCNConv layer with asymmetric (in-degree)  normalization
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = GCNConv(
            in_channels, out_channels, add_self_loops=False, normalize=False
        )

    def _normalize(self, edge_index):

        # get inverse degree
        deg = torch_geometric.utils.degree(edge_index[0])
        deg_inv = deg.pow(-1.0)
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)

        # edge weights are simply the inverse degree
        # of the receiving node
        edge_weight = deg_inv[edge_index[0]]

        return edge_index, edge_weight

    def forward(self, x, edge_index):
        edge_index, edge_weight = self._normalize(edge_index)
        return self.conv(x, edge_index, edge_weight)
