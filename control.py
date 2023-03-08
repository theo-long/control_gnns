import torch
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter


class NullControl(nn.Module):
    """
    Just returns 0
    Keeps GCNBlock code cleaner, by always having control 'active' (less logic)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # nn.Module needs a parameter
        self.parameter = nn.Parameter(torch.empty(0))

    def forward(self, x, control_edge_index):
        return 0


class ControlGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels, add_self_loops=False, normalize=False)

    def _row_normalize(self, edge_index):

        edge_weight = torch.ones(
            (edge_index.size(1),), device=edge_index.device
        )
    
        # theo had index[1]
        deg = scatter(edge_weight, edge_index[0], dim=0, reduce='sum')
        deg_inv = deg.pow_(-1.0)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        edge_weight = deg_inv * edge_weight

        return edge_index, edge_weight

    def forward(self, x, edge_index):
        edge_index, edge_weight = self._row_normalize(edge_index)
        return self.conv(x, edge_index, edge_weight)
