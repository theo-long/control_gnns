import torch
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import scatter


class ControlGCNConv(nn.Module):
    """
    wraps a GCNConv layer with asymmetric (in-degree) normalization
    """

    def __init__(self, channels):
        super().__init__()

        self.conv = GCNConv(channels, channels, add_self_loops=False, normalize=False)

    def _normalize(self, edge_index):

        # get inverse in-degree
        deg = torch_geometric.utils.degree(edge_index[1])
        deg_inv = deg.pow(-1.0)
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)

        # edge weights are simply the inverse degree
        # of the receiving node
        edge_weight = deg_inv[edge_index[1]]

        return edge_index, edge_weight

    def forward(self, x, edge_index):
        edge_index, edge_weight = self._normalize(edge_index)
        return self.conv(x, edge_index, edge_weight)


class ControlMP(MessagePassing):
    """
    adapted from practical 2 codebase
    """

    def __init__(self, channels, norm=nn.BatchNorm1d, aggr="add", control_init=None):
        super().__init__(aggr=aggr)

        self.mlp_msg = nn.Sequential(
            norm(2 * channels),
            nn.Linear(2 * channels, channels),
            nn.ReLU(),
            norm(channels),
            nn.Linear(channels, channels),
        )

        final_norm = norm(channels)

        if control_init is not None:
            nn.init.constant_(final_norm.weight, control_init)

        self.mlp_upd = nn.Sequential(
            norm(2 * channels),
            nn.Linear(2 * channels, channels),
            nn.ReLU(),
            norm(channels),
            nn.Linear(channels, channels),
            final_norm,
        )

    def forward(self, h, edge_index):
        out = self.propagate(edge_index, h=h)
        return out

    def message(self, h_i, h_j):
        return self.mlp_msg(torch.cat([h_i, h_j], dim=-1))

    def update(self, aggr_out, h):
        return self.mlp_upd(torch.cat([h, aggr_out], dim=-1))

CONTROL_DICT = {"gcn": ControlGCNConv, "mp": ControlMP}
