import torch
import torch.nn as nn

import torch_geometric


class NullControl(nn.Module):
    """
    Just returns 0
    Keeps GCNBlock code cleaner, by always having control 'active'
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.parameter = nn.Parameter(torch.empty(0))

    def forward(self, x, edge_index, node_rankings):
        return 0


class AdjacencyControl(nn.Module):
    """
    Experiment 1 in our plan
    Excited node interacts only via exisiting edges (unidirectionally)
    """

    def __init__(self, feature_dim, node_stat, k):
        super().__init__()

        self.k = k
        self.node_stat = node_stat

        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, edge_index, node_rankings):

        x = self.linear(x)

        # get (sparse) adjacency
        A = torch_geometric.utils.to_torch_coo_tensor(edge_index)

        # find nodes with ranking better than k (0 is best)
        node_mask = node_rankings[self.node_stat] <= self.k

        # apply mask row-wise
        B = A * node_mask

        x = B @ x

        return x


class DenseControl(nn.Module):
    """
    Experiment 2 in our plan
    Excited node interacts with all other nodes (unidirectionally)
    """

    def __init__(self, feature_dim, node_stat, k):
        super().__init__()

        self.k = k
        self.node_stat = node_stat

        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, edge_index, node_rankings):

        x = self.linear(x)

        # find nodes with ranking better than k (0 is best)
        row_active = (node_rankings[self.node_stat] <= self.k).to(torch.float).view(1, -1)

        # tile row into square matrix
        B = torch.tile(row_active, (x.shape[0], 1))

        x = B @ x

        return x


CONTROL_DICT = {
        'null' : NullControl,
        'adj' : AdjacencyControl,
        'dense' : DenseControl
        }
