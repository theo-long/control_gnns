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


class Control(nn.Module):
    """
    Base class for control, override _get_B for different strategies
    """

    def __init__(self, feature_dim, node_stat, k):
        super().__init__()

        self.k = k
        self.node_stat = node_stat
        self.linear = nn.Linear(feature_dim, feature_dim)

    def _get_B(self, node_rankings):
        raise NotImplementedError
    
    def forward(self, x, edge_index, node_rankings):

        x = self.linear(x)

        # TODO handle multiple values? or at least track number of active nodes?

        # find nodes with ranking better than k (0 is best)
        active_nodes = node_rankings[self.node_stat] <= self.k

        # gets B matrix as per child class strategy
        B = self._get_B(edge_index, active_nodes)

        x = B @ x

        return x


class AdjacencyControl(Control):

    """
    Experiment 1 in our plan
    Excited node interacts only via exisiting edges (unidirectionally)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_B(self, edge_index, active_nodes):

        # TODO normalisation?

        # get (sparse) adjacency
        A = torch_geometric.utils.to_torch_coo_tensor(edge_index)

        # apply mask row-wise
        B = A * active_nodes

        return B


class DenseControl(Control):
    """
    Experiment 2 in our plan
    Excited node interacts with all other nodes (unidirectionally)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_B(self, edge_index, active_nodes):

        # TODO needs normalising
        # TODO not currently sparse

        active_nodes = active_nodes.to(torch.float).view(1, -1)

        # tile row into square matrix
        B = torch.tile(active_nodes, (len(active_nodes), 1))

        return B


CONTROL_DICT = {"null": NullControl, "adj": AdjacencyControl, "dense": DenseControl}
