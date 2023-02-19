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

    def forward(self, x, edge_index, batch_index, node_rankings):
        return 0


class Control(nn.Module):
    """
    Base class for control, override _get_B for different strategies
    """

    def __init__(self, feature_dim, node_stat, k, normalise, alpha):
        super().__init__()

        self.node_stat = node_stat
        self.k = k
        self.normalise = normalise
        self.linear = nn.Linear(feature_dim, feature_dim)
        # currently learnable, does this actually help?
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def _get_B(self):
        raise NotImplementedError

    def _normalise_B(self, B):

        # get degrees of B
        D_B = torch.sparse.sum(B, dim=1).to_dense()

        # get 1/degrees
        D_B_inv = D_B ** -1

        # mutliply rows by 1/degrees, convert nans to zero
        B = torch.nan_to_num(B * D_B_inv.view(-1, 1), nan=0.0)

        return B

    def forward(self, x, edge_index, batch_index, node_rankings):

        x = self.linear(x)

        with torch.no_grad():

            # find nodes with ranking better than k (0 is best)
            active_nodes = (node_rankings[self.node_stat] <= self.k)

            # gets B matrix as per child class strategy
            B = self._get_B(edge_index, batch_index, active_nodes)

            if self.normalise:
                B = self._normalise_B(B)

        x = self.alpha * (B @ x)

        return x


class AdjacencyControl(Control):
    """
    Experiment 1 in our plan
    Excited node interacts only via exisiting edges (unidirectionally)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_B(self, edge_index, batch_index, active_nodes):

        # get (sparse) adjacency
        A = torch_geometric.utils.to_torch_coo_tensor(edge_index)

        # apply mask row-wise
        B = A * active_nodes

        return B


class DenseControl(Control):
    """
    Experiment 2 in our plan
    Excited node interacts with all other nodes (in same graph) (unidirectionally)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_B(self, edge_index, batch_index, active_nodes):

        # TODO process not fully sparse (some ops couldn't do sparse)

        # tile active_nodes into square matrix
        B = torch.tile(active_nodes, (active_nodes.shape[-1], 1))

        # TODO this is slow (has for loop) maybe possible without?
        # generate block diagonal mask (prevents edges between graphs in batch)
        graph_sizes = torch.unique_consecutive(batch_index, return_counts=True)[1]
        tensor_list = [torch.ones((graph_sizes[i], graph_sizes[i])) for i in range(graph_sizes.shape[0])]
        mask = torch.block_diag(*tensor_list)

        # apply mask element wise
        B = B * mask

        # zero out active node self adjacency
        torch.diagonal(B).zero_()

        return B.to_sparse_coo()


CONTROL_DICT = {"null": NullControl, "adj": AdjacencyControl, "dense": DenseControl}
