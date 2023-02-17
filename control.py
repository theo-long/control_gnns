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

    def forward(self, x, edge_index, batch_index):
        return 0


class UnidirectionalAdjacencyControl(nn.Module):
    """
    Experiment 1 in our plan
    Excited node interacts with other nodes via exisiting edges (unidirectionally)
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.linear = nn.Linear(feature_dim, feature_dim)

        self.k = 1

    def _get_B(self, edge_index, batch_index):

        A = torch_geometric.utils.to_dense_adj(edge_index)[0]

        D = torch.sum(A, dim=1)

        _, sample_sizes = torch.unique_consecutive(batch_index, return_counts=True)

        batch_topk_nodes = []

        position = 0

        # iterate over every sample in batch and find topk_nodes
        for sample_size in sample_sizes:

            D_sample = D[position : position + sample_size]

            # need to handle multiplicity
            _, sample_topk_nodes = torch.topk(D, self.k)

            batch_topk_nodes.append(sample_topk_nodes)

            position += sample_size

        batch_topk_nodes = torch.cat(batch_topk_nodes)

        mask_vec = torch.zeros(A.shape[-1])
        mask_vec[batch_topk_nodes] = 1

        B = A * mask_vec

        return B

    def forward(self, x, edge_index, batch_index):

        x = self.linear(x)

        B = self._get_B(edge_index, batch_index)

        x = B @ x

        return x
