import torch
import torch.nn as nn

import torch_geometric
from torch_geometric.utils import spmm, scatter
from utils import get_device


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


class Control(nn.Module):
    """
    Base class for control, forward for different strategies (convolutional vs. message-passing)
    """

    def __init__(self, feature_dim, device=None):
        super().__init__()

        if device is None:
            self.device = get_device()
        else:
            self.device = device

    def forward(self, x, control_edge_index):
        raise NotImplementedError


class ConvolutionalControl(Control):
    """
    Convolutional layer : B H W
    """

    def __init__(self, feature_dim, normalize=True, **kwargs):
        super().__init__(feature_dim, **kwargs)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def normalize(self, control_edge_index):
        edge_weight = torch.ones(
            (control_edge_index.size(1),), device=control_edge_index.device
        )
        deg = scatter(edge_weight, control_edge_index[1], dim=0, reduce="sum")
        inv_deg = deg.pow(-1)
        inv_deg = inv_deg.masked_fill_(inv_deg == float("inf"), 0)
        edge_weight = inv_deg * edge_weight

        return control_edge_index, edge_weight

    def forward(self, x, control_edge_index):
        if self.normalize:
            control_edge_index, edge_weight = self.normalize(control_edge_index)

        x = self.linear(x)
        return spmm(control_edge_index, x)


class MessagePassingControl(Control):
    """
    Message passing control layer
    """

    def __init__(self, feature_dim, normalize=True, **kwargs):
        super().__init__(feature_dim, **kwargs)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, control_edge_index):
        pass
