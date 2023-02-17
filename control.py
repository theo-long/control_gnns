import torch
import torch.nn as nn


class NullControl(nn.Module):
    """
    Just returns 0
    Keeps GCNBlock code cleaner, by always having control 'active'
    """

    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.empty(0))

    def forward(self, x, edge_index):
        return 0
