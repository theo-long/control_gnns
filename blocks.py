import torch 
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from control import NullControl


class MLPBlock(nn.Module):
    """
    ReLU MLP with dropout, can be used as encoder and decoder
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        dropout_rate,
    ):
        super().__init__()

        assert num_layers >= 2, "an MLP needs at least 2 layers"

        self.dropout_rate = dropout_rate

        layers = [nn.Linear(input_dim, hidden_dim)]

        # only enters loop if num_layers >= 3
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # no relu or dropout after last linear layer
        x = self.layers[-1](x)

        return x


class GCNBlock(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_layers,
        dropout_rate,
        linear=False,
        control_factory=NullControl,
    ):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.linear = linear

        convs = []
        controls = []

        for _ in range(num_layers):
            convs.append(GCNConv(feature_dim, feature_dim))
            controls.append(control_factory())

        self.convs = nn.ModuleList(convs)
        self.controls = nn.ModuleList(controls)


    def forward(self, x, edge_index):

        for conv, control in zip(self.convs, self.controls):
            x = conv(x, edge_index) + control(x, edge_index)

            if not self.linear:
                x = F.relu(x)

            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x


class GCNBlockTimeInv(nn.Module):
    def __init__(
        self,
        feature_dim,
        depth,
        dropout_rate,
        linear=False,
        control_factory=NullControl,
    ):
        super().__init__()

        self.depth = depth
        self.dropout_rate = dropout_rate
        self.linear = linear

        self.conv = GCNConv(feature_dim, feature_dim)
        self.control = control_factory()

    def forward(self, x, edge_index):

        for _ in range(self.depth):
            x = self.conv(x, edge_index) + self.control(x, edge_index)

            if not self.linear:
                x = F.relu(x)

            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x
