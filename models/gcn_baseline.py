from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool


class LinearGCN(nn.Module):
    """A basic GNN where all the convolutional layers are linear i.e. no activations.
    The only non-linearity is in the input encoder and decoder."""

    def __init__(self) -> None:
        super().__init__()


class BasicGCN(nn.Module):
    """The original GCN architecture from Kipf and Welling."""

    def __init__(
        self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate
    ) -> None:
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        conv_layers = []
        for i in range(num_layers - 1):
            conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        conv_layers.append(GCNConv(hidden_dim, output_dim))
        self.conv_layers = nn.ModuleList(conv_layers)

        self.dropout_rate = dropout_rate

    def forward(self, data):
        x = self.embedding(data.x)

        for layer in self.conv_layers[:-1]:
            x = layer(x, data.edge_index)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv_layers[-1](x, data.edge_index)
        yhat = global_add_pool(x, data.batch)

        return yhat
