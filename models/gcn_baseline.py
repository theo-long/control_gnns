from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool


class BasicMLP(nn.Module):
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
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        # no relu or dropout after last linear layer
        x = self.layers[-1](x)

        return x


class GCN(nn.Module):
    """
    The original GCN architecture from Kipf and Welling, with additional node encoding/decoding MLP layers.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_conv_layers,
        num_encoding_layers,
        num_decoding_layers,
        dropout_rate,
        linear=False,
    ):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.linear = linear

        self.encoder = BasicMLP(
            input_dim, hidden_dim, hidden_dim, num_encoding_layers, dropout_rate
        )

        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.conv_layers = nn.ModuleList(conv_layers)

        self.decoder = BasicMLP(
            hidden_dim, output_dim, hidden_dim, num_encoding_layers, dropout_rate
        )

    def forward(self, data):

        x = data.x

        x = self.encoder(x)

        for layer in self.conv_layers:
            x = layer(x, data.edge_index)

            if not self.linear:
                x = nn.functional.relu(x)

            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_add_pool(x, data.batch)

        x = self.decoder(x)

        return x


class GraphMLP(nn.Module):
    """
    An MLP Baseline which has no convolutional layers.
    It simply encodes the node features, applies graph sum pooling, then decodes the graph-level features.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_encoding_layers,
        num_decoding_layers,
        dropout_rate,
    ):

        super().__init__()

        self.dropout_rate = dropout_rate

        self.encoder = BasicMLP(
            input_dim, hidden_dim, hidden_dim, num_encoding_layers, dropout_rate
        )

        self.decoder = BasicMLP(
            hidden_dim, output_dim, hidden_dim, num_encoding_layers, dropout_rate
        )

    def forward(self, data):

        x = data.x

        x = self.encoder(x)

        x = global_add_pool(x, data.batch)

        x = self.decoder(x)

        return x
