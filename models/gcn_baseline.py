from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool


class BasicGCN(nn.Module):
    """The original GCN architecture from Kipf and Welling, with additional node encoding/decoding MLP layers."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_conv_layers,
        num_encoding_layers=1,
        num_decoding_layers=1,
        dropout_rate=0.0,
    ) -> None:
        super().__init__()

        encoding_layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_encoding_layers - 1):
            encoding_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.encoding_layers = nn.ModuleList(encoding_layers)

        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.conv_layers = nn.ModuleList(conv_layers)

        decoding_layers = []
        for _ in range(num_decoding_layers - 1):
            decoding_layers.append(nn.Linear(hidden_dim, hidden_dim))
        decoding_layers.append(nn.Linear(hidden_dim, output_dim))
        self.decoding_layers = nn.ModuleList(decoding_layers)

        self.dropout_rate = dropout_rate

    def forward(self, data):
        x = data.x

        for layer in self.encoding_layers:
            x = layer(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        for layer in self.conv_layers:
            x = layer(x, data.edge_index)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_add_pool(x, data.batch)

        for layer in self.decoding_layers[:-1]:
            x = layer(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        yhat = self.decoding_layers[-1](x)

        return yhat


class LinearGCN(BasicGCN):
    """A basic GCN where all the convolutional layers are linear i.e. no activation functions.
    The only non-linearity is in the input encoder and decoder."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_conv_layers,
        num_encoding_layers,
        num_decoding_layers,
        dropout_rate,
    ) -> None:
        super().__init__(
            input_dim,
            output_dim,
            hidden_dim,
            num_conv_layers,
            num_encoding_layers,
            num_decoding_layers,
            dropout_rate,
        )

    def forward(self, data):
        x = data.x

        for layer in self.encoding_layers:
            x = layer(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        for layer in self.conv_layers:
            # Note that we have no relu activations in this case
            x = layer(x, data.edge_index)
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_add_pool(x, data.batch)

        for layer in self.decoding_layers[:-1]:
            x = layer(x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        yhat = self.decoding_layers[-1](x)

        return yhat


class GraphMLP(BasicGCN):
    """An MLP Baseline which has no convolutional layers.
    It simply encodes the node features, applies graph sum pooling, then decodes the graph-level features.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_encoding_layers=1,
        num_decoding_layers=1,
        dropout_rate=0.0,
    ) -> None:
        super().__init__(
            input_dim,
            output_dim,
            hidden_dim,
            num_encoding_layers,
            num_decoding_layers,
            dropout_rate,
            num_conv_layers=0,
        )
