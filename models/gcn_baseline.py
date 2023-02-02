from torch import nn


class LinearGCN(nn.Module):
    """A basic GNN where all the convolutional layers are linear i.e. no activations.
    The only non-linearity is in the input encoder and decoder."""

    def __init__(self) -> None:
        super().__init__()


class BasicGCN(nn.Module):
    """The original GCN architecture from Kipf and Welling."""

    def __init__(self) -> None:
        super().__init__()
