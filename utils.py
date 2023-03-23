import os
import csv
from abc import ABC, abstractmethod
import logging
from typing import Union, Tuple

import torch
import numpy as np

from torch_sparse import SparseTensor
from torch import Tensor

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class Logger(ABC):
    """Abstract Logger Class"""

    @abstractmethod
    def log(message):
        pass


class CSVLogger(Logger):
    """Basic CSV Logger"""

    def __init__(self, filename, cols, const=None) -> None:
        assert not os.path.isfile(filename)
        self.logger = logger
        self.filename = filename
        self.cols = cols
        self.const = const

        with open(self.filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(self.cols)

    def update_const(self, const):
        self.const = const

    def log(self, message):
        if isinstance(message, dict):

            parsed_message = []

            for col in self.cols:

                if col in message.keys():
                    parsed_message.append(message[col])
                else:
                    parsed_message.append(self.const[col])

            message = parsed_message

        with open(self.filename, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(message)


class BasicLogger(Logger):
    """Basic Command Line Logger"""

    def __init__(self) -> None:
        self.logger = logger

    def log(self, message):
        if isinstance(message, dict):
            message = ", ".join([f"{k}:{v}" for k, v in message.items()])
        self.logger.info(message)


class PrintLogger(Logger):
    """Logger that just uses print method"""

    def log(self, message):
        if isinstance(message, dict):
            message = ", ".join([f"{k}:{v}" for k, v in message.items()])
        print(message)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


CALLABLE_DICT = {
    "log": np.log,
    "sqrt": np.sqrt,
}


def parse_callable_string(callable_str: str):
    if callable_str.isdigit():
        return lambda n: int(callable_str)
    else:
        try:
            val = float(callable_str)
            return lambda n: val * n
        except ValueError:
            pass

        try:
            callable = CALLABLE_DICT[callable_str]
            return callable
        except KeyError:
            raise ValueError(
                f"Invalid callable string {callable_str} passed to parser."
            )


def ptr2index(ptr: Tensor) -> Tensor:
    ind = torch.arange(ptr.numel() - 1, dtype=ptr.dtype, device=ptr.device)
    return ind.repeat_interleave(ptr[1:] - ptr[:-1])


def to_edge_index(adj: Union[Tensor, SparseTensor]) -> Tuple[Tensor, Tensor]:
    r"""Converts a :class:`torch.sparse.Tensor` or a
    :class:`torch_sparse.SparseTensor` to edge indices and edge attributes.
    Args:
        adj (torch.sparse.Tensor or SparseTensor): The adjacency matrix.
    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> adj = to_torch_coo_tensor(edge_index)
        >>> to_edge_index(adj)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([1., 1., 1., 1., 1., 1.]))
    """
    if isinstance(adj, SparseTensor):
        row, col, value = adj.coo()
        if value is None:
            value = torch.ones(row.size(0), device=row.device)
        return torch.stack([row, col], dim=0).long(), value

    if adj.layout == torch.sparse_coo:
        return adj.indices().detach().long(), adj.values()

    if adj.layout == torch.sparse_csr:
        row = ptr2index(adj.crow_indices().detach())
        col = adj.col_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    if adj.layout == torch.sparse_csc:
        col = ptr2index(adj.ccol_indices().detach())
        row = adj.row_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    raise ValueError(f"Expected sparse tensor layout (got '{adj.layout}')")
