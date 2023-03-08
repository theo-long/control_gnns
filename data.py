import math
from typing import Callable, Any, Optional

import torch
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


import networkx as nx
from scipy import stats

import pathlib

SPLITS_LOC = pathlib.Path(__file__).parent / "test_train_splits"


def _get_active_nodes(n_control, control_metric, node_rankings):
    return node_rankings[control_metric] <= n_control


def _generate_control_adjacency(edge_index, active_nodes, control_type):

    if control_type == "adj":
        # get (sparse) adjacency
        A = torch_geometric.utils.to_torch_coo_tensor(edge_index)

        # apply mask row-wise
        B = (A * active_nodes).to_dense()

    elif control_type == "dense":
        B = (active_nodes * 1).repeat(len(active_nodes), 1)

    else:
        raise ValueError("Unrecognized control type, must be adj or dense")

    return B.nonzero().T


class ControlTransform(BaseTransform):
    def __init__(
        self, control_metric: str, control_type: str, n_control_nodes: Callable
    ) -> None:
        super().__init__()
        self.control_metric = control_metric
        self.control_type = control_type
        self.n_control_nodes = n_control_nodes

    def __call__(self, data: Data) -> Data:
        #data.n_control = int(self.n_control_nodes(data.x.shape[0]))
        data.n_control = int(self.n_control_nodes)
        active_nodes = _get_active_nodes(
            data.n_control, self.control_metric, data.node_rankings
        )
        data.control_edge_index = _generate_control_adjacency(
            data.edge_index, active_nodes, self.control_type
        )
        return data


def degree(data: Data):
    "wraps nx.degree to return flat tensor"

    graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
    degree = nx.degree(graph)
    return torch.tensor(list(degree))[:, 1].flatten()


def betweenness_centrality(data: Data):
    "wraps nx.betweenness_centrality to return flat tensor"

    graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
    between_cent = nx.betweenness_centrality(graph)
    return torch.tensor(list(between_cent.values()))


def node_rankings(data: Data, stat_func: Callable):
    "finds node rankings per stat_func"

    node_stat = stat_func(data).numpy()

    # ranks (negative of) node_stat 'competition style'
    stat_rankings = stats.rankdata(-node_stat, method="min")

    # convert to tensor
    stat_rankings = torch.tensor(stat_rankings, dtype=torch.int32)

    return stat_rankings


def add_node_rankings(data: Data):
    """
    used as a pre_transform for TUDataset, adds node ranking field to (graph) Data objects
    NOTE: any changes will not take effect without first deleting datasets/PROTEINS/processed
    """

    # adds new fields for node rankings
    degree_rankings = node_rankings(data, degree)
    between_cent_rankings = node_rankings(data, betweenness_centrality)

    data.node_rankings = {
        "degree": degree_rankings,
        "b_centrality": between_cent_rankings,
    }

    return data


def get_tu_dataset(
    name,
    control_metric: Optional[str] = None,
    control_type: Optional[str] = None,
    n_control_nodes: Optional[Callable] = None,
):
    if control_metric is not None:
        control_transform = ControlTransform(
            control_metric, control_type, n_control_nodes
        )
    else:
        control_transform = None
    dataset = TUDataset(
        root="./datasets",
        name=name,
        use_node_attr=True,
        pre_transform=add_node_rankings,
        transform=control_transform,
    )
    return dataset


def get_test_val_train_split(name, seed: int = 0):
    # only 10 possible splits
    seed = int(seed % 10)
    train_file, val_file, test_file = (
        SPLITS_LOC / f"{name}_train.index",
        SPLITS_LOC / f"{name}_val.index",
        SPLITS_LOC / f"{name}_test.index",
    )
    splits = []
    for fn in [train_file, val_file, test_file]:
        with open(fn) as f:
            lines = f.readlines()
            index = lines[seed]
            splits.append([int(i) for i in index.strip().split(",")])
    return splits


def generate_dataloaders(dataset: TUDataset, splits, batch_size):
    loaders = []
    for split in splits:
        loaders.append(DataLoader(dataset[split], batch_size=batch_size, shuffle=True))
    return tuple(loaders)


def random_toy_graph(num_nodes=8, feature_dim=4):

    # at most half possible edges exist
    max_edges = torch.randint(high=num_nodes ** 2, size=(1,)) // 4

    # generate networkx graph, convert to torch_geometric
    temp_graph = torch_geometric.utils.from_networkx(
        nx.gnm_random_graph(num_nodes, max_edges)
    )

    x = torch.rand((num_nodes, feature_dim))
    y = torch.randint(2, (num_nodes,))

    # Data objects do not play nice with direct changes
    graph = Data(x=x, y=y, edge_index=temp_graph.edge_index)

    return graph
