import math
from typing import Callable

import torch
import torch_geometric

from torch_geometric.data.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


import networkx as nx
from scipy import stats

import pathlib

SPLITS_LOC = pathlib.Path(__file__).parent / "test_train_splits"


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


def get_tu_dataset(name):
    dataset = TUDataset(
        root="./datasets",
        name=name,
        use_node_attr=True,
        pre_transform=add_node_rankings,
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
    max_edges = torch.randint(high=num_nodes**2, size=(1,)) // 4

    # generate networkx graph, convert to torch_geometric
    temp_graph = torch_geometric.utils.from_networkx(
        nx.gnm_random_graph(num_nodes, max_edges)
    )

    x = torch.rand((num_nodes, feature_dim))
    y = torch.randint(2, (num_nodes,))

    # Data objects do not play nice with direct changes
    graph = Data(x=x, y=y, edge_index=temp_graph.edge_index)

    return graph
