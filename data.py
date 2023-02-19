import math
from typing import Callable

import torch
import torch_geometric

from torch_geometric.data.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

import networkx as nx


TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2


def add_node_rankings(data: Data):
    """
    used as a pre_transform for TUDataset, adds node ranking field to (graph) Data objects
    NOTE: any changes will not take effect without first deleting datasets/PROTEINS/processed
    """

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

        stat = stat_func(data)

        # finds ranking data
        sorted_stat, indices = torch.sort(stat, descending=True)
        _, rankings = torch.unique_consecutive(sorted_stat, return_inverse=True)

        # creates tensor of ranking data
        stat_rankings = torch.zeros(data.x.shape[0], dtype=torch.int32)
        stat_rankings[indices] = rankings.to(torch.int32)

        return stat_rankings

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


def generate_dataloaders(dataset: TUDataset, batch_size):
    # random shuffle before splitting
    dataset = dataset.shuffle()
    loaders = []
    index = 0
    for split in [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT]:
        upper_index = index + math.ceil(len(dataset) * split)
        loaders.append(
            DataLoader(dataset[index:upper_index], batch_size=batch_size, shuffle=True)
        )
        index = upper_index
    return tuple(loaders)


def random_toy_graph(num_nodes=4, feature_dim=4):

    max_edges = torch.randint(high=num_nodes**2, size=(1,))
    temp_graph = torch_geometric.utils.from_networkx(
        nx.gnm_random_graph(num_nodes, max_edges)
    )

    x = torch.rand((num_nodes, feature_dim))
    y = torch.randint(2, (num_nodes,))

    graph = Data(x=x, y=y, edge_index=temp_graph.edge_index)

    return graph
