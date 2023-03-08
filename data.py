import math
from typing import Callable, Any, Optional
import pathlib

from scipy import stats
import networkx as nx

import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


SPLITS_LOC = pathlib.Path(__file__).parent / "test_train_splits"


class RankingTransform(BaseTransform):
    """
    used as a pre_transform for TUDataset, adds node ranking field to (graph) Data objects
    NOTE: any changes will not take effect without first deleting datasets/PROTEINS/processed
    """
    def __init__(self) -> None:
        super().__init__()

    def _degree(self, data: Data):
        "wraps nx.degree to return flat tensor"
    
        graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
        degree = nx.degree(graph)
        return torch.tensor(list(degree))[:, 1].flatten()
    
    def _betweenness_centrality(self, data: Data):
        "wraps nx.betweenness_centrality to return flat tensor"
    
        graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
        between_cent = nx.betweenness_centrality(graph)
        return torch.tensor(list(between_cent.values()))
    
    def _node_rankings(self, data: Data, stat_func: Callable):
        "finds node rankings per stat_func"
    
        node_stat = stat_func(data).numpy()
    
        # ranks (negative of) node_stat 'competition style'
        stat_rankings = stats.rankdata(-node_stat, method="min")
    
        # convert to tensor
        stat_rankings = torch.tensor(stat_rankings, dtype=torch.int32)
    
        return stat_rankings

    def __call__(self, data: Data) -> Data:

        # adds new fields for node rankings
        degree_rankings = self._node_rankings(data, self._degree)
        between_cent_rankings = self._node_rankings(data, self._betweenness_centrality)

        data.node_rankings = {
            "degree": degree_rankings,
            "b_centrality": between_cent_rankings,
        }

        return data


class ControlTransform(BaseTransform):
    def __init__(self, control_type: str, metric: str, k: int) -> None:
        super().__init__()

        self.control_type = control_type
        self.metric = metric

        # TODO replace with callable
        self.k = k

    def _gen_control_edge_index(self, edge_index, active_nodes):

        if self.control_type == "adj":
    
            # get (sparse) adjacency
            A = torch_geometric.utils.to_torch_coo_tensor(edge_index)
    
            # apply mask row-wise
            B = (A * active_nodes).to_dense()
    
        # TODO check!
        elif self.control_type == "dense":
            B = (active_nodes * 1).repeat(len(active_nodes), 1)
    
        else:
            raise ValueError("Unrecognized control type, must be adj or dense")
    
        return B.nonzero().T

    def __call__(self, data: Data) -> Data:

        # TODO replace with callable
        k = self.k

        active_nodes = data.node_rankings[self.metric] <= self.k

        data.control_edge_index = self._gen_control_edge_index(data.edge_index, active_nodes)

        return data


def get_tu_dataset(name, transform):

    dataset = TUDataset(
        root="./datasets",
        name=name,
        use_node_attr=True,
        pre_transform=RankingTransform(),
        transform=transform,
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


class ToyDataset(InMemoryDataset):
    def __init__(self, transform, pre_transform):
        super().__init__(transform=transform, pre_transform=pre_transform)

        data_list = [pre_transform(self._random_toy_graph()) for _ in range(10)]
        self.data, self.slices = self.collate(data_list)

    def _random_toy_graph(self, num_nodes=8, feature_dim=4):
    
        # at most half possible edges exist
        max_edges = torch.randint(low=num_nodes, high=num_nodes ** 2, size=(1,))
    
        # generate networkx graph, convert to torch_geometric
        temp_graph = torch_geometric.utils.from_networkx(
            nx.gnm_random_graph(num_nodes, max_edges)
        )
    
        x = torch.rand((num_nodes, feature_dim))
        y = torch.randint(2, (num_nodes,))
    
        # Data objects do not play nice with direct changes
        graph = Data(x=x, y=y, edge_index=temp_graph.edge_index)
    
        return graph
