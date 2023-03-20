import math
from typing import Callable, Any, Optional, Union
import pathlib
from utils import to_edge_index

from scipy import stats, sparse
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci

import torch
import torch_sparse
import torch_geometric
from torch_geometric.utils import to_torch_coo_tensor, from_networkx
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.datasets import TUDataset, Planetoid, WikipediaNetwork
from torch_geometric.loader import DataLoader


SPLITS_LOC = pathlib.Path(__file__).parent / "test_train_splits"
DATASET_DICT = {
    "PROTEINS": (TUDataset, {"use_node_attr": True}, False),
    "ENZYMES": (TUDataset, {"use_node_attr": True}, False),
    "cora": (Planetoid, {"split": "geom-gcn"}, True),
    "pubmed": (Planetoid, {"split": "geom-gcn"}, True),
    "citeseer": (Planetoid, {"split": "geom-gcn"}, True),
    "chameleon": (WikipediaNetwork, {"geom_gcn_preprocess": True}, True),
    "squirrel": (WikipediaNetwork, {"geom_gcn_preprocess": True}, True),
}


class RankingTransform(BaseTransform):
    """
    used as a pre_transform for TUDataset, adds node ranking field to (graph) Data objects
    NOTE: any changes will not take effect without first deleting datasets/PROTEINS/processed
    """

    def __init__(self) -> None:
        super().__init__()

    def _degree(self, data: Data):
        """wraps nx.degree to return flat tensor"""

        graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
        degree = nx.degree(graph)
        return torch.tensor(list(degree))[:, 1].flatten()

    def _betweenness_centrality(self, data: Data):
        """wraps nx.betweenness_centrality to return flat tensor"""

        graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
        between_cent = nx.betweenness_centrality(graph)
        return torch.tensor(list(between_cent.values()))

    def _pagerank_centrality(self, data: Data):
        """calculates page rank centrality"""
        graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
        pr_cent = nx.pagerank(graph)
        return torch.tensor(list(pr_cent.values()))

    def _curvature(self, data: Data):
        graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
        forman_curvature = FormanRicci(graph)
        forman_curvature.compute_ricci_curvature()
        curvature_data = from_networkx(forman_curvature.G)
        return curvature_data.formanCurvature * -1.0

    def _node_rankings(self, data: Data, stat_func: Callable):
        """finds node rankings per stat_func"""

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
        pr_rankings = self._node_rankings(data, self._pagerank_centrality)
        curvature = self._node_rankings(data, self._curvature)

        data.node_rankings = {
            "degree": degree_rankings,
            "b_centrality": between_cent_rankings,
            "pr_centrality": pr_rankings,
            "curvature": curvature,
        }

        return data


class TwoHopTransform(BaseTransform):
    """
    used to generate a two-hop adjacency matrix
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Data) -> Data:
        adj_matrix = to_torch_coo_tensor(data.edge_index)
        two_hop_adj_matrix = adj_matrix @ adj_matrix - to_torch_coo_tensor(
            *torch_sparse.eye(data.x.shape[0])
        )
        two_hop_edge_index = to_edge_index(two_hop_adj_matrix)
        data.two_hop_edge_index = two_hop_edge_index[0]
        return data


class ControlTransform(BaseTransform):
    """
    used to identify the edges to 'activate' in control modules
    applied as transform while dataloading
    """

    def __init__(
        self, control_edges: str, metric: str, num_active: Callable, self_adj: bool
    ) -> None:
        super().__init__()

        self.control_edges = control_edges
        self.metric = metric

        self.num_active = num_active
        self.self_adj = self_adj

    def _gen_control_edge_index(self, edge_index, active_nodes):
        "generates the control_edge_index"

        if self.control_edges in ["adj", "two_hop"]:

            # I did this like this to avoid a for loop (over the number of active nodes)
            # not sure how much it actually speeds it up
            expanded = edge_index[0:1, :].expand(active_nodes.size(0), -1)
            indices = (
                (expanded == active_nodes.view(-1, 1))
                .int()
                .sum(dim=0)
                .nonzero()
                .flatten()
            )

            control_edge_index = edge_index[:, indices]

            if self.self_adj:
                # generate self adjacency edges
                self_adj_edges = active_nodes.repeat(2, 1)

                # add to the control edge index
                control_edge_index = torch.cat(
                    [control_edge_index, self_adj_edges], dim=1
                )

        elif self.control_edges == "dense":

            num_nodes = edge_index.max() + 1

            # I did this like this to avoid a for loop (over the number of active nodes)
            # this one actually gives a decent speedup
            source_nodes = active_nodes.repeat_interleave(num_nodes)
            dest_nodes = torch.arange(num_nodes).repeat(active_nodes.size(0))
            edges = torch.stack([source_nodes, dest_nodes])

            if not self.self_adj:
                # remove the self adjacency edges
                control_edge_index = edges[:, (edges[0] != edges[1])]
            else:
                # keep the self adjacency edges
                control_edge_index = edges

        else:
            raise ValueError("Unrecognized control type, must be adj or dense")

        return control_edge_index

    def __call__(self, data: Data) -> Data:

        k = self.num_active(data.x.shape[0])

        active_nodes = (data.node_rankings[self.metric] <= k).nonzero().flatten()

        if self.control_edges == "two_hop":
            base_edge_index = data.two_hop_edge_index
        else:
            base_edge_index = data.edge_index

        data.control_edge_index = self._gen_control_edge_index(
            base_edge_index, active_nodes
        )

        return data


def get_dataset(
    name,
    control_type,
    control_edges,
    control_metric,
    num_active: Callable,
    control_self_adj,
):

    if control_type != "null":
        transform = ControlTransform(
            control_edges, control_metric, num_active, control_self_adj
        )
    else:
        transform = None

    dataset_class, dataset_kwargs, is_node_classifier = DATASET_DICT[name]

    dataset = dataset_class(
        root="./datasets",
        name=name,
        pre_transform=Compose([RankingTransform(), TwoHopTransform()]),
        transform=transform,
        **dataset_kwargs,
    )

    return dataset, is_node_classifier


def get_test_val_train_split(name, split: int = 0):
    # only 10 possible splits
    split = int(split % 10)
    train_file, val_file, test_file = (
        SPLITS_LOC / f"{name}_train.index",
        SPLITS_LOC / f"{name}_val.index",
        SPLITS_LOC / f"{name}_test.index",
    )
    splits = []
    for fn in [train_file, val_file, test_file]:
        with open(fn) as f:
            lines = f.readlines()
            index = lines[split]
            splits.append([int(i) for i in index.strip().split(",")])
    return splits


def get_test_val_train_mask(
    dataset: Union[WikipediaNetwork, Planetoid], split: int = 0
):
    return (
        dataset[0].train_mask[:, split].to(torch.bool),
        dataset[0].val_mask[:, split].to(torch.bool),
        dataset[0].test_mask[:, split].to(torch.bool),
    )


def generate_dataloaders(dataset: TUDataset, dataset_name, batch_size, split=0):

    splits = get_test_val_train_split(dataset_name, split)

    loaders = []
    for split in splits:
        loaders.append(DataLoader(dataset[split], batch_size=batch_size, shuffle=True))
    return tuple(loaders)


class ToyDataset(InMemoryDataset):
    """
    a toy dataset used in debug_control.py
    """

    def __init__(self, transform, pre_transform):
        super().__init__(transform=transform, pre_transform=pre_transform)

        data_list = [pre_transform(self._random_toy_graph()) for _ in range(10)]
        self.data, self.slices = self.collate(data_list)

    def _random_toy_graph(self, num_nodes=8, feature_dim=4):

        num_edges = num_nodes * 2

        # generate networkx graph, convert to torch_geometric
        temp_graph = torch_geometric.utils.from_networkx(
            nx.gnm_random_graph(num_nodes, num_edges)
        )

        x = torch.rand((num_nodes, feature_dim))
        y = torch.randint(2, (num_nodes,))

        # Data objects do not play nice with direct changes
        graph = Data(x=x, y=y, edge_index=temp_graph.edge_index)

        return graph
