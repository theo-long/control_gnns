import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import networkx as nx
from networkx.generators import (
    random_geometric_graph,
    random_clustered_graph,
    complete_graph,
)
from torch_geometric.utils import (
    contains_self_loops,
    add_self_loops,
    add_remaining_self_loops,
    is_undirected,
    to_undirected,
    from_networkx,
    to_networkx,
    degree,
    coalesce,
    remove_self_loops,
)

from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import stochastic_blockmodel_graph
import torch_geometric
import torch

import numpy as np
import itertools
import random
import math

from typing import Optional, Callable


def get_synthetic_data(num_nodes, num_bridges, dist1, dist2, seed):
    np.random.seed(seed)
    # G = random_geometric_graph(n, 4)
    assert num_nodes % 4 == 0, "must have multiple 4 number of nodes in this dataset"
    half_nodes = int(num_nodes / 2)
    G1 = complete_graph(half_nodes)
    G2 = complete_graph(half_nodes)
    d1 = from_networkx(G1)
    d2 = from_networkx(G2)
    bridges = torch.cat(
        [
            torch.randint(low=0, high=half_nodes, size=(num_bridges, 1)),
            torch.randint(low=half_nodes, high=num_nodes, size=(num_bridges, 1)),
        ],
        dim=1,
    ).T
    edge_index = torch.cat(
        [d1.edge_index, d2.edge_index + half_nodes, bridges], dim=1
    ).type(torch.LongTensor)
    edge_index = coalesce(edge_index)  # remove dupes
    edge_index, _ = remove_self_loops(edge_index)  # remove self_loops
    edge_index = to_undirected(edge_index)

    # make fatures
    # bi-modal
    [w1, m1, s1] = dist1  # 0.3, 0.0, 1.0
    [w2, m2, s2] = dist2  # 0.7, 5.0, 1.0

    n1 = int(w1 * half_nodes)
    n2 = int(w2 * half_nodes)
    while n1 + n2 != half_nodes:
        n1 += 1
    G11 = np.random.normal(m1, s1, n1)
    G12 = np.random.normal(m2, s2, n2)
    f1 = np.concatenate((G11, G12))  # [:, np.newaxis]
    np.random.shuffle(f1)  # neccessary to shuffle to remove bimodal signal

    # uni-modal
    # mu = w1 * m1 + w2 * m2 #expected mean
    # sd = np.sqrt((w1*s1**2 + w2*s2**2)/(w1 + w2)) #expected sd
    mu = np.mean(f1)  # actual mean
    sd = np.std(f1)  # actual sd
    f2 = np.random.normal(mu, sd, half_nodes)
    d = 1  # 3  # 1
    b = 1  # 2  # #1 #batch / towers
    x = (
        torch.cat(
            [
                torch.from_numpy(f1.astype(np.float32)),
                torch.from_numpy(f2.astype(np.float32)),
            ]
        )
        .repeat(1, 1, 1)
        .permute(2, 1, 0)
    )  # [num_nodes,towers,dim]

    # make labels
    y = torch.cat(
        [torch.zeros(half_nodes), torch.ones(half_nodes)]
    )  # .type(torch.LongTensor)

    # make masks
    # 10 train / 10 val / 80 test
    n_train = int(num_nodes * 0.10)
    n_val = int(num_nodes * 0.10)
    n_test = num_nodes - n_train - n_val
    all_idx = list(range(num_nodes))
    train_idx = [all_idx.pop(np.random.choice(len(all_idx))) for i in range(n_train)]
    val_idx = [all_idx.pop(np.random.choice(len(all_idx))) for i in range(n_val)]
    test_idx = [all_idx.pop(np.random.choice(len(all_idx))) for i in range(n_test)]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    train_mask = get_mask(train_idx)
    val_mask = get_mask(val_idx)
    test_mask = get_mask(test_idx)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    draw_data_synthetic(data, num_nodes)

    return data


def draw_data_synthetic(data, n):
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, with_labels=True)  # , node_color='k')  node_size=1500)
    # pos_dict = {k: v + np.array([0, 0.08]) for k, v in pos.items()}
    # labels_dict = {i: np.round(x[i, 0, 0].item(), 2) for i in range(x.shape[0])}
    # nx.draw_networkx_labels(G, pos=pos_dict, labels=labels_dict)  # draw node labels/names
    plt.show()

    x = data.x
    plt.hist(x[: n // 2, 0, 0], bins=10, alpha=0.5, label="bi-modal")
    plt.hist(x[n // 2 :, 0, 0], bins=10, alpha=0.5, label="uni-modal")
    plt.legend()
    plt.show()


class TreeDataset(InMemoryDataset):
    """Synthetic Tree dataset from https://arxiv.org/pdf/2006.05205.pdf"""

    def __init__(self, depth, transform=None, pre_transform=None, **kwargs):
        super().__init__(transform=transform, pre_transform=pre_transform)
        self.depth = depth
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()
        self.criterion = torch.nn.functional.cross_entropy

        data = self.generate_data()
        self.data, self.slices = self.collate(data)

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue
            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([right_child, cur_node])
            stack.append((right_child, max_node))
            stack.append((left_child, right_child - 1))
        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self, add_self_loops=True):
        edge_index = torch.tensor(self.edges).t()
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(
                edge_index=edge_index,
            )
        return edge_index

    def generate_data(self):
        data_list = []

        for comb in self.get_combinations():
            edge_index = self.create_blank_tree(add_self_loops=True)
            nodes = torch.tensor(self.get_nodes_features(comb), dtype=torch.long)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1))
            label = self.label(comb)
            data_list.append(
                Data(x=nodes, edge_index=edge_index, out_mask=root_mask, y=label - 1)
            )

        return data_list

    def get_combinations(self):
        # returns: an iterable of [key, permutation(leaves)]
        # number of combinations: (num_leaves!)*num_choices
        num_leaves = len(self.leaf_indices)
        num_permutations = 1000
        max_examples = 32000

        if self.depth > 3:
            per_depth_num_permutations = min(
                num_permutations, math.factorial(num_leaves), max_examples // num_leaves
            )
            permutations = [
                np.random.permutation(range(1, num_leaves + 1))
                for _ in range(per_depth_num_permutations)
            ]
        else:
            permutations = random.sample(
                list(itertools.permutations(range(1, num_leaves + 1))),
                min(num_permutations, math.factorial(num_leaves)),
            )

        return itertools.chain.from_iterable(
            zip(range(1, num_leaves + 1), itertools.repeat(perm))
            for perm in permutations
        )

    def get_nodes_features(self, combination):
        # combination: a list of indices
        # Each leaf contains a one-hot encoding of a key, and a one-hot encoding of the value
        # Every other node is empty, for now
        selected_key, values = combination

        # The root is [one-hot selected key] + [0 ... 0]
        nodes = [(selected_key, 0)]

        for i in range(1, self.num_nodes):
            if i in self.leaf_indices:
                leaf_num = self.leaf_indices.index(i)
                node = (leaf_num + 1, values[leaf_num])
            else:
                node = (0, 0)
            nodes.append(node)
        return nodes

    def label(self, combination):
        selected_key, values = combination
        return int(values[selected_key - 1])

    def get_dims(self):
        # get input and output dims
        in_dim = len(self.leaf_indices) + 1
        out_dim = len(self.leaf_indices)
        return in_dim, out_dim


class LinearDataset(InMemoryDataset):
    """
    A dataset of linearly connected complete graphs
    """

    def __init__(
        self, num_nodes, num_parts, transform=None, pre_transform=None, **kwargs
    ):
        super().__init__(transform=transform, pre_transform=pre_transform)

        data = self._generate_linear_graph(num_nodes, num_parts)
        self.data, self.slices = self.collate([data])

    def _generate_linear_graph(self, num_nodes, num_parts):

        complete_graph = torch_geometric.utils.from_networkx(
            nx.complete_graph(num_nodes)
        )
        edge_index = torch.cat(
            [complete_graph.edge_index + i * num_nodes for i in range(num_parts)],
            dim=-1,
        )

        # Add single bridge edge between parts
        in_bridges = torch.tensor(
            [
                [i * num_nodes for i in range(num_parts - 1)],
                [i * num_nodes for i in range(1, num_parts)],
            ]
        )
        out_bridges = in_bridges[[1, 0]]
        edge_index = torch.cat([edge_index, in_bridges, out_bridges], -1)

        x = torch.ones((num_nodes * num_parts, 1)).to(torch.float32)
        y = torch.arange(0, num_parts).repeat_interleave(num_nodes)

        # Generate masks
        train_mask = torch.ones(num_nodes * num_parts, 10)
        val_mask = torch.ones(num_nodes * num_parts, 10)
        test_mask = torch.ones(num_nodes * num_parts, 10)

        data = torch_geometric.data.Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask,
        )

        return data


class LabelPropagationDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        name,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(transform, pre_transform, **kwargs)
        self.data, self.slices = self.collate(
            [self._generate_data() for i in range(5000)]
        )

    def _generate_data(self):
        block_sizes = [20, 20, 20, 20]
        edge_probs = torch.tensor(
            [
                [0.8, 0.01, 0.0, 0.0],
                [0.01, 0.8, 0.01, 0.0],
                [0.0, 0.01, 0.8, 0.01],
                [0.0, 0.0, 0.01, 0.8],
            ]
        )
        edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)

        # Need to ensure connectivity

        x = torch.zeros(sum(block_sizes), 1, dtype=torch.long)
        x[0][0] = torch.randint(1, 10, size=(1,)).item()
        x[-1][0] = torch.randint(11, 20, size=(1,)).item()

        # The goal is to propagate the label at x[-1][0] to the node at x[0][0]
        y = x[-1][0]
        out_mask = torch.zeros(sum(block_sizes), dtype=torch.bool)
        out_mask[0] = 1

        return Data(edge_index=edge_index, x=x, y=y, out_mask=out_mask)

    def get_dims(self):
        return 21, 21


if __name__ == "__main__":
    n = 100
    num_bridges = 16

    dist1 = [0.5, 0.0, 1.0]
    dist2 = [0.5, 15.0, 1.0]

    data = get_synthetic_data(n, num_bridges, dist1, dist2, seed=1)
    # draw_data_synthetic(data, n)
