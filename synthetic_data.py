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

from torch_geometric.data import Data, InMemoryDataset
import torch


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


if __name__ == "__main__":
    n = 100
    num_bridges = 16

    dist1 = [0.5, 0.0, 1.0]
    dist2 = [0.5, 15.0, 1.0]

    data = get_synthetic_data(n, num_bridges, dist1, dist2, seed=1)
    # draw_data_synthetic(data, n)
