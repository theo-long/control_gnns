from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

import math

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2


def get_tu_dataset(name):
    dataset = TUDataset(root="./datasets", name=name, use_node_attr=True)
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
