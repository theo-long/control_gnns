from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

import pathlib

SPLITS_LOC = pathlib.Path("./test_train_splits")


def get_tu_dataset(name):
    dataset = TUDataset(root="./datasets", name=name, use_node_attr=True)
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
