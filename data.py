from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader


def get_protein_dataset():
    dataset = TUDataset(root="./datasets", name="PROTEIN", use_node_attr=True)
    return dataset


def generate_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
