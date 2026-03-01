import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple


class MetaStatDataset(Dataset):
    def __init__(self, datasets: List, targets: List):
        self.datasets = [torch.tensor(d, dtype=torch.float32) for d in datasets]
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.datasets[idx], self.targets[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    padded_x = pad_sequence(xs, batch_first=True)
    y = torch.stack(ys)
    return padded_x.unsqueeze(-1), lengths, y


def load_data(path):
    """Load dataset from .npz or .pkl file."""
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)

        X = data["datasets"]
        y = data["targets"]

        # Optional extra info
        # dists = data["dists"]
        # return X, y, dists
    else:
        raise ValueError(f"Unsupported file format: {path}")

    return X, y
