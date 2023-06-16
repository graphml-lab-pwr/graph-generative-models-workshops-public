import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomNoiseInitialization(BaseTransform):
    """Transform for initiliazing node features with Gaussian noise."""

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, data: Data) -> Data:
        data.x = torch.randn(data.num_nodes, self.dim, device=data.edge_index.device)
        return data
