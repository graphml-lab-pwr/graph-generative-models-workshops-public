from typing import Any, List, Tuple, Union

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

from src.data.generate import generate_dataset


class GeneratedDataset(InMemoryDataset):
    """Dataset wrapper for the community dataset"""

    def __init__(
        self,
        root: str,
        name: str,
        dataset_kwargs: dict[str, Any],
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = name
        self.dataset_kwargs = dataset_kwargs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"processed_{self.name}.pt"]

    def process(self):
        # Read data into huge `Data` list.
        graphs = generate_dataset(self.name, **self.dataset_kwargs)
        data_list = [from_networkx(g) for g in graphs]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
