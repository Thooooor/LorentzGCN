import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from abc import ABC


class EdgeDataset(Dataset, ABC):
    """
    EdgeDataset for training data
    """

    def __init__(
            self,
            edge_index: torch.Tensor,
            neg_edge_index: torch.Tensor,
            split: str = "train",
            num_negatives: int = 1,
    ):
        super().__init__()
        self.num_negatives = num_negatives
        self.split = split
        self.edge_index = edge_index
        self.neg_edge_index = neg_edge_index

    def __len__(self):
        return self.edge_index.size(1)

    def __getitem__(self, idx):
        return self.edge_index[0, idx], self.edge_index[1, idx], self.neg_edge_index[idx]

class BaseSampler:
    def __init__(self, num_users, num_items, num_negatives, train_edge_index, batch_size=512):
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.num_nodes = num_users + num_items
        self.batch_size = batch_size
        self.train_edge_index = train_edge_index
    
    def structured_negative_sampling(self):
        """
        Structured negative sampling
        :return:
        """
        i, j = self.train_edge_index.long()
        idx_1 = i * self.num_items + j
        i = i.repeat(self.num_negatives)
        k = torch.randint(self.num_users, self.num_nodes, (i.size(0),), dtype=torch.long)  # (e*t)
        idx_2 = i * self.num_nodes + k

        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)  # (e*t,)
        rest = mask.nonzero(as_tuple=False).view(-1)
        while rest.numel() > 0:  # pragma: no cover
            # sample from the item set
            tmp = torch.randint(self.num_users, self.num_nodes, (rest.numel(),), dtype=torch.long)
            idx_2 = i[rest] * self.num_nodes + tmp
            mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
            k[rest] = tmp
            rest = rest[mask.nonzero(as_tuple=False).view(-1)]

        assert k.min() > self.num_users - 1 and k.max() < self.num_nodes

        return self.train_edge_index[0], self.train_edge_index[1], k
    
    def get_data_loader(self):
        """
        Data loader
        :return:
        """
        users, pos_items, neg_items = self.structured_negative_sampling()
        
        # reshape negative items
        if self.num_negatives > 1:
            neg_items = neg_items.view(-1, self.num_negatives)
        
        dataset = EdgeDataset(self.train_edge_index, neg_items, split="train")
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
