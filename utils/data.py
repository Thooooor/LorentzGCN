import pickle
from abc import ABC

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


class EdgeDataset(Dataset, ABC):
    """
    EdgeDataset for training data
    """

    def __init__(
            self,
            edge_index: torch.Tensor,
            data_size: int,
            split: str = "train",
            num_negatives: int = 1,
    ):
        super().__init__()
        self.num_negatives = num_negatives
        self.data_size = data_size
        self.split = split
        self.edge_index = edge_index

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.edge_index[0][idx], self.edge_index[1][idx], self.edge_index[2][idx]


class UserDataset(Dataset, ABC):
    """
    UserDataset for validation and test data
    """

    def __init__(
            self,
            edge_index: torch.Tensor,
            users: torch.Tensor,
            data: dict,
            split: str = "valid",
    ):
        super().__init__()
        self.edge_index = edge_index
        self.users = users
        self.data_size = users.shape[0]
        self.split = split
        self.data = data

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.users[idx]

    def get_items(self, users, num_users):
        """
        Get items for users
        :param num_users: 
        :param users:
        :return:
        """
        items = []
        for user in users:
            items.append([item_id + num_users for item_id in self.data[user]])

        return items


class Taobao(Dataset, ABC):
    """Taobao dataset"""

    def __init__(self, path, batch_size=128, num_negatives=1):
        super().__init__()
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.path = path
        self.num_users = 0
        self.num_items = 0

        train_unique_users, train_item, train_user = [], [], []
        test_unique_users, test_item, test_user = [], [], []
        valid_unique_users, valid_item, valid_user = [], [], []

        self.train_size = 0
        self.test_size = 0
        self.valid_size = 0

        # Load data
        train_dict = self.train_dict
        for user, items in train_dict.items():
            train_user.extend([user] * len(items))
            train_item.extend(items)
            self.train_size += len(items)
            self.num_users = max(self.num_users, user)
            self.num_items = max(self.num_items, max(items))
        self.train_unique_users = torch.tensor(list(train_dict.keys()), dtype=torch.long)

        valid_dict = self.valid_dict
        for user, items in valid_dict.items():
            valid_user.extend([user] * len(items))
            valid_item.extend(items)
            self.valid_size += len(items)
            self.num_users = max(self.num_users, user)
            self.num_items = max(self.num_items, max(items))
        self.valid_unique_users = torch.tensor(list(valid_dict.keys()), dtype=torch.long)

        test_dict = self.test_dict
        for user, items in test_dict.items():
            test_user.extend([user] * len(items))
            test_item.extend(items)
            self.test_size += len(items)
            self.num_users = max(self.num_users, user)
            self.num_items = max(self.num_items, max(items))
        self.test_unique_users = torch.tensor(list(test_dict.keys()), dtype=torch.long)

        self.num_users += 1  # Index starts from 0
        self.num_items += 1
        self.num_nodes = self.num_users + self.num_items

        # Item index starts from num_users
        self.train_user = torch.tensor(train_user, dtype=torch.long)
        self.train_item = torch.tensor(train_item, dtype=torch.long) + self.num_users
        self._train_edge_index = torch.stack([self.train_user, self.train_item], dim=0)

        self.valid_user = torch.tensor(valid_user, dtype=torch.long)
        self.valid_item = torch.tensor(valid_item, dtype=torch.long) + self.num_users
        self._valid_edge_index = torch.stack([self.valid_user, self.valid_item], dim=0)

        self.test_user = torch.tensor(test_user, dtype=torch.long)
        self.test_item = torch.tensor(test_item, dtype=torch.long) + self.num_users
        self._test_edge_index = torch.stack([self.test_user, self.test_item], dim=0)

    def structured_negative_sampling(self):
        """
        Structured negative sampling
        :return:
        """
        i, j = self._train_edge_index.long()
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

        return self._train_edge_index[0], self._train_edge_index[1], k

    @property
    def train_edge_index(self):
        return self._train_edge_index

    @property
    def valid_edge_index(self):
        return self._valid_edge_index

    @property
    def test_edge_index(self):
        return self._test_edge_index

    @property
    def train_data(self):
        return Data(x=self.train_unique_users, edge_index=self.train_edge_index)

    @property
    def valid_data(self):
        return Data(x=self.valid_unique_users, edge_index=self.valid_edge_index)

    @property
    def test_data(self):
        return Data(x=self.test_unique_users, edge_index=self.test_edge_index)

    @property
    def train_dict(self):
        fr = open(self.path + '/train.pkl', 'rb')
        train_dict = pickle.load(fr)
        fr.close()
        return train_dict

    @property
    def valid_dict(self):
        fr = open(self.path + '/val.pkl', 'rb')
        valid_dict = pickle.load(fr)
        fr.close()
        return valid_dict

    @property
    def test_dict(self):
        fr = open(self.path + '/test.pkl', 'rb')
        test_dict = pickle.load(fr)
        fr.close()
        return test_dict

    @property
    def train_set(self):
        users, pos_items, neg_items = self.structured_negative_sampling()
        sampled_edge_index = torch.stack([users, pos_items, neg_items], dim=0)
        return EdgeDataset(sampled_edge_index, self.train_size, split="train")

    @property
    def valid_set(self):
        return UserDataset(self.valid_edge_index, self.valid_unique_users, self.valid_dict, split="valid")

    @property
    def test_set(self):
        return UserDataset(self.test_edge_index, self.test_unique_users, self.test_dict, split="test")

    @property
    def train_loader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    @property
    def valid_loader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

    @property
    def test_loader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
