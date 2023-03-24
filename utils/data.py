import pickle
from abc import ABC

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling


class EdgeDataset(Dataset, ABC):
    """
    EdgeDataset for training data
    """

    def __init__(
            self,
            edge_index: torch.Tensor,
            data_size: int,
            split: str = "train",
    ):
        super().__init__()
        self.data_size = data_size
        self.split = split

        if self.split == "train":
            self.edge_index = torch.stack(structured_negative_sampling(edge_index))
        else:
            self.edge_index = edge_index

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if self.split == "train":
            return self.edge_index[0][idx], self.edge_index[1][idx], self.edge_index[2][idx]
        else:
            return self.edge_index[0][idx], self.edge_index[1][idx]


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

    def get_items(self, users):
        """
        Get items for users
        :param users:
        :return:
        """
        items = []
        for user in users:
            items.append(self.data[int(user)])

        return items


class Taobao(Dataset, ABC):
    """Taobao dataset"""

    def __init__(self, path, batch_size=128):
        super().__init__()
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
        self._train_edge_index = torch.tensor([train_user, train_item], dtype=torch.long)

        self.valid_user = torch.tensor(valid_user, dtype=torch.long)
        self.valid_item = torch.tensor(valid_item, dtype=torch.long) + self.num_users
        self._valid_edge_index = torch.tensor([valid_user, valid_item], dtype=torch.long)

        self.test_user = torch.tensor(test_user, dtype=torch.long)
        self.test_item = torch.tensor(test_item, dtype=torch.long) + self.num_users
        self._test_edge_index = torch.tensor([test_user, test_item], dtype=torch.long)

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
        return EdgeDataset(self.train_edge_index, self.train_size, split="train")

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
