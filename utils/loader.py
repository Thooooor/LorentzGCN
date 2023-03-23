from abc import ABC
import pickle
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix


class DataLoader(Dataset, ABC):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.num_user = 0
        self.num_item = 0
        train_file = path + '/train.pkl'
        test_file = path + '/test.pkl'
        valid_file = path + '/val.pkl'

        train_unique_users, train_item, train_user = [], [], []
        test_unique_users, test_item, test_user = [], [], []
        valid_unique_users, valid_item, valid_user = [], [], []

        self.train_size = 0
        self.test_size = 0
        self.valid_size = 0

        fr = open(train_file, 'rb')
        train_data = pickle.load(fr)
        fr.close()
        train_unique_users = train_data.keys()
        for user, items in train_data.items():
            train_user.extend([user] * len(items))
            train_item.extend(items)
            self.train_size += len(items)
            self.num_user = max(self.num_user, user)
            self.num_item = max(self.num_item, max(items))
        self.train_unique_users = np.array(train_unique_users)
        self.train_user = np.array(train_user)
        self.train_item = np.array(train_item)

        fr = open(valid_file, 'rb')
        valid_data = pickle.load(fr)
        fr.close()
        valid_unique_users = valid_data.keys()
        for user, items in valid_data.items():
            valid_user.extend([user] * len(items))
            valid_item.extend(items)
            self.valid_size += len(items)
            self.num_user = max(self.num_user, user)
            self.num_item = max(self.num_item, max(items))
        self.valid_unique_users = np.array(valid_unique_users)
        self.valid_user = np.array(valid_user)
        self.valid_item = np.array(valid_item)

        fr = open(test_file, 'rb')
        test_data = pickle.load(fr)
        fr.close()
        test_unique_users = test_data.keys()
        for user, items in test_data.items():
            test_user.extend([user] * len(items))
            test_item.extend(items)
            self.test_size += len(items)
            self.num_user = max(self.num_user, user)
            self.num_item = max(self.num_item, max(items))
        self.test_unique_users = np.array(test_unique_users)
        self.test_user = np.array(test_user)
        self.test_item = np.array(test_item)

        self.num_user += 1
        self.num_item += 1

        self.user_item_net = csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                        shape=(self.num_user, self.num_item))

        self.user_degree = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.user_degree[self.user_degree == 0.] = 1.

        self.item_degree = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.item_degree[self.item_degree == 0.] = 1.
