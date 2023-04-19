import logging
from time import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from .base import EdgeDataset


class WeightedSampler:
    def __init__(self, num_users, num_items, num_negatives, train_edge_index, batch_size=512):
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.num_nodes = num_users + num_items
        self.batch_size = batch_size
        self.train_edge_index = train_edge_index
    
    def weighted_negative_sampling(self, rating_matrix):
        """
        Structured negative sampling
        :return:
        """
        i, j = self.train_edge_index
        # print(i.shape, j.shape)
        all_ratings = rating_matrix[i]
        item_ratings = rating_matrix[i, j-self.num_users]
        item_ratings = item_ratings.reshape(-1, 1)
        weight_matrix = -1 * torch.abs(all_ratings - item_ratings)
        weight_matrix = torch.softmax(weight_matrix, dim=1)
        
        idx_1 = i * self.num_items + j
        i = i.repeat(self.num_negatives)
        neg_items_list = torch.multinomial(weight_matrix, self.num_negatives, replacement=False)  # (e*t)
        neg_items_list = neg_items_list.transpose(0, 1).flatten() + self.num_users
        idx_2 = i * self.num_nodes + neg_items_list

        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)  # (e*t,)
        rest = mask.nonzero(as_tuple=False).view(-1)
        while rest.numel() > 0:  # pragma: no cover
            # sample from the item set
            tmp = torch.randint(self.num_users, self.num_nodes, (rest.numel(),), dtype=torch.long)
            idx_2 = i[rest] * self.num_nodes + tmp
            mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
            neg_items_list[rest] = tmp
            rest = rest[mask.nonzero(as_tuple=False).view(-1)]

        assert neg_items_list.min() > self.num_users - 1 and neg_items_list.max() < self.num_nodes

        return neg_items_list.view(-1, self.num_negatives)

    def get_data_loader(self, rating_matrix, sampler="weighted"):
        """
        Data loader
        :return:
        """
        start = time()
        neg_items_list = self.weighted_negative_sampling(rating_matrix)
        logging.info(f"Negative sampling time: {time() - start}")
        dataset = EdgeDataset(self.train_edge_index, neg_items_list, split="train")
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
