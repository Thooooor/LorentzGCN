from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj


class Base(torch.nn.Module):
    """Base class for GNN Recommender System."""

    def __init__(
            self,
            num_users: int,
            num_items: int,
            graph,
            embedding_size=64,
            num_layers=3,
            num_negatives=1
    ):
        super(Base, self).__init__()

        self.num_negatives = num_negatives
        self.graph = graph.cuda()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(self.num_nodes, embedding_size)
        self.adj_mat = torch.ones((self.num_nodes, self.num_nodes))
        self.adj_mat[self.num_users:, self.num_users:] = 0

    @abstractmethod
    def forward(self, edge_index: Adj):
        """
        Forward propagation.
        :param edge_index: [2, num_edges]
        :return: [num_edges, 1]
        """
        pass

    @abstractmethod
    def compute_embedding(self):
        """
        Get embedding of all nodes.
        :return: [num_nodes, embedding_dim]
        """
        pass

    def score_function(self, user_ids: Tensor, item_ids: Tensor):
        """
        Score function for given users and items.
        :param user_ids: [num_users]
        :param item_ids: [num_items]
        :return: [num_users, num_items]
        """
        user_embeddings = self.embedding(user_ids)
        item_embeddings = self.embedding(item_ids)

        scores = user_embeddings @ item_embeddings.t()

        return scores

    def pair_wise_score_function(self, user_ids: Tensor, item_ids: Tensor):
        pass

    def bpr_loss(self, edge_index: Adj, neg_edge_index: Adj, lambda_reg: float = 1e-4):
        """
        Bayesian Personalized Ranking loss.
        :param edge_index: [2, num_edges]
        :param neg_edge_index: [2, num_edges]
        :param lambda_reg: float
        :return: float
        """
        pos_scores = self.forward(edge_index)
        neg_scores = self.forward(neg_edge_index)

        log_prob = F.logsigmoid(-(pos_scores - neg_scores)).mean()
        return -log_prob + lambda_reg * self.regularization_loss
        # return (pos_scores - neg_scores).sum()

    @property
    def regularization_loss(self):
        return self.embedding.weight.norm(p=2).pow(2)

    def recommend(self, user_ids: Tensor, top_k=10):
        """
        Recommend top-k items for given users.
        :param user_ids: [num_users]
        :param top_k: int
        :return: [num_users, top_k]
        """
        user_embeddings = self.embedding(user_ids)
        item_embeddings = self.embedding.weight[self.num_users:]

        scores = user_embeddings @ item_embeddings.t()
        _, indices = scores.topk(top_k, dim=-1)

        return indices

    def get_user_rating(self, user_ids: Tensor):
        """
        Get rating scores of all items for given users.
        :param user_ids: [num_users]
        :return: [num_users, num_items]
        """
        all_embeddings = self.compute_embedding()
        user_embeddings = all_embeddings(user_ids)
        item_embeddings = all_embeddings.weight[self.num_users:]

        scores = user_embeddings @ item_embeddings.t()

        return scores
