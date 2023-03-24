import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.typing import Adj


class Base(torch.nn.Module):
    """Base class for GNN Recommender System."""
    def __init__(
            self,
            num_users: int,
            num_items: int,
            embedding_size=64,
            num_layers=3,
    ):
        super(Base, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(self.num_nodes, embedding_size)

    def bpr_loss(self, edge_index: Adj, neg_edge_index: Adj, lambda_reg: float = 1e-4):
        """
        Bayesian Personalized Ranking loss.
        :param edge_index: [2, num_edges]
        :param neg_edge_index: [2, num_edges]
        :param lambda_reg: float
        :return: float
        """
        n_pairs = edge_index.size(1)
        pos_out = self.forward(edge_index)
        neg_out = self.forward(neg_edge_index)

        log_prob = F.logsigmoid(pos_out - neg_out).mean()
        regularization = 0
        if lambda_reg != 0:
            regularization = lambda_reg * self.embedding.weight.norm(p=2).pow(2)

        return (-log_prob + regularization) / n_pairs

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
        user_embeddings = self.embedding(user_ids)
        item_embeddings = self.embedding.weight[self.num_users:]

        scores = user_embeddings @ item_embeddings.t()

        return scores
