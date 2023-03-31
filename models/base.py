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
            embedding_dim=64,
            num_layers=3,
            scale=0.1,
            margin=1.0
    ):
        super(Base, self).__init__()
        self.graph = graph.cuda()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.margin = margin
        self.embedding = torch.nn.Embedding(self.num_nodes, embedding_dim).cuda()
        
        self.reset_parameters(scale)
        
    def reset_parameters(self, scale):
        """Reinitialize learnable parameters.

        Args:
            scale: scale of uniform distribution.
        """
        torch.nn.init.uniform_(self.embedding.weight, a=-scale, b=scale)

    @abstractmethod
    def forward(self, edge_index: Adj):
        """
        Forward propagation.
        :param edge_index: [2, num_edges]
        :return: [num_edges, 1]
        """
        out = self.compute_embedding()
        
        out_src = out[edge_index[0]]
        out_dst = out[edge_index[1]]
        
        return self.score_function(out_src, out_dst)

    @abstractmethod
    def compute_embedding(self):
        """
        Get embedding of all nodes.
        :return: [num_nodes, embedding_dim]
        """
        pass

    def score_function(self, src_embbeding: Tensor, dst_embedding: Tensor):
        """
        Score function for given users and items.
        :param src_embbeding: [num_users, embedding_dim]
        :param dst_embedding: [num_items, embedding_dim]
        :return: [num_users, num_items]
        """
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

    def margin_loss(self, edge_index: Adj, neg_edge_index: Adj, margin: float = 0.1):
        """
        Margin-based loss.
        :param edge_index: [2, num_edges]
        :param neg_edge_index: [2, num_edges]
        :param margin: float
        :return: float
        """
        pos_scores = self.forward(edge_index)
        neg_scores = self.forward(neg_edge_index)
        
        loss = pos_scores - neg_scores + margin
        loss[loss < 0] = 0
        return loss.sum()

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
        scores = self.get_user_rating(user_ids)
        _, indices = scores.topk(top_k, dim=-1)

        dst_index = indices + self.num_users

        return dst_index

    def get_user_rating(self):
        """
        Get rating scores of all items for given users.
        :param user_ids: [num_users]
        :return: [num_users, num_items]
        """
        out = self.compute_embedding()
        probs_matrix = np.zeros((self.num_users, self.num_items))
        
        for user in range(self.num_users):
            user_embedding = out[user]
            user_embeddings = user_embedding.repeat(self.num_items).view(self.num_items, -1)
            item_embeddings = out[np.arange(self.num_users, self.num_nodes), :]
            scores = self.score_function(user_embeddings, item_embeddings)

            probs = scores.detach().cpu().numpy() * -1
            probs_matrix[user] = np.reshape(probs, [-1, ])
            
        return probs_matrix
