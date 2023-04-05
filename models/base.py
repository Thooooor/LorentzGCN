from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj


class Base(torch.nn.Module):
    """Base class for GNN Recommender System."""

    def __init__(self, num_users, num_items, graph, args):
        super(Base, self).__init__()
        self.graph = graph.cuda()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.num_layers
        self.margin = args.margin
        self.embedding = torch.nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim).cuda()
        
        self.reset_parameters(args.scale)
        
    def reset_parameters(self, scale):
        """Reinitialize learnable parameters.

        Args:
            scale: scale of uniform distribution.
        """
        torch.nn.init.uniform_(self.embedding.weight, a=-scale, b=scale)

    @abstractmethod
    def forward(self, edge_index: Adj, include_embedding: bool = False):
        """
        Forward propagation. 
        :param edge_index: [2, num_edges]
        :param include_embedding: bool (default: False) 
        :return: [num_edges, 1]
        """
        out = self.compute_embedding()
        
        out_src = out[edge_index[0]]
        out_dst = out[edge_index[1]]
        
        scores = self.score_function(out_src, out_dst)
        
        if include_embedding:
            return scores, out_src, out_dst
        else:
            return scores

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
        log_prob = F.softplus(-(pos_scores - neg_scores)).mean()
        
        user_embedding = self.embedding.weight[edge_index[0]]
        pos_item_embedding = self.embedding.weight[edge_index[1]]
        neg_item_embedding = self.embedding.weight[neg_edge_index[1]]
        regularization_loss = (1/2) * (user_embedding.norm(p=2).pow(2) + pos_item_embedding.norm(p=2).pow(2) + neg_item_embedding.norm(p=2).pow(2)) / float(edge_index.shape[1])
        
        return log_prob + lambda_reg * regularization_loss

    def margin_loss(self, users, pos_items, neg_items_list, margin: float = 0.1):
        """
        Margin-based loss.
        :param edge_index: [2, num_edges]
        :param neg_edge_index: [2, num_edges]
        :param margin: float
        :return: float
        """
        edge_index = torch.stack([users, pos_items], dim=0)
        pos_scores = self.forward(edge_index)
        
        num_negatives = neg_items_list.shape[1]
        neg_scores_list = []
        for i in range(num_negatives):
            neg_edge_index = torch.stack([users, neg_items_list[:, i]], dim=0)
            neg_scores_list.append(self.forward(neg_edge_index))
        neg_scores = torch.stack(neg_scores_list, dim=1)
        neg_scores = torch.mean(neg_scores, dim=1)
        
        loss = pos_scores - neg_scores + margin
        loss[loss < 0] = 0
        return loss.sum()

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
