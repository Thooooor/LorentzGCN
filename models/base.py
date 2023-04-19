from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
import logging

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
        self.probs_matrix = torch.zeros((self.num_users, self.num_items))
        self.neg_items_list = None
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
    
    def structured_negative_sampling(self, users, pos_items, num_negatives: int = 1):
        """
        Structured negative sampling
        :return:
        """
        i, j = users.cpu(), pos_items.cpu()
        idx_1 = i * self.num_items + j
        i = i.repeat(num_negatives)
        neg_items_list = torch.randint(self.num_users, self.num_nodes, (i.size(0),), dtype=torch.long)  # (e*t)
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

        return neg_items_list.view(-1, num_negatives)
    
    def weighted_negative_sampling(self, users, pos_items, num_negatives: int = 1):
        i, j = users, pos_items
        
        all_ratings = self.probs_matrix[i]
        item_ratings = self.probs_matrix[i, j-self.num_users]
        item_ratings = item_ratings.reshape(-1, 1)
        weight_matrix = -torch.abs(all_ratings - item_ratings)
        weight_matrix = torch.softmax(weight_matrix, dim=1)

        idx_1 = i * self.num_items + j
        i = i.repeat(num_negatives)
        neg_items_list = torch.multinomial(weight_matrix, num_negatives, replacement=False)  # (e*t)
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

        return neg_items_list.view(-1, num_negatives)

    def margin_loss(self, users, pos_items, num_negatives, margin: float = 0.1):
        """
        Margin-based loss.
        :param edge_index: [2, num_edges]
        :param neg_edge_index: [2, num_edges]
        :param margin: float
        :return: float
        """
        out = self.compute_embedding()
        
        user_embedding = out[users]
        pos_item_embedding = out[pos_items]
        pos_scores = self.score_function(user_embedding, pos_item_embedding)
        
        neg_items_list = self.structured_negative_sampling(users, pos_items, num_negatives)
        
        neg_scores_list = []
        for i in range(num_negatives):
            neg_item_embedding = out[neg_items_list[:, i]]
            neg_scores_list.append(self.score_function(user_embedding, neg_item_embedding))
        neg_scores = torch.stack(neg_scores_list, dim=1)
        neg_scores = torch.mean(neg_scores, dim=1)
        
        loss = pos_scores - neg_scores + margin
        loss[loss < 0] = 0
        return loss.sum()
    
    def margin_loss_1(self, users, pos_items, neg_items_list, margin: float = 0.1):
        """
        Margin-based loss.
        :param edge_index: [2, num_edges]
        :param neg_edge_index: [2, num_edges]
        :param margin: float
        :return: float
        """
        out = self.compute_embedding()
        
        user_embedding = out[users]
        pos_item_embedding = out[pos_items]
        pos_scores = self.score_function(user_embedding, pos_item_embedding)
        
        num_negatives = neg_items_list.shape[1]
        neg_scores_list = []
        for i in range(num_negatives):
            neg_item_embedding = out[neg_items_list[:, i]]
            neg_scores_list.append(self.score_function(user_embedding, neg_item_embedding))
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
        
        self.probs_matrix = torch.tensor(probs_matrix)
        
        return probs_matrix
