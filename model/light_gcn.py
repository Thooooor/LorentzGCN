from abc import ABC

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, SparseTensor, OptTensor
from torch_geometric.utils import spmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class LightGCN(torch.nn.Module):
    """LightGCN model."""

    def __init__(
            self,
            num_users: int,
            num_items: int,
            embedding_size=64,
            num_layers=3,
            dropout=0.2,
            alpha=None
    ):
        super(LightGCN, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = torch.nn.Embedding(self.num_nodes, embedding_size)
        self.convs = torch.nn.ModuleList([LightConv() for _ in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index: Adj):
        """

        :param edge_index: [2, num_edges]
        :return: [num_edges, 1]
        """
        out = self.get_embedding(edge_index)

        out_src = out[edge_index[0]]
        out_dst = out[edge_index[1]]
        return (out_src * out_dst).sum(dim=-1)

    def get_embedding(self, edge_index: Adj):
        """
        Get embedding of all nodes.
        :param edge_index: [2, num_edges]
        :return: [num_nodes, embedding_dim]
        """
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]

        return out

    def bpr_loss(self, edge_index: Adj, neg_edge_index: Adj):
        """
        Bayesian Personalized Ranking loss.
        :param edge_index: [2, num_edges]
        :param neg_edge_index: [2, num_edges]
        :return: loss value
        """
        pos_out = self.forward(edge_index)
        neg_out = self.forward(neg_edge_index)

        loss = - torch.log(torch.sigmoid(pos_out - neg_out)).mean()

        return loss

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


class LightConv(MessagePassing, ABC):
    """LightGCN layer."""

    def __init__(self, normalize=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(LightConv, self).__init__(**kwargs)
        self.normalize = normalize

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """

        :param x: [num_nodes, embedding_dim]
        :param edge_index: [2, num_edges]
        :param edge_weight: [num_edges]
        :return: [num_nodes, embedding_dim]
        """
        if self.normalize and isinstance(edge_index, Tensor):
            out = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                           add_self_loops=False, flow=self.flow, dtype=x.dtype)
            edge_index, edge_weight = out
        elif self.normalize and isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim),
                                  add_self_loops=False, flow=self.flow, dtype=x.dtype)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        """

        :param x_j: [num_edges, embedding_dim]
        :param edge_weight: [num_edges]
        :return: [num_edges, embedding_dim]
        """
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor = None) -> Tensor:
        """

        :param adj_t: [num_nodes, num_edges]
        :param x: [num_nodes, embedding_dim]
        :return: [num_nodes, embedding_dim]
        """
        return spmm(adj_t, x, self.node_dim, self.node_dim)
