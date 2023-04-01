import torch
from torch import Tensor
from torch_geometric.typing import Adj

from .base import Base
from layers import LightLayer


class LightGCN(Base):
    """LightGCN models."""

    def __init__(self, num_users, num_items, graph, args):
        super(LightGCN, self).__init__(num_users, num_items, graph, args)

        alpha = 1. / (self.num_layers + 1)
        self.alpha = torch.tensor([alpha] * (self.num_layers + 1))

        self.convs = torch.nn.ModuleList([LightLayer() for _ in range(self.num_layers)])
    
    def loss(self, edge_index, neg_edge_index, lambda_reg = 1e-4):
        # return self.bpr_loss(edge_index, neg_edge_index, lambda_reg)
        return self.margin_loss(edge_index, neg_edge_index, self.margin)
    
    def score_function(self, src_embbeding: Tensor, dst_embedding: Tensor):
        # return (src_embbeding * dst_embedding).sum(dim=-1)
        return torch.sum((src_embbeding - dst_embedding).pow(2), dim=-1)

    def compute_embedding(self):
        """
        Get embedding of all nodes.
        :return: [num_nodes, embedding_dim]
        """
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, self.graph)
            out = out + x * self.alpha[i + 1]

        return out
