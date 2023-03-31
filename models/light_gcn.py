import torch
from torch import Tensor
from torch_geometric.typing import Adj

from .base import Base
from layers import LightLayer


class LightGCN(Base):
    """LightGCN models."""

    def __init__(self, num_users, num_items, graph, args):
        super(LightGCN, self).__init__(num_users, num_items, graph, args)

        if alpha is None:
            alpha = 1. / (self.num_layers + 1)
        if isinstance(alpha, Tensor):
            assert alpha.size(0) == self.num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (self.num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.convs = torch.nn.ModuleList([LightLayer() for _ in range(self.num_layers)])
    
    def score_function(self, src_embbeding: Tensor, dst_embedding: Tensor):
        return (src_embbeding * dst_embedding).sum(dim=-1)

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
