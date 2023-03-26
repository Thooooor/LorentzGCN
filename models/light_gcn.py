import torch
from torch import Tensor
from torch_geometric.typing import Adj

from .base import Base
from layers import LightLayer


class LightGCN(Base):
    """LightGCN models."""

    def __init__(
            self,
            num_users: int,
            num_items: int,
            graph,
            embedding_size=64,
            num_layers=3,
            alpha=None
    ):
        super(LightGCN, self).__init__(
            num_users=num_users,
            num_items=num_items,
            graph=graph,
            embedding_size=embedding_size,
            num_layers=num_layers,
        )

        if alpha is None:
            alpha = 1. / (num_layers + 1)
        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.convs = torch.nn.ModuleList([LightLayer() for _ in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        torch.nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index: Adj):
        """

        :param edge_index: [2, num_edges]
        :return: [num_edges, 1]
        """
        out = self.compute_embedding()

        out_src = out[edge_index[0]]
        out_dst = out[edge_index[1]]

        return (out_src * out_dst).sum(dim=-1)

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
