from abc import ABC

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, SparseTensor, OptTensor
from torch_geometric.utils import spmm


class LightLayer(MessagePassing, ABC):
    """LightGCN layer."""

    def __init__(self, normalize=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(LightLayer, self).__init__(**kwargs)
        self.normalize = normalize

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
