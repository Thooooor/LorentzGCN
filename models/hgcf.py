import torch
from torch import Tensor
from manifolds import Hyperboloid, ManifoldParameter
from layers import HyperbolicLayer
from .base import Base


class HGCF(Base):
    def __init__(self, num_users, num_items, graph, args):
        super().__init__(num_users, num_items, graph, args)
        
        self.c = torch.tensor([args.c]).cuda()
        self.manifold = Hyperboloid()
        
        self.embedding.weight = torch.nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))
        self.embedding.weight = ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)

        hgc_layers = []
        hgc_layers.append(HyperbolicLayer(self.manifold, self.embedding_dim, self.c, args.network, self.num_layers))
        self.convs = torch.nn.Sequential(*hgc_layers)
        
        self.encode_graph = True
        
    def loss(self, edge_index, neg_edge_index):
        return self.margin_loss(edge_index, neg_edge_index, self.margin)
    
    def score_function(self, src_embbeding: Tensor, dst_embedding: Tensor):
        return self.manifold.sqdist(src_embbeding, dst_embedding, c=self.c)
        
    def compute_embedding(self):
        """
        Get embedding of all nodes.
        :return: [num_nodes, embedding_dim]
        """
        x = self.embedding.weight
        x_hyp = self.manifold.proj(x, c=self.c)
        
        if self.encode_graph is True:
            out, _ = self.convs((x_hyp, self.graph))
        else:
            out = self.convs(x_hyp)

        return out
