import torch
from torch import Tensor
from manifolds import Hyperboloid, ManifoldParameter
from layers import LorentzLayer
from .base import Base


class LorentzGCN(Base):
    def __init__(self, num_users, num_items, graph, args):
        super().__init__(num_users, num_items, graph, args)
        self.c = torch.tensor([args.c]).cuda()
        self.manifold = Hyperboloid()
        
        self.embedding.weight = torch.nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))
        self.embedding.weight = ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)
        
        self.layers = torch.nn.ModuleList([LorentzLayer(self.manifold, self.c) for _ in range(self.num_layers)])

    def loss(self, edge_index, neg_edge_index, margin=1.0):
        return self.margin_loss(edge_index, neg_edge_index, margin)
    
    def score_function(self, src_embbeding: Tensor, dst_embedding: Tensor):
        return self.manifold.sqdist(src_embbeding, dst_embedding, c=self.c)
    
    def stack_layers(self, x):
        out = [x]
        
        for i in range(self.num_layers):
            out.append(self.layers[i](out[i], self.graph))
            
        return sum(out[1:])

    def compute_embedding(self):
        """
        Get embedding of all nodes.
        :return: [num_nodes, embedding_dim]
        """
        x = self.embedding.weight
        x = self.manifold.proj(x, c=self.c)

        out = self.stack_layers(x)
        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)

        return out
