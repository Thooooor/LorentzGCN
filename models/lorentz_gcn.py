import torch
from torch import Tensor
from manifolds import Hyperboloid, ManifoldParameter
from layers import LorentzLayer
import numpy as np
from .base import Base


class LorentzGCN(Base):
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        graph, 
        embedding_dim=64, 
        num_layers=3,
        scale=0.1,
        margin=1.0,
        ):
        super().__init__(num_users, num_items, graph, embedding_dim, num_layers, scale, margin)
        self.c = torch.tensor([1]).cuda()
        self.manifold = Hyperboloid()
        self.graph = graph.cuda()

    def score_function(self, src_embbeding: Tensor, dst_embedding: Tensor):
        return self.manifold.dist(src_embbeding, dst_embedding, c=self.c)
        