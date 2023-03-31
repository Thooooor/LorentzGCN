import torch
from torch import Tensor
from manifolds import Hyperboloid, ManifoldParameter
from layers import LorentzLayer
import numpy as np
from .base import Base


class LorentzGCN(Base):
    def __init__(self, num_users, num_items, graph, args):
        super().__init__(num_users, num_items, graph, args)
        self.c = torch.tensor([args.c]).cuda()
        self.manifold = Hyperboloid()
        self.graph = graph.cuda()

    def score_function(self, src_embbeding: Tensor, dst_embedding: Tensor):
        return self.manifold.dist(src_embbeding, dst_embedding, c=self.c)
