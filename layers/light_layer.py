import torch.nn as nn
from torch import Tensor, spmm


class LightLayer(nn.Module):
    def __init__(self, normalize=True, **kwargs):
        super(LightLayer, self).__init__()
        self.normalize = normalize
        
    def forward(self, x: Tensor, adj) -> Tensor:
        return spmm(adj, x)
