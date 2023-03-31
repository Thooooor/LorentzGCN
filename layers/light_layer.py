import torch.nn as nn
from torch import Tensor, spmm


class LightLayer(nn.Module):
    def __init__(self, normalize=True, **kwargs):
        super(LightLayer, self).__init__()
        self.normalize = normalize
        
    def forward(self, x: Tensor, edge_index) -> Tensor:
        return spmm(edge_index, x)
