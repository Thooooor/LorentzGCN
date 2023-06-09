import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class HyperbolicLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, embbedding_dim, c, network, num_layers):
        super(HyperbolicLayer, self).__init__()
        self.agg = HyperbolicAggregate(manifold, c, embbedding_dim, network, num_layers)

    def forward(self, input):
        x, adj = input
        h = self.agg.forward(x, adj)
        output = h, adj
        return output


class StackGCNs(Module):
    def __init__(self, num_layers):
        super(StackGCNs, self).__init__()

        self.num_gcn_layers = num_layers - 1

    def plainGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def resSumGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return sum(output[1:])

    def resAddGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        if self.num_gcn_layers == 1:
            return torch.spmm(adj, x_tangent)
        for i in range(self.num_gcn_layers):
            if i == 0:
                output.append(torch.spmm(adj, output[i]))
            else:
                output.append(output[i] + torch.spmm(adj, output[i]))
        return output[-1]

    def denseGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            if i > 0:
                output.append(sum(output[1:i + 1]) + torch.spmm(adj, output[i]))
            else:
                output.append(torch.spmm(adj, output[i]))
        return output[-1]


class HyperbolicAggregate(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, network, num_layers):
        super(HyperbolicAggregate, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.stackGCNs = getattr(StackGCNs(num_layers), network)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        output = self.stackGCNs((x_tangent, adj))
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class LorentzLayer(nn.Module):
    """
    Lorentz graph convolution layer.
    """

    def __init__(self, manifold, c):
        super(LorentzLayer, self).__init__()
        self.manifold = manifold
        self.c = c

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        out = torch.spmm(adj, x_tangent)
        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        return out
    
    def extra_repr(self):
        return 'c={}'.format(self.c)
