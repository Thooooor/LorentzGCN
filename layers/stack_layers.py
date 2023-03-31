from torch.nn import Module


class StackLayers(Module):
    def __init__(self, layers, num_layers, adj):
        super(StackLayers, self).__init__()
        self.layers = layers
        self.num_layers = num_layers - 1
        self.adj = adj


class PlainLayers(StackLayers):
    def __init__(self, layers, num_layers, adj):
        super(PlainLayers, self).__init__(layers, num_layers, adj)

    def forward(self, x):
        out = [x]
        
        for i in range(self.num_layers):
            out.append(self.layers[i](self.adj, out[i]))
        
        return out[-1]


class ResSumLayers(StackLayers):
    def __init__(self, layers, num_layers, adj):
        super(ResSumLayers, self).__init__(layers, num_layers, adj)

    def forward(self, x):
        out = [x]
        
        for i in range(self.num_layers):
            out.append(self.layers[i](self.adj, out[i]))
        
        return sum(out[1:])
    

class ResAddLayers(StackLayers):
    def __init__(self, layers, num_layers, adj):
        super(ResAddLayers, self).__init__(layers, num_layers, adj)

    def forward(self, x):
        out = [x]
        
        if self.num_layers == 1:
            return self.layers(self.adj, x)
        
        for i in range(self.num_layers):
            if i == 0:
                out.append(self.layers[i](self.adj, out[i]))
            else:
                out.append(out[i] + self.layers[i](self.adj, out[i]))
        
        return out[-1]
    
    
class DenseLayers(StackLayers):
    def __init__(self, layers, num_layers, adj):
        super(DenseLayers, self).__init__(layers, num_layers, adj)

    def forward(self, x):
        out = [x]
        
        for i in range(self.num_layers):
            if i > 0:
                out.append(sum(out[1:i + 1]) + self.layers[i](self.adj, out[i]))
            else:
                out.append(self.layers[i](self.adj, out[i]))
        
        return out[-1]
