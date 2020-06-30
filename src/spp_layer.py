import torch 
import torch.nn as nn

class SPPLayer(nn.Module):
    def __init__(self, pool_num=[1, 4, 16], pool_type='max'):
        super(SPPLayer, self).__init__()
        self.pool_num = pool_num
        self.pool_type = pool_type
        self.name = "SpatialPyramidPooling"
        self.pool_layers = []

        if pool_type == 'max' :
            pool_func = nn.AdaptiveMaxPool2d
        elif pool_type == 'avg':
            pool_func = nn.AdaptiveAvgPool2d
        else :
            raise NotImplementedError(f"Unknown pooling type {pool_type},\
                                      expected 'max' or 'avg'")
        for n in self.pool_num:
            side_length = n**(1/2) 
            if side_length.is_integer():
                self.pool_layers.append(pool_func(int(side_length)))
            else :
                raise ValueError(f"{n} is not a square number")
                
        
    def forward(self, x):
        bs, c, h, w = x.size()
        out = []
        for layer in self.pool_layers:
            out.append(layer(x).view(bs, c, -1))
        
        return torch.cat(out, dim=-1)