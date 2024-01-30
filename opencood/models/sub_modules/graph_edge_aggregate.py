import torch
import torch.nn as nn
import dgl.function as fn

class EdgeAggregate(nn.Module):
    """
        Param: 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, g, h, e):
        g.edata['e'] = e
        # Aggrate edge weights from all neighbor
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'edge_conn'))
        
        return g.ndata['edge_conn']