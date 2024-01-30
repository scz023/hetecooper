import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, args):
        super(Mlp, self).__init__()
        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

    def forward(self, x):
        psm = self.cls_head(x)
        rm = self.reg_head(x)
        output_dict = {'psm': psm, 'rm': rm}

        return output_dict
