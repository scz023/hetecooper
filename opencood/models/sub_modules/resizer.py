"""
Learnable Resizer
"""

import random
import torch
from torch import nn

from opencood.models.sub_modules.wg_fusion_modules import SwapFusionEncoder


class residual_block(nn.Module):
    def __init__(self, input_dim):
        super(residual_block, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(),
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim)
        )

    def forward(self, x):
        x = x + self.module(x)
        return x

class ChannelSelector(nn.Module):
    def __init__(self, out_dim, repeat = 2):
        super(ChannelSelector, self).__init__()
        self.out_dim = out_dim
        self.repeat = repeat
        self.proj =  nn.ModuleList([nn.Conv2d(2 * out_dim, out_dim, 1) for _ in range(repeat)])
        
    def forward(self, x):
        
        in_dim = 2 * self.out_dim
        for i in range(self.repeat):
            C = x.size()[1]
            # 随机丢弃/随机填充使x通道数为2*in_dim
            if C != in_dim :
                origin_indices = list(range(C))
                if C >= in_dim:
                    retain_indices = random.sample(origin_indices, in_dim)
                elif C < in_dim:
                    retain_indices = random.sample(origin_indices, in_dim - C)
                    retain_indices.extend(origin_indices)
                retain_indices.sort()
                x = x[:,retain_indices]  
            # 映射到与输入维度相同
            x = self.proj[i](x)
        
        return x


class Resizer(nn.Module):
    def __init__(self, args):
        super(Resizer, self).__init__()
        # channel selection
        self.channel_selector = ChannelSelector(args['out_dim'])
        # window+grid attention
        self.wg_att_1 = SwapFusionEncoder(args['wg_att'])
        self.bn = nn.BatchNorm2d(args['out_dim'])
        
        # window+grid attention
        self.wg_att_2 = SwapFusionEncoder(args['wg_att'])

        # residual blocks
        self.res_blocks =  nn.ModuleList([residual_block(args['out_dim']) \
                                        for _ in range(args['num_blocks'])])

    def forward(self, x, H, W):
        
        x = self.channel_selector(x)        
        res_x = x

        # wg attention
        x = self.wg_att_1(x)
        # Add Normalization
        x = x + res_x
        x = self.bn(x)
        # naive feature resizer
        x = torch.nn.functional.interpolate(x, [H, W], mode='bilinear', align_corners=False)
        
        # res blocks
        for res_bloc in self.res_blocks:
            x = res_bloc(x)
        
        # res
        x = torch.nn.functional.interpolate(res_x, [H, W], mode='bilinear', align_corners=False)
        
        # wg attention
        x = self.wg_att_2(x)
        
        return x
    
# if __name__ == "__main__":
#     args = { 'out_dim': 256,
#         'wg_att':{
#           'input_dim': 256,
#           'mlp_dim': 256,
#           'window_size': 2,
#           'agent_size': 2,
#           'dim_head': 32,
#           'drop_out': 0.1,
#           'depth': 1},
#         'num_blocks': 2}
#     re = Resizer(args)
#     x = torch.rand((2, 256, 100, 20))
#     x = re(x, 200, 100)
    # print(x.size())