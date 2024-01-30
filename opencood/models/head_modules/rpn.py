import torch
import torch.nn as nn
import torch.nn.functional as F
# from opencood.models.voxel_net import RPN

# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x

# Region Proposal Network
class Rpn(nn.Module):
    def __init__(self, args):
        super(Rpn, self).__init__()
        self.block_1 = [Conv2d(128, 64, 3, 2, 1)]
        self.block_1 += [Conv2d(64, 64, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(64, 32, 3, 2, 1)]
        self.block_2 += [Conv2d(32, 32, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(32, 64, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(64, 64, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 4, 0),
                                      nn.BatchNorm2d(64))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(32, 64, 2, 2, 0),
                                      nn.BatchNorm2d(64))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(64, 64, 1, 1, 0),
                                      nn.BatchNorm2d(64))
        
        self.deconv_restore = nn.Sequential(nn.ConvTranspose2d(192, 192, 1, 2, 0, 1),
                                      nn.BatchNorm2d(192))
    

        self.score_head = Conv2d(192, args['anchor_num'], 1, 1, 0,
                                 activation=False, batch_norm=False)
        self.reg_head = Conv2d(192, 7 * args['anchor_num'] , 1, 1, 0,
                               activation=False, batch_norm=False)

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        x = torch.cat((x_0, x_1, x_2), 1)
        x = self.deconv_restore(x)
        output_dict = {'psm': self.score_head(x), 'rm': self.reg_head(x)}

        return output_dict

