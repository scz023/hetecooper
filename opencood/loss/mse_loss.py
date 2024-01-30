# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()
        # self.mse = nn.MSELoss(reduce=False, reduction='sum')
    
    def forward(self, feature, feature_cooper_proj):
        # MSEloss = self.MSE(feature, feature_cooper_proj)
        # self.MSEloss = MSEloss

        MSEloss = self.mse(feature, feature_cooper_proj)
        self.MSEloss = MSEloss
        return MSEloss


    def logging(self, epoch, batch_id, batch_len, writer = None, pbar=None):
        # print(self.MSEloss)
        print_msg = "[epoch {}][{}/{}], || CMSE_Loss: {:.2f}".format(epoch, batch_id + 1, batch_len, self.MSEloss)   
        if pbar is None:
            print(print_msg)   
        else:
            pbar.set_description(print_msg)

            