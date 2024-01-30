# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CosimLoss(nn.Module):
    def __init__(self):
        super(CosimLoss, self).__init__()
        # self.MSE = nn.MSELoss()
        self.cosim = nn.CosineSimilarity(dim=1)
    
    def forward(self, feature, feature_cooper_proj):
        # MSEloss = self.MSE(feature, feature_cooper_proj)
        # self.MSEloss = MSEloss

        Cosimloss = torch.mean(1 - self.cosim(feature, feature_cooper_proj))
        self.Cosimloss = Cosimloss
        return Cosimloss


    def logging(self, epoch, batch_id, batch_len, writer = None, pbar=None):
        print_msg = "[epoch {}][{}/{}], || Cosim_Loss: {:.2f}".format(epoch, batch_id + 1, batch_len, self.Cosimloss)   
        if pbar is None:
            print(print_msg)   
        else:
            pbar.set_description(print_msg)

            