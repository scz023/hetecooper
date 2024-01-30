# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        # self.MSE = nn.MSELoss()
        self.CEloss = nn.CrossEntropyLoss()
    
    def forward(self, feature, feature_cooper_proj):
        # MSEloss = self.MSE(feature, feature_cooper_proj)
        # self.MSEloss = MSEloss

        celoss = self.CEloss(feature, feature_cooper_proj)
        self.celoss = celoss
        return celoss


    def logging(self, epoch, batch_id, batch_len, writer = None, pbar=None):
        print_msg = "[epoch {}][{}/{}], || CEloss_Loss: {:.2f}".format(epoch, batch_id + 1, batch_len, self.celoss)   
        if pbar is None:
            print(print_msg)   
        else:
            pbar.set_description(print_msg)

            