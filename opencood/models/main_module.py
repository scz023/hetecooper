import numpy as np
import torch.nn as nn
import glob
import importlib
import yaml
import sys
import os
import re
from datetime import datetime

import torch
import torch.optim as optim
from collections import OrderedDict
import timm

from opencood.utils.feature_show import feature_show


class MainModule(nn.Module):
    def __init__(self, hypes):
        super(MainModule, self).__init__()
        self.hypes = hypes
        # 因为异构情况下不同模型数据统一预处理, gride_size按照其他智能体设置计算会出错, 故在此处重新计算
        lidar_range = hypes['encoder']['args']['lidar_range']
        voxel_size = hypes['encoder']['args']['voxel_size']
        grid_size = (np.array(lidar_range[3:6]) -
                     np.array(lidar_range[0:3])) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        hypes['encoder']['args']['grid_size'] = grid_size

        self.encoder = self.create_sub_module('encoder')
        self.fuse = self.create_sub_module('fuse')
        self.head = self.create_sub_module('head')
    
    def forward(self, data_dict, neb_middle_info = None):
        middle_info = self.encoder(data_dict)
        psm_single = self.head(middle_info["features_2d"])['psm']
        middle_info["psm_single"] = psm_single
        # feature_show(middle_info["features_2d"][0], 'pp0.4.png')
        # feature_show(neb_middle_info["features_2d"][0], 'pp0.2.png')
        if neb_middle_info:
            fused_feature, communication_rates = self.fuse(middle_info, neb_middle_info)
        else:
            fused_feature, communication_rates = self.fuse(middle_info)
        
        ouput_dict = self.head(fused_feature)
        ouput_dict['com'] = communication_rates
        return ouput_dict


    def create_sub_module(self, sub_mode):
        """
        Import the module "models/[model_name].py

        Parameters
        __________
        hypes : dict
            Dictionary containing parameters.

        Returns
        -------
        model : opencood,object
            Model object.
        """
        backbone_name = self.hypes[sub_mode]['core_method']
        backbone_config = self.hypes[sub_mode]['args']

        model_filename = "opencood.models." + sub_mode + '_modules.' +backbone_name
        model_lib = importlib.import_module(model_filename)
        model = None
        target_model_name = backbone_name.replace('_', '')

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                model = cls

        if model is None:
            print('backbone not found in models folder. Please make sure you '
                'have a python file named %s and has a class '
                'called %s ignoring upper/lower case' % (model_filename,
                                                        target_model_name))
            exit(0)
        instance = model(backbone_config)
        return instance
