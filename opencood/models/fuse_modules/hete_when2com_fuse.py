# -*- coding: utf-8 -*-
# Author: Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch.nn as nn
import torch

from opencood.models.fuse_modules.when2com import When2comFusion
from opencood.models.sub_modules.proj_encoder import ProjEncoder


DEBUG = False

class HeteWhen2comFuse(nn.Module):
    def __init__(self, args):
        super(HeteWhen2comFuse, self).__init__()

        self.max_cav = args['max_cav']

        # # 针对不同模型分别训练编码器
        self.proj_PointPillar = ProjEncoder(input_dim=256, out_dim=args['in_channels'], raito=3)

        self.proj_Second = ProjEncoder(input_dim=512, out_dim=args['in_channels'], raito=3)

        self.proj_VoxelNet = ProjEncoder(input_dim=128, out_dim=args['in_channels'], raito=3)

        self.proj_dict = {
                          "PointPillar": self.proj_PointPillar, 
                          "Second": self.proj_Second, 
                          "VoxelNet":self.proj_VoxelNet
                          }
        
        self.fusion_net = When2comFusion(args)


    def forward(self, middle_info, neb_middle_info=None):

        spatial_features_2d = middle_info['features_2d']
        record_len = middle_info['record_len']
        pairwise_t_matrix = middle_info['pairwise_t_matrix']

        if neb_middle_info:
            B = pairwise_t_matrix.shape[0] 
            N, _, H, W = spatial_features_2d.size()
            neb_spatial_features_2d = neb_middle_info['features_2d']
            # 若参与协作智能体encoder不同, 进行特征空间编码
            if neb_middle_info['model_name'] != middle_info['model_name']: 
                neb_spatial_features_2d = self.proj_dict[neb_middle_info['model_name']](neb_spatial_features_2d)
            neb_spatial_features_2d = torch.nn.functional.interpolate(neb_spatial_features_2d, size=(H, W), mode='bilinear', align_corners=True)
            # 计算要被替换的neb的索引, 在每一个record_len的轮次, 每个批次的第一个特征置零, 得到转换索引
            idx = [True]*N
            sum = 0
            for len in record_len:
                idx[sum] = False
                sum = sum + len

            spatial_features_2d[idx] = neb_spatial_features_2d[idx]

        fused_feature = self.fusion_net(spatial_features_2d,
                                        record_len,
                                        pairwise_t_matrix)

        return fused_feature, 1
