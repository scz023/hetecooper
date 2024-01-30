"""
Implementation of Where2comm fusion.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
from opencood.models.sub_modules.proj_encoder import ProjEncoder
from opencood.utils.feature_show import feature_show


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """

        _, _, H, W = batch_confidence_maps[0].shape

        communication_masks = []
        communication_rates = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            L = communication_maps.shape[0]
            if self.training:
                # Official training proxy objective
                K = int(H * W * random.uniform(0, 1))
                communication_maps = communication_maps.reshape(L, H * W)
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.threshold:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)

            communication_rate = communication_mask.sum() / (L * H * W)
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        return communication_masks, communication_rates


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class HeteWhere2commFuse(nn.Module):
    def __init__(self, args):
        super(HeteWhere2commFuse, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

        self.fully = args['fully']
        if self.fully:
            print('constructing a fully connected communication graph')
        else:
            print('constructing a partially connected communication graph')


        # # 针对不同模型分别训练编码器
        self.proj_PointPillar = ProjEncoder(input_dim=256, out_dim=args['in_channels'], raito=3)

        self.proj_Second = ProjEncoder(input_dim=512, out_dim=args['in_channels'], raito=3)

        self.proj_VoxelNet = ProjEncoder(input_dim=128, out_dim=args['in_channels'], raito=3)
        
        # self.proj_PointPillar6 = ProjEncoder(input_dim=256, out_dim=args['in_channels'], raito=3)
        # 
        # self.proj_PointPillar5 = ProjEncoder(input_dim=256, out_dim=args['in_channels'], raito=3)
        
        # self.proj_PointPillar3 = ProjEncoder(input_dim=256, out_dim=args['in_channels'], raito=3)
        
        # self.proj_PointPillar2 = ProjEncoder(input_dim=256, out_dim=args['in_channels'], raito=3)


        self.proj_dict = {
                        #   "PointPillar6": self.proj_PointPillar6,
                        #   "PointPillar5": self.proj_PointPillar5,
                        #   "PointPillar3": self.proj_PointPillar3,
                        #   "PointPillar2": self.proj_PointPillar2,
                          "PointPillar": self.proj_PointPillar, 
                          "Second": self.proj_Second, 
                          "VoxelNet":self.proj_VoxelNet
                          }

        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttentionFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttentionFusion(args['in_channels'])

        self.naive_communication = Communication(args['communication'])



    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        # 由于每个batch的num_cav数不同，故无法用规则的张量表示，只能用list替代
        return split_x

    def forward(self, middle_info, neb_middle_info=None):
        """
        Fusion process (including preprocess, fuse_model, postprocess).

        Parameters:
            x: Input data, (B * sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """
        x = middle_info["features_2d"]
        pairwise_t_matrix = middle_info["pairwise_t_matrix"]
        psm_single = middle_info["psm_single"]
        record_len = middle_info['record_len']
        N, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0] 
 
        # feature_show(x[0], '/home/scz/hetecooper/hetecooper/pointpillar.png')
        # 1. Communication (mask the features)
        if self.fully:
            communication_rates = torch.tensor(1).to(x.device)
        else:
            # Prune
            batch_confidence_maps = self.regroup(psm_single, record_len)
            communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
            x = x * communication_masks
            
        # 若邻居信息非空, 对邻居信息进行预处理
        if neb_middle_info is not None:
            neb_batch_confidence_maps = self.regroup(neb_middle_info["psm_single"], neb_middle_info['record_len'])
            neb_feature = neb_middle_info['features_2d']
            # neb_feature = neb_feature * neb_communication_masks
            # 若参与协作智能体encoder不同, 进行特征空间编码
            # feature_show(neb_feature[0], '/home/scz/hetecooper/hetecooper/voxel-before.png')
            neb_feature = self.proj_dict[neb_middle_info['model_name']](neb_feature)
            # feature_show(neb_feature[0], '/home/scz/hetecooper/hetecooper/voxel-after.png')
            # if neb_middle_info['model_name'] != middle_info['model_name']: 
                # neb_feature = self.proj_dict[neb_middle_info['model_name']](neb_feature)
            # 同样对neb_feature和communication_masks进行放缩以防止直接对mask后的neb_feature执行插值带来的零损失问题
            neb_communication_masks, communication_rates = self.naive_communication(neb_batch_confidence_maps, B) 
            neb_feature = neb_feature * neb_communication_masks
            neb_feature = torch.nn.functional.interpolate(neb_feature, size=(H, W), mode='bilinear', align_corners=True)  
            # neb_feature = torch.zeros_like(neb_feature)
            # neb_communication_masks = torch.nn.functional.interpolate(neb_communication_masks, size=(H, W), mode='nearest')    
            
            # 将x每个batch后N个特征替换neb_feature对应的特征
            idx = [True]*N
            sum = 0
            for len in record_len:
                idx[sum] = False
                sum = sum + len
            x[idx] = neb_feature[idx]


        # 2. Split the features
        # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
        # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
        batch_node_features = self.regroup(x, record_len) # b, sum(n_cav), c, h, w


        # 3. Fusion
        x_fuse = []
        for b in range(B):
            fuse_feature = batch_node_features[b]
            x_fuse.append(self.fuse_modules(fuse_feature))

        x_fuse = torch.stack(x_fuse)

        return x_fuse, communication_rates