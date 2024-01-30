"""
Implementation of Where2comm fusion.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.fuse_modules.graphformer_fuse import Graphformer

from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
# from opencood.models.graph_modules.GraphgpsFusion import GraphgpsFusion
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
        # 这里无需改成 communication_masks = torch.stack(communication_masks), 因为后续处理是直接对一整个批次的x进行mask操作, 故contact即可
    
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


class HeteGraphformerFuse(nn.Module):
    def __init__(self, args):
        super(HeteGraphformerFuse, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']
        self.point_cloud_range = args['point_cloud_range']
        self.max_cav = args['max_cav']

        self.fully = args['fully']
        if self.fully:
            print('constructing a fully connected communication graph')
        else:
            print('constructing a partially connected communication graph')

        # # 针对不同模型分别训练编码器
        self.proj_PointPillar = ProjEncoder(input_dim=256, out_dim=args['in_channels'], raito=3)

        self.proj_Second = ProjEncoder(input_dim=512, out_dim=args['in_channels'], raito=3)

        self.proj_VoxelNet = ProjEncoder(input_dim=128, out_dim=args['in_channels'], raito=3)
        
        self.channel_dict = {
                          "PointPillar": 256, 
                          "Second": 512, 
                          "VoxelNet": 128
                          }

        self.proj_dict = {
                          "PointPillar": self.proj_PointPillar, 
                          "Second": self.proj_Second, 
                          "VoxelNet":self.proj_VoxelNet
                          }


        
        # if args['fuse_mode'] == 'GPS':
        #     self.fuse_modules = GraphgpsFusion(args)
        if args['fuse_mode'] == 'Graphformer':
            self.fuse_modules = Graphformer(args)
        
        self.atten = AttentionFusion(args['in_channels'])

        self.naive_communication = Communication(args['communication'])
        
        self.fc1 = nn.Conv2d(args['in_channels'], args['in_channels']*2, 1)
        self.fc2 = nn.Conv2d(args['in_channels']*2, args['in_channels'], 1)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
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
        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0] 
 
        # feature_show(x[1], '/home/scz/hetecooper/hetecooper/pointpillar.png')
        # 1. Communication (mask the features)
        if self.fully:
            communication_rates = torch.tensor(1).to(x.device)
        else:
            # Prune
            batch_confidence_maps = self.regroup(psm_single, record_len)
            communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
            x = x * communication_masks
            
            # 若邻居信息非空, 同样对邻居信息进行预处理
            if neb_middle_info is not None:
                neb_batch_confidence_maps = self.regroup(neb_middle_info["psm_single"], neb_middle_info['record_len'])
                neb_feature = neb_middle_info['features_2d']
                # feature_show(neb_feature[1], '/home/scz/hetecooper/hetecooper/second-before.png')
                
                # 若特征通道数与映射模块输入维度不同, 随机丢弃/随机填充以对齐通道维度
                C_neb = neb_feature.size()[1]
                standard_channel = self.channel_dict[neb_middle_info['model_name']]
                if  C_neb != standard_channel:
                    origin_indices = list(range(C_neb))
                    if C_neb >= standard_channel:
                        retain_indices = random.sample(origin_indices, standard_channel)
                    elif C_neb < standard_channel:
                        retain_indices = random.sample(origin_indices, standard_channel - C_neb)
                        retain_indices.extend(origin_indices)
                    retain_indices.sort()
                    neb_feature = neb_feature[:,retain_indices]                    
                    
                # 若参与协作智能体encoder不同, 进行特征空间编码
                if neb_middle_info['model_name'] != middle_info['model_name']: 
                    neb_feature = self.proj_dict[neb_middle_info['model_name']](neb_feature)
                # feature_show(neb_feature[1], '/home/scz/hetecooper/hetecooper/second-after.png')
                # mask后再筛选无效特征
                neb_communication_masks, communication_rates = self.naive_communication(neb_batch_confidence_maps, B)
                neb_feature = neb_feature * neb_communication_masks
                # neb_feature = torch.zeros_like(neb_feature)
                neb_feature = self.regroup(neb_feature, neb_middle_info['record_len'])

        # 2. Split the features
        # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
        # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
        batch_node_features = self.regroup(x, record_len) # b, sum(n_cav), c, h, w


        # 3. Fusion
        x_fuse = []
        for b in range(B):
            neighbor_feature = batch_node_features[b]
            ego = neighbor_feature[0]

            if neighbor_feature.size()[0] < 2:
                # 若场景内仅ego, 直接将特征加入x_fuse
                x_fuse.append(neighbor_feature[0])
                continue
            # 多辆车的情况
            fuse_feature = [ego]
            # 可以进一步扩写成循环, 依次与每辆车融合
            n_cav = record_len[b]
            for i in range(1, min(n_cav, self.max_cav)):
                if neb_middle_info is not None:
                    # 若存在邻居信息
                    neb = neb_feature[b][i]
                    neb_confidence_map =  neb_batch_confidence_maps[b][i]
                    neb_point_cloud_range = neb_middle_info['point_cloud_range']
                else:
                    # 不存在邻居信息, 使用ego代替邻居信息
                    neb = neighbor_feature[i]
                    neb_confidence_map =  batch_confidence_maps[b][i]
                    neb_point_cloud_range = self.point_cloud_range
                # graph based fuse
                fuse_feature.append(self.fuse_modules(ego, neb, neb_confidence_map, neb_point_cloud_range))
            fuse_feature = torch.stack(fuse_feature)
            # neb_fuse = torch.mean(torch.stack(neb_fuse), dim=0)
            fuse_feature = self.atten(fuse_feature)
            # Feed Forward
            fuse_feature = F.relu(self.fc1(fuse_feature))
            fuse_feature = self.fc2(fuse_feature)
            x_fuse.append(fuse_feature)
            
            # # 两辆车的情况
            # if neb_middle_info is not None:
            #     # 若存在邻居信息
            #     neb = neb_feature[b][1]
            #     neb_confidence_map =  neb_batch_confidence_maps[b][1]
            #     neb_point_cloud_range = neb_middle_info['point_cloud_range']
            # else:
            #     # 不存在邻居信息, 使用ego代替邻居信息
            #     neb = neighbor_feature[1]
            #     neb_confidence_map =  batch_confidence_maps[b][1]
            #     neb_point_cloud_range = self.point_cloud_range
            # # graph based fuse
            # x_fuse.append(self.fuse_modules(ego, neb, neb_confidence_map, neb_point_cloud_range))

        # 将每个batch的特征重新组合在一起
        x_fuse = torch.stack(x_fuse)

        return x_fuse, communication_rates
