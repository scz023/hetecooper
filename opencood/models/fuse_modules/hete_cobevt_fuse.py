import torch
import torch.nn as nn
from einops import rearrange, repeat

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.sub_modules.proj_encoder import ProjEncoder


class HeteCobevtFuse(nn.Module):
    def __init__(self, args):
        super(HeteCobevtFuse, self).__init__()
        
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

        self.fusion_net = SwapFusionEncoder(args)


    def forward(self, middle_info, neb_middle_info=None):
        
        spatial_features_2d = middle_info['features_2d']
        record_len = middle_info['record_len']
        pairwise_t_matrix = middle_info['pairwise_t_matrix']

        if neb_middle_info:
            B = pairwise_t_matrix.shape[0] 
            N, _, H, W = spatial_features_2d.size()
            neb_spatial_features_2d = neb_middle_info['features_2d']
            # 若参与协作智能体encoder不同, 进行特征空间编码转换
            if neb_middle_info['model_name'] != middle_info['model_name']: 
                neb_spatial_features_2d = self.proj_dict[neb_middle_info['model_name']](neb_spatial_features_2d)
            neb_spatial_features_2d = torch.nn.functional.interpolate(neb_spatial_features_2d, size=(H, W), mode='bilinear', align_corners=True)
            
            # 计算要被替换的neb的索引
            idx = [True]*N
            sum = 0
            for len in record_len:
                idx[sum] = False
                sum = sum + len

            spatial_features_2d[idx] = neb_spatial_features_2d[idx]

        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=regroup_feature.shape[3],
                          new_w=regroup_feature.shape[4])

        fused_feature = self.fusion_net(regroup_feature, com_mask)


        return fused_feature, 1
