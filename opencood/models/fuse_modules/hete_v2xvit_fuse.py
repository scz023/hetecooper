import torch
import torch.nn as nn


from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.fuse_modules.v2xvit_basic import V2XTEncoder
from opencood.models.sub_modules.proj_encoder import ProjEncoder


class V2XTransformer(nn.Module):
    def __init__(self, args):
        super(V2XTransformer, self).__init__()

        encoder_args = args['encoder']
        self.encoder = V2XTEncoder(encoder_args)

    def forward(self, x, mask, spatial_correction_matrix):
        output = self.encoder(x, mask, spatial_correction_matrix)
        output = output[:, 0]
        return output

class HeteV2xvitFuse(nn.Module):
    def __init__(self, args):
        super(HeteV2xvitFuse, self).__init__()

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

        self.fusion_net = V2XTransformer(args)


    def forward(self, middle_info, neb_middle_info=None):

        spatial_features_2d = middle_info['features_2d']
        record_len = middle_info['record_len']
        spatial_correction_matrix =  middle_info["spatial_correction_matrix"]

        prior_encoding = middle_info['prior_encoding'].unsqueeze(-1).unsqueeze(-1)
        
        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # 若邻居发来消息, 将除ego外的所有信息替换为Neb
        if neb_middle_info:
            _,_,H,W = spatial_features_2d.size()
            neb_spatial_features_2d = neb_middle_info['features_2d']
            # 若参与协作智能体encoder不同, 进行特征空间编码转换
            neb_spatial_features_2d = neb_middle_info['features_2d']
            if neb_middle_info['model_name'] != middle_info['model_name']: 
                neb_spatial_features_2d = self.proj_dict[neb_middle_info['model_name']](neb_spatial_features_2d)
            neb_spatial_features_2d = torch.nn.functional.interpolate(neb_spatial_features_2d, size=(H, W), mode='bilinear', align_corners=True)
            neb_regroup_feature, _ = regroup(neb_spatial_features_2d,
                                        record_len,
                                        self.max_cav)
            regroup_feature = torch.cat((regroup_feature[:,:1], neb_regroup_feature[:, 1:]), dim=1)


        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)


        return fused_feature, 1
