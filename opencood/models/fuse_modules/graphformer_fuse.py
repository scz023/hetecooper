import math
from turtle import update
import dgl
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import time
import torch.nn.functional as F
import random
import numpy as np
from opencood.graph_utils.graph_utils_cuda import cal_overlap_xy, cal_overlap_dis, cal_overlap, mapping_edgeidx
# import networkx as nx


from opencood.models.fuse_modules.mswin import *
from opencood.models.sub_modules.base_transformer import *
from opencood.models.sub_modules.graph_edge_aggregate import EdgeAggregate
from opencood.models.sub_modules.graph_transformer_edge_layer_patches import GraphTransformerLayer
from opencood.utils.feature_show import feature_show

def dim_expand(input, out_dim):
    # 将输入张量input通过正/余弦位置编码提升到out_dim维
    # input: [n, c_in]
    # out_dim: c_out
    in_dim = input.size()[1]
    L = int(out_dim // (2*in_dim) + 1)
    freq_bands = ((2**torch.linspace(0, L-1, L).repeat_interleave(2)) * torch.pi).repeat(in_dim).to(input.device)

    theta = input.repeat_interleave(L*2, dim=1) * freq_bands
    output = torch.ones_like(theta).to(input.device)
    for d in range(in_dim):
        start = int(d*2*L)
        end = int((d+1)*2*L)
        output[:,start:end:2] = torch.sin(theta[:,start:end:2] )
        output[:,start+1:end:2] = torch.cos(theta[:,start+1:end:2])
    
    # only positive edge weight avalible
    # output = F.relu(output)

    return output[:, :out_dim]

def sc_padding(x, window_size):
    # calcute padding size and scaled h, w
    padding_left, padding_right, padding_top, padding_bottom = 0, 0, 0, 0
    _, h, w= x.size()
    h_sc, w_sc = h // window_size, w // window_size
    res_h = h % window_size
    
    if res_h > 0:
        h_sc = h_sc + 1
        padding_bottom = window_size - res_h
    res_w = w % window_size
    if res_w > 0:
        w_sc = w_sc + 1
        padding_right = window_size - res_w
    return [h_sc, w_sc], [padding_left, padding_right, padding_top, padding_bottom]

def sc_unpadding(x, padding):
    if padding[1] > 0:
        x = x[:, :, :-padding[1]]
    if padding[3] > 0:
        x = x[:, :-padding[3], :]
    return x

class XYEncoder(nn.Module):
    # 基于距离的位置编码器
    def __init__(self, dim):
        super(XYEncoder, self).__init__()
        # channels = (dim + 1) // 2
        self.linear = nn.Sequential(
                            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                            # nn.ReLU()
                            )
        # nn.init.xavier_normal_(self.linear[0].weight)
        # nn.init.constant_(self.linear[0].bias, 1e-4)
    
    def compute_pe(self, distance, C, H, W):
        distance = distance.repeat(C, 1, 1)
        # return distance

        dis_pe = torch.zeros(C, H, W, device = distance.device)
        
        c = torch.arange(0, C, 1, device=distance.device).repeat(H, W, 1).permute(2, 0, 1)
        # return c
        # st = time.time()
        # div_sin = torch.pow(10000, c[::2] / C)
        # equal to 1 / {10000(c/C)}
        div_sin = torch.exp(-c[::2]*(math.log(10000.0)/C) )
        # end = time.time()
        # print(end-st)
        # return c_sin
        if C % 2 == 0:
            div_cos = div_sin
        else:
            div_cos = div_sin[:-1]
        # return c_cos.size()
    
        # even use sim
        dis_pe[::2] = torch.sin(distance[::2] * div_sin) / math.sqrt(C)
        dis_pe[1::2] = torch.cos(distance[1::2]* div_cos) / math.sqrt(C)
        # dis_pe[::2] = torch.sin(distance[::2] * div_sin) 
        # dis_pe[1::2] = torch.cos(distance[1::2]* div_cos) 
        
        return dis_pe

        # dis_pe = F.relu(dis_pe)
        # return dis_pe
        # dis_pe.requires_grad = False
        # dis_pe = self.linear(dis_pe.unsqueeze(0)).squeeze(0)


    def forward(self, x, scale):
        # x : c, h, w
        # scale[0]-H-y, scale[1]-W-x
        C, H, W = x.size()
        
        # caculate distance from point to feature center
        
        i = torch.arange(0, H, 1, device=x.device).unsqueeze(0).t().repeat(1, W) 
        # print(i)
        j = torch.arange(0, W, 1, device=x.device).repeat(H, 1)

        # x_range = scale[0] * torch.abs(scale[0] * (j + 0.5 - W/2.0))
        # y_range = scale[1] * torch.abs(scale[1] * (H/2.0 - i - 0.5))
        x_range = scale[0] * scale[0] * (j + 0.5 - W/2.0)
        y_range = scale[1] * scale[1] * (H/2.0 - i - 0.5)

        # 全局位置编码也只考虑距离，而无正负号的差别
        # x_range = torch.abs(x_range)
        # y_range = torch.abs(y_range)

        x_pe = self.compute_pe(x_range, (C+1)//2, H, W)
        y_pe = self.compute_pe(y_range, C//2, H, W)
        
        x_pe.requires_grad = False
        y_pe.requires_grad = False

        # x 与 y 不存在耦合关系, 分别映射
        # x_pe = self.linear(x_pe.unsqueeze(0)).squeeze(0)
        # if C % 2 == 0:
        #     y_pe = self.linear(y_pe.unsqueeze(0)).squeeze(0)
        # else:
        #     tail = torch.zeros((1, H, W), device=x.device)
        #     y_pe = torch.cat((y_pe, tail), dim=0)
        #     y_pe = self.linear(y_pe.unsqueeze(0)).squeeze(0)
        #     y_pe = y_pe[:-1]
            
        
        dis_pe = torch.cat((x_pe, y_pe), dim=0)
        dis_pe = self.linear(dis_pe.unsqueeze(0)).squeeze(0)
        return dis_pe
        x = torch.where(x!=0, x + dis_pe, x)
        # x = x + dis_pe
        
        return x

class DisEncoder(nn.Module):
    # 基于距离的位置编码器
    def __init__(self, dim):
        super(DisEncoder, self).__init__()
        self.linear = nn.Sequential(
                            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                            # nn.ReLU()
                            )
        nn.init.xavier_normal_(self.linear[0].weight)
        nn.init.constant_(self.linear[0].bias, 0)

    def forward(self, x, point_cloud_range):
        # x : c, h, w
        
        
        C, H, W = x.size()
        
        # scale[0]-H, scale[1]-W
        scale = torch.tensor(((point_cloud_range[4] - point_cloud_range[1])/H, (point_cloud_range[3] - point_cloud_range[0])/W), device=x.device)
        
        # caculate distance from point to feature center
        
        i = torch.arange(0, H, 1, device=x.device).unsqueeze(0).t().repeat(1, W)
        # print(i)
        j = torch.arange(0, W, 1, device=x.device).repeat(H, 1)
        # print(j)
        distance = torch.sqrt( torch.square(scale[0] * (i - (H-1)/2.0)) + torch.square(scale[1] * (j - (W-1)/2.0)) ) 
        # return distance
        distance = distance.repeat(C, 1, 1)
        # return distance

        dis_pe = torch.zeros(C, H, W, device = x.device)
        
        c = torch.arange(0, C, 1, device=x.device).repeat(H, W, 1).permute(2, 0, 1)
        # return c
        # st = time.time()
        # div_sin = torch.pow(10000, c[::2] / C)
        # equal to 1 / {10000(c/C)}
        div_sin = torch.exp(-c[::2]*(math.log(10000.0)/C))
        # end = time.time()
        # print(end-st)
        # return c_sin
        if C % 2 == 0:
            div_cos = div_sin
        else:
            div_cos = div_sin[:-1]
        # return c_cos.size()
    
        # even use sim
        dis_pe[::2] = torch.sin(distance[::2] * div_sin) / math.sqrt(C)
        dis_pe[1::2] = torch.cos(distance[1::2]* div_cos) / math.sqrt(C)

        # dis_pe = F.relu(dis_pe)
        # return dis_pe
        dis_pe.requires_grad = False
        dis_pe = self.linear(dis_pe.unsqueeze(0)).squeeze(0)
        return dis_pe
        # x = x + dis_pe
        x = torch.where(x!=0, x + dis_pe, x)
        
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads, dropout=0.0, batch_norm = False):
        super(SelfAttention, self).__init__()
        inner_dim = in_dim // num_heads
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.Q = nn.Linear(in_dim, inner_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, inner_dim * num_heads, bias=True)
        self.V = nn.Linear(in_dim, inner_dim * num_heads, bias=True)
        self.out = nn.Linear(in_dim, in_dim, bias=True)

        self.FFN_layer1 = nn.Linear(in_dim, in_dim*2)
        self.FFN_layer2 = nn.Linear(in_dim*2, in_dim)

        if self.batch_norm:
            self.batch_norm = nn.BatchNorm1d(in_dim)



    def forward(self, h):
        # h: patches n c
        # patches, N, C = h.size()
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # patches n c1*m -> m patches n c1
        Q_h = rearrange(Q_h, 'patches n (c m) -> m patches n c', m = self.num_heads)
        K_h = rearrange(K_h, 'patches n (c m) -> m patches n c', m = self.num_heads)
        V_h = rearrange(V_h, 'patches n (c m) -> m patches n c', m = self.num_heads)

        qk = torch.einsum('m p i c, m p j c -> m p i j', Q_h, K_h)
        qk_scale = qk / np.sqrt(self.inner_dim)
        qk_scale = F.softmax(qk_scale, dim=-1)

        h_out = torch.einsum('m p i j, m p j c -> m p i c', qk_scale, V_h)

        h_out = rearrange(h_out, 'm patches n c -> patches n (m c)')
        h_out = self.out(h_out)

        h_out = self.FFN_layer1(h_out)
        h_out = F.relu(h_out)
        h_out = F.dropout(h_out, self.dropout, training=self.training)
        h_out = self.FFN_layer2(h_out)

        if self.batch_norm:
            h_out = self.batch_norm(h_out.permute(0, 2, 1)).permute(0, 2, 1)

        return h_out

class RatioMulXyEdgeEnc(nn.Module):
    def __init__(self, out_dim, inner_dim) -> None:
        super().__init__()
        self.out = nn.Linear(out_dim, out_dim, bias=True)
        self.out_dim = out_dim
        self.inner_dim = out_dim

    def compute_pe(self, distance, C, H, W):
        distance = distance.repeat(C, 1, 1)
        # return distance

        dis_pe = torch.zeros(C, H, W, device = distance.device)
        
        c = torch.arange(0, C, 1, device=distance.device).repeat(H, W, 1).permute(2, 0, 1)
        # return c
        # st = time.time()
        # div_sin = torch.pow(10000, c[::2] / C)
        # equal to 1 / {10000(c/C)}
        div_sin = torch.exp(-c[::2]*(math.log(10000.0)/C) )
        # end = time.time()
        # print(end-st)
        # return c_sin
        if C % 2 == 0:
            div_cos = div_sin
        else:
            div_cos = div_sin[:-1]
        # return c_cos.size()
    
        # even use sim
        dis_pe[::2] = torch.sin(distance[::2] * div_sin)
        dis_pe[1::2] = torch.cos(distance[1::2]* div_cos)
        # dis_pe[::2] = torch.sin(distance[::2] * div_sin) 
        # dis_pe[1::2] = torch.cos(distance[1::2]* div_cos) 
        
        return dis_pe

    def forward(self, values):
        # (ouverlap_ration + 1) * expand( x, y)
        C, H, W = self.inner_dim, values.size()[0], 2
        xy = values[:,2:4].unsqueeze(0)
        xy_pe = self.compute_pe(xy, (C+1)//2, H, W)
        # x和y方向上的编码组合起来, 形成位置编码
        xy_pe = xy_pe.permute(1, 2, 0).reshape(int(H), -1)[:, :C]
        xy_pe = xy_pe / math.sqrt(C/2)
        xy_pe = self.out(xy_pe)

        # 处理边权值中的overlap
        ratio = (values[:,1] + 1).unsqueeze(1).repeat(1, C) / 2
        
        e = xy_pe * ratio
        return e


class Graphformer(nn.Module):
    def __init__(self, args):
        super(Graphformer, self).__init__()
        # search_range: neb搜索search_range*voxel_size范围内的ego   
        # feature_dim: backbone提取得到特征的通道数
        # args['anchor_number']: 检测头的anchor数量，与confidence_map特征的维度相同
        feature_dim = args['in_channels']
        num_heads = args['head_num']
        n_layers = args['layer_num']
        layer_norm = args['layer_norm']
        batch_norm = args['batch_norm']
        residual = args['residual']
        self.search_range = args['search_range']
        self.max_nodes = args['max_nodes']
        self.point_cloud_range = args['point_cloud_range']
        self.global_atten = False
        self.split_atten = args['split_atten']            

        self.dropout = args['dropout']
        self.graph_dropout = args['graph_dropout']
        self.delta_xyz = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 1)
        )
        self.delta_ryp = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 1)
        )
        # 初始化参数为0
        for module in [self.delta_xyz, self.delta_ryp]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)


        # add mutiscale parameters here
        # 使用修改后的cfg全局变量进行初始化
        self.window_size = args['window_size']
        self.dis_enc = DisEncoder(feature_dim)
        self.edge_enc = nn.Sequential(
                            nn.Linear(3, feature_dim),
                            # nn.LeakyReLU()
                        )
        self.ratio_mul_xy_edge_enc = RatioMulXyEdgeEnc(feature_dim, 16)
        # self.scales = nn.ModuleList([])
        # for k in range(len(self.window_size)):
        #     self.layer = nn.ModuleList([GraphTransformerLayer(feature_dim, feature_dim, self.window_size[k], num_heads, self.dropout,
        #                                                         layer_norm, batch_norm, residual) \
        #                                                     for _ in range(n_layers-1) ]) 
        #     self.layer.append(GraphTransformerLayer(feature_dim, feature_dim, self.window_size[k], num_heads, \
        #                 self.dropout, layer_norm, batch_norm, residual))
            
        #     self.scales.append(self.layer)

        # Used for feature collaboration map visulization
        self.aggregate_edge = EdgeAggregate()

        self.layers = nn.ModuleList([GraphTransformerLayer(feature_dim, feature_dim, self.window_size[-1], num_heads, self.dropout,
                                                            layer_norm, batch_norm, residual) \
                                                        for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(feature_dim, feature_dim, self.window_size[-1], num_heads, \
                       self.dropout, layer_norm, batch_norm, residual))
        
        # self.global_atten = args['global_atten']
        if args['global_atten']['enabled']:
            self.global_atten = True
            # add global network here
            self.global_attetion = SelfAttention(feature_dim,
                                                 args['global_atten']['head_num'],
                                                 args['global_atten']['dropout'],
                                                 args['global_atten']['batch_norm'],
                                                 )
        if self.split_atten:
            self.out = SplitAttn(feature_dim)
        else:
            self.out = nn.Conv2d(in_channels=feature_dim * len(self.window_size), out_channels=feature_dim, kernel_size=1)



    def forward(self, ego, neb, neb_confidence_map, neb_point_cloud_range):
        # ego: 自身特征，input_dim: [256, 100, 252]
        # neb: 邻居特征，input_dim: [256, 100, 252]
        # neb_communication_maps: 邻居节点置信图
        # origin_bevfeature_size: 体素化后初始特征大小[64, 200, 504]，每个特征点表示真实世界中voxel_size大小空间内的特征
        
        # 距离位置编码
        ego_init = ego + self.dis_enc(ego, self.point_cloud_range)
        neb_init = neb + self.dis_enc(neb, self.point_cloud_range)
            
        # feature_show(ego_init, '/home/scz/hetecooper/hetecooper/ego_before.png')
        # feature_show(neb_init, '/home/scz/hetecooper/hetecooper/neb_before.png')
        # window_size = 1
        ups = []
        for k in range(len(self.window_size)):
            window_size = self.window_size[k]
            # rearrange ego and neb feature nodes to patches, if cannot be divided, padding
            scaled_size_ego, padding_ego = sc_padding(ego_init, window_size)
            scaled_size_neb, padding_neb = sc_padding(neb_init, window_size)
            ego = F.pad(ego_init, padding_ego)
            neb = F.pad(neb_init, padding_neb)
            ego = rearrange(ego, 'c (h_sc ws_h) (w_sc ws_w) -> (h_sc w_sc) (ws_h ws_w) c',
                            ws_h = window_size, ws_w = window_size)
            neb = rearrange(neb, 'c (h_sc ws_h) (w_sc ws_w) -> (h_sc w_sc) (ws_h ws_w) c',
                            ws_h = window_size, ws_w = window_size)
            div_idx, N, C = ego.size()
            # nodes = h_sc * w_sc, which is total feature points
            # n = ws_h * ws_w. which is number of window patches
            neb_nodes = torch.max(rearrange(neb, 'nodes n c -> nodes (n c)'), dim=1)[0]
            neb_nonzero_indices = torch.nonzero(neb_nodes > 1e-6).squeeze(1).to(torch.int).to(ego.device)
            # ego_nodes = torch.max(rearrange(ego, 'nodes n c -> nodes (n c)'), dim=1)[0]
            # ego_nonzero_indices = torch.nonzero(ego_nodes > 1e-6).squeeze(1).to(torch.int).to(ego.device)
            if neb_nonzero_indices.size()[0] > 0:
                indices, values = self.build_graph(torch.Tensor(scaled_size_ego).to(ego.device), \
                    scaled_size_neb, self.search_range*window_size, neb_nonzero_indices, neb_point_cloud_range)
                
                g = dgl.DGLGraph((indices[0], indices[1]))
                h = torch.cat((ego, neb), dim=0)
                
                # 根据相关关系计算边权重
                e = self.ratio_mul_xy_edge_enc(values)
                # e = torch.ones_like((values[:,1])).unsqueeze(1).repeat(1, 256)
                # e = values[:, 2].unsqueeze(1).repeat(1, C)
                
                e = e.unsqueeze(1).repeat(1, N, 1) # repeat edgeattr to adaptive nodes shapes

                # 特征协作图可视化
                h_edge = self.aggregate_edge(g, h, e)
                ego_edge = h_edge[:div_idx]
                neb_edge = h_edge[div_idx:]
                ego_edge = rearrange(ego_edge, '(h_sc w_sc) (ws_h ws_w) c -> c (h_sc ws_h) (w_sc ws_w)',
                                h_sc = scaled_size_ego[0], ws_h = window_size)
                ego_edge = sc_unpadding(ego_edge, padding_ego)

                neb_edge = rearrange(neb_edge, '(h_sc w_sc) (ws_h ws_w) c -> c (h_sc ws_h) (w_sc ws_w)',
                    h_sc = scaled_size_neb[0], ws_h = window_size)
                neb_edge = sc_unpadding(neb_edge, padding_neb)
                # feature_show(ego_edge, '/home/scz/hetecooper/hetecooper/ego_edge.png')

                for conv in self.layers:
                # for conv in self.scales[k]:
                    # loacal attention
                    h, e = conv(g, h, e)
                    # Per agent global attention
                    # if window_size>1 and self.global_atten:
                    #     ego = h[ego_nonzero_indices.to(torch.long)]
                    #     neb = h[div_idx + neb_nonzero_indices.to(torch.long)]

                    #     ego = self.global_attetion(ego)
                    #     neb = self.global_attetion(neb)
                        
                    #     h_new = torch.zeros_like(h)
                    #     h_new[ego_nonzero_indices.to(torch.long)] = ego
                    #     h_new[div_idx + neb_nonzero_indices.to(torch.long)] = neb
                    #     h = h_new
                        # h = torch.cat((ego, neb), dim=0)
                
                ego = h[:div_idx]
                neb = h[div_idx:]

            ego = rearrange(ego, '(h_sc w_sc) (ws_h ws_w) c -> c (h_sc ws_h) (w_sc ws_w)',
                            h_sc = scaled_size_ego[0], ws_h = window_size)
            ego = sc_unpadding(ego, padding_ego)
            # feature_show(ego, '/home/scz/hetecooper/hetecooper/ego_after.png')
            ups.append(ego)

            # neb = rearrange(neb, '(h_sc w_sc) (ws_h ws_w) c -> c (h_sc ws_h) (w_sc ws_w)',
            #     h_sc = scaled_size_neb[0], ws_h = window_size)
            # neb = sc_unpadding(neb, padding_neb)
        if self.split_atten:
            for k in range(len(ups)):
                # c h w -> B, L, H, W, C
                ups[k] = ups[k].permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
            ups = self.out(ups)
            ups = ups.squeeze(0).squeeze(0).permute(2, 0, 1)
        else:
            ups = torch.cat(ups, dim=0)
            ups = self.out(ups.unsqueeze(0)).squeeze(0)
        return ups

    # 建立邻接矩阵
    def build_graph(self, ego, neb, search_range, neb_nonzero_indices, neb_point_cloud_range):
        
        # 构建稀疏图
        # 计算用于寻找neb邻接ego节点的超参数
        H_ego, W_ego = ego
        H_neb, W_neb = neb
        # point_cloud_range到H W对应依据
        #  lidar_range:  [x_min, y_min, z_min, x_max, y_max, z_max]
        #  H = (y_max - y_min) / voxel_size
        #  W = (x_max - x_min) / voxel_size
        point_cloud_range = self.point_cloud_range
        # sc[0]-H, scz[1]-W
        sc_ego = torch.tensor(((point_cloud_range[4] - point_cloud_range[1])/H_ego, (point_cloud_range[3] - point_cloud_range[0])/W_ego), device=ego.device)
        sc_neb = torch.tensor(((neb_point_cloud_range[4] - neb_point_cloud_range[1])/H_neb, (neb_point_cloud_range[3] - neb_point_cloud_range[0])/W_neb), device=ego.device)

        neb_num = neb_nonzero_indices.size()[0]        
        # maximum nebs is 10
        MAX_NEB = int(self.max_nodes)
        indices = -1 * torch.ones((neb_num * MAX_NEB, 2), device=ego.device).to(torch.int) # 初始化indices值为-1，后续若没有赋值，则排除
        values = torch.zeros((neb_num * MAX_NEB, 4), device=ego.device).to(torch.float)
        
        # 计算neb节点与ego节点交叉部分面积，作为边权值
        # values-[real_distanse, overlap/S_ego],
        # indices-[neb_idx(source), ego_idx(target)], torch_geometric中默认消息传递方向: soure_2_targer, 这样即可实现将neb消息融合到ego
        cal_overlap_xy(MAX_NEB, search_range, neb_nonzero_indices.to(torch.int), sc_neb, sc_ego, H_ego, W_ego, H_neb, W_neb, indices, values) 
        # cal_overlap_dis(MAX_NEB, search_range, neb_nonzero_indices.to(torch.int), sc_neb, sc_ego, H_ego, W_ego, H_neb, W_neb, indices, values) 
        # cal_overlap(MAX_NEB, neb_nonzero_indices.to(torch.int), sc_neb, sc_ego, H_ego, W_ego, H_neb, W_neb, indices, values) 

        
        #筛选出合理的索引值, 排除预置的值为-1的索引
        mask = (indices[:, 0] >= 0) & (indices[:, 1] >= 0) & (indices[:, 0] < H_neb*W_neb) & (indices[:, 1] < H_ego*W_ego)
        # mask = torch.all((indices >= 0) & (indices < H_ego*W_ego), dim=1)
        indices = indices[mask].t()
        values = values[mask]

        # distance和overlap为0时，学不出信息，还可能影响其他参数， overlap可能有误差，提供一定的初始先验
        # values[:, 0] = self.search_range - values[:,0]
        # values[:, 1] = values[:, 1]
        
        # neb置信图在通道维度的最大值作为信息的可靠程度
        # neb_conf = neb_confidence_map.view(neb_confidence_map.size()[0], -1).sigmoid().max(dim=0)[0].squeeze(0)
        # # neb_conf.requires_grad = False
        # # ego信息熵softmax后与以做差作为对邻居特征的需求程度
        # ego_need = F.softmax(ego.view(C_ego, -1), dim=0)
        # ego_need = torch.clamp(ego_need, min=1e-5, max=0.999)
        # ego_need = - torch.sum(ego_need * torch.log(ego_need), dim=0)
        # ego_need = 1 - ego_need.sigmoid()

        # # ego_need取出协作需求度, neb_conf取出特征可靠度, 二者组合得到边特征
        # ego_index = indices[1].to(torch.long)
        # neb_index = indices[0].to(torch.long) 
        # # 组合overlap, ego_need, neb_conf成为边的权值
        # values = torch.stack((values, ego_need[ego_index], neb_conf[neb_index])).t()

        indices = indices.detach()
        indices.requires_grad = False

        values = values.detach()
        values.requires_grad = False

        # 为neb的索引加上ego的长度, 这是建图时构建节点集的操作为torch.cat(ego, neb)
        indices[0] = indices[0] + H_ego*W_ego
        
        # 添加ego-ego、neb-neb的连接
        # neb_index = torch.unique(neb_index)
        # len_neb = neb_index.size()[0]
        # neb_neb = torch.cat((neb_index-W_neb-1, neb_index-W_neb, neb_index-W_neb+1, neb_index-1, neb_index, neb_index+1, neb_index+W_neb-1,neb_index+W_neb, neb_index+W_neb+1,), dim=0) #与周围9个节点建立邻接关系
        # neb_indices = torch.stack([neb_index.repeat(9), neb_neb])
        # mask = torch.all((neb_indices >= 0) & (neb_indices < H_neb*W_neb), dim=0)
        # # neb_indices = neb_indices[:, mask] + H_ego * W_ego

        # ego_index = torch.unique(ego_index)
        # len_ego = ego_index.size()[0]
        # ego_ego = torch.cat((ego_index-W_ego-1, ego_index-W_ego, ego_index-W_ego+1, ego_index-1, ego_index, ego_index+1, ego_index+W_ego-1,ego_index+W_ego, ego_index+W_ego+1,), dim=0) #与周围9个节点建立邻接关系
        # ego_indices = torch.stack([ego_index.repeat(9), ego_ego])
        # mask = torch.all((ego_indices >= 0) & (ego_indices < H_ego*W_ego), dim=0)
        # ego_indices = ego_indices[:, mask]

        # #  并使用detach取消indices与values与来源的关联, 以防止错误的梯度传递
        # indices = torch.cat((indices, neb_indices, ego_indices), dim=1).detach()
        # values = torch.cat((values, torch.ones((len_neb *9 + len_ego*9, 3), device=ego.device))).detach()
        # indices.requires_grad = False
        # values.requires_grad = False


        # dropout层
        if self.training:
            # dropout后的outputs are scaled by a factor of  1 / ( 1- p) , 因此在这里需要还原回来
            mask = F.dropout(torch.ones_like(indices[0]).to(torch.float), self.graph_dropout, self.training) * (1-self.graph_dropout)
            mask =  (mask > 1e-5) | (mask < -1e-5 )
            # mask = torch.
            values = values[mask, :]
            indices = indices[:, mask]


        # 根据距离线性层计算定位误差值
        # dis = values[:,0].clone()
        # delta_xyz = self.delta_xyz(dis.unsqueeze(1)).squeeze()
        
        # x, y轴相对距离的绝对值
        values[:,2:4] = torch.abs(values[:,2:4]) 

        # delta_div_dis = delta_xyz / (dis + 1e-7) 
        # delta_x = delta_xyz * values[:, 2] / (dis + 1e-6)
        # delta_y = delta_xyz * values[:, 3] / (dis + 1e-6)

        values = values.clone()
        # delta_div_dis = torch.clamp(delta_div_dis, -2, 2)

        #扩展egi-->neb的边, 并修正边权值
        
        # 无向图条件下, ego-->neb边权值与neb-->ego相等
        # values[:,0] = values[:,0] + delta_xyz
        # neb_area = sc_neb[0]*sc_neb[1]
        # delta_div_n = delta_xyz**2 /neb_area
        # values[:, 0] = (values[:,0] + delta_xyz） / self.search_range
        # values[:,1] = (values[:,1] + delta_div_n) / (1 + delta_div_n)
        # values[:,2] = (values[:,2] + delta_div_dis) / (1 +  delta_div_dis)
        # values[:,3] = (values[:,3] +  delta_div_dis) / (1 +  delta_div_dis)
        # values = torch.cat((values, values), dim=0)  # n, 2
        # indices = torch.cat((indices, torch.stack([indices[1], indices[0]])), dim=1)    # 2, n
        
        
        # 有向图条件下, ego-->neb边权值重新计算
        # dis不变
        value_neb = torch.zeros_like(values)
        # dis误差修正, ego->neb 与 neb->ego统一
        # values[:,0] = values[:,0] + delta_xyz
        value_neb[:,0] = values[:,0] / self.search_range
        values[:, 0] = values[:,0] / self.search_range
        
        # overlap_ratio计算
        ego_area = sc_ego[0]*sc_ego[1]
        neb_area = sc_neb[0]*sc_neb[1]
        value_neb[:,1] = values[:,1]*neb_area/ego_area
        
        # neb--> ego overlap_ratio误差修正
        # delta_div_n = delta_xyz**2 /neb_area
        # values[:,1] = (values[:,1] + delta_div_n) / (1 + delta_div_n)
        # ego--> neb overlap_ratio误差修正
        # delta_div_n = delta_xyz**2 /ego_area
        # value_neb[:,1] = (value_neb[:,1] + delta_div_n) / (1 + delta_div_n)
        
        # sin与cos重新计算
        # values[:,2] = (values[:,2] + delta_div_dis) / (1 +  delta_div_dis)
        # values[:,3] = (values[:,3] +  delta_div_dis) / (1 +  delta_div_dis)
        # x 与 y 重新计算
        # values[:,2] = values[:,2] + delta_x
        # values[:,3] = values[:,3] + delta_y
        value_neb[:,2] = values[:,2] 
        value_neb[:,3] = values[:,3]
        # values[:,2:4] = torch.abs(values[:,2:4])
        values = torch.cat((values, value_neb), dim=0)
        indices = torch.cat((indices, torch.stack([indices[1], indices[0]])), dim=1)    # 2, n


        # 添加self_loop
        loop_back = torch.unique(indices[0])
        N_nodes = loop_back.size()[0]
        values = torch.cat((values, torch.cat((\
                                torch.zeros((N_nodes, 1), device=ego.device), \
                                torch.ones((N_nodes, 1), device=ego.device),\
                                torch.zeros((N_nodes, 1), device=ego.device), \
                                torch.zeros((N_nodes, 1), device=ego.device),\
                                    ), dim=1)), dim=0)
        indices = torch.cat((indices, loop_back.repeat(2, 1)), dim = 1)
        
        # 添加索引最大的边及其权值, 以保证索引能取到最大值, 并且保证至少有两条边
        values = torch.cat((values, torch.Tensor([[0.0, 1, 0.0 ,0.0]]).to(values.device)), dim=0)
        indices = torch.cat((indices, torch.Tensor([[H_ego*W_ego+H_neb*W_neb-1], [H_ego*W_ego+H_neb*W_neb-1]]).to(indices.device)), dim=1)

        values = torch.cat((values, torch.Tensor([[0.0, 1, 0.0, 0.0]]).to(values.device)), dim=0)
        indices = torch.cat((indices, torch.Tensor([[0], [0]]).to(indices.device)), dim=1)
        
        return indices.to(torch.long), values

        