import math
from turtle import update
import dgl
import torch
import torch.nn as nn
import torch.nn.init as init
import time
import torch.nn.functional as F
import random
import numpy as np
from opencood.graph_utils.graph_utils_cuda import cal_overlap_dis, cal_overlap, mapping_edgeidx
# import networkx as nx


from opencood.models.fuse_modules.mswin import *
from opencood.models.sub_modules.base_transformer import *
from opencood.models.sub_modules.graph_transformer_edge_layer import GraphTransformerLayer

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


class XYEncoder(nn.Module):
    # 基于距离的位置编码器
    def __init__(self, dim):
        super(XYEncoder, self).__init__()
        self.linear = nn.Sequential(
                            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                            # nn.ReLU()
                            )
        nn.init.xavier_normal_(self.linear[0].weight)
        nn.init.constant_(self.linear[0].bias, 0)
    
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
        return dis_pe

        # dis_pe = F.relu(dis_pe)
        # return dis_pe
        # dis_pe.requires_grad = False
        # dis_pe = self.linear(dis_pe.unsqueeze(0)).squeeze(0)


    def forward(self, x, scale):
        # x : c, h, w
        # scale[0]-H, scale[1]-W
        C, H, W = x.size()
        
        # caculate distance from point to feature center
        
        i = torch.arange(0, H, 1, device=x.device).unsqueeze(0).t().repeat(1, W)
        # print(i)
        j = torch.arange(0, W, 1, device=x.device).repeat(H, 1)

        x_range = scale[0] * (i )
        y_range = scale[1] * (j )

        x_pe = self.compute_pe(x_range, (C+1)//2, H, W)
        y_pe = self.compute_pe(y_range, C//2, H, W)

        dis_pe = torch.cat((x_pe, y_pe), dim=0)

        # dis_pe = F.relu(dis_pe)
        # return dis_pe
        dis_pe.requires_grad = False
        dis_pe = self.linear(dis_pe.unsqueeze(0)).squeeze(0)
        # return dis_pe
        x = x + dis_pe
        
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

    def forward(self, x, scale):
        # x : c, h, w
        # scale[0]-H, scale[1]-W
        C, H, W = x.size()
        
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

        # dis_pe = F.relu(dis_pe)
        # return dis_pe
        dis_pe.requires_grad = False
        dis_pe = self.linear(dis_pe.unsqueeze(0)).squeeze(0)
        # return dis_pe
        x = x + dis_pe
        
        return x


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
        self.point_cloud_range = args['point_cloud_range']
        pwindow_config = args['pwindow_att_config']
        feed_forward_config = args['feed_forward']
            

        self.dropout = args['dropout']
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

        # 使用修改后的cfg全局变量进行初始化
        self.dis_enc = DisEncoder(feature_dim)
        self.edge_enc = nn.Sequential(
                            nn.Linear(4, feature_dim),
                            # nn.LeakyReLU()
                        )

        self.layers = nn.ModuleList([GraphTransformerLayer(feature_dim, feature_dim, num_heads, self.dropout,
                                                            layer_norm, batch_norm, residual) \
                                                        for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(feature_dim, feature_dim, num_heads, \
                       self.dropout, layer_norm, batch_norm, residual))

        

        # self.mswin =  PreNorm(pwindow_config['dim'],
        #                         PyramidWindowAttention(pwindow_config['dim'],
        #                                        heads=pwindow_config['heads'],
        #                                        dim_heads=pwindow_config[
        #                                            'dim_head'],
        #                                        drop_out=pwindow_config[
        #                                            'dropout'],
        #                                        window_size=pwindow_config[
        #                                            'window_size'],
        #                                        relative_pos_embedding=
        #                                        pwindow_config[
        #                                            'relative_pos_embedding'],
        #                                        fuse_method=pwindow_config[
        #                                            'fusion_method']))
        # self.ff = PreNorm(pwindow_config['dim'],
        #                 FeedForward(pwindow_config['dim'],  feed_forward_config['mlp_dim'],
        #                             dropout=feed_forward_config['dropout']))


    def forward(self, ego, neb, neb_confidence_map, neb_point_cloud_range):
        # ego: 自身特征，input_dim: [256, 100, 252]
        # neb: 邻居特征，input_dim: [256, 100, 252]
        # neb_communication_maps: 邻居节点置信图
        # origin_bevfeature_size: 体素化后初始特征大小[64, 200, 504]，每个特征点表示真实世界中voxel_size大小空间内的特征


        # # 若neb发来的特征存在可用信息，执行图卷积
        neb_nodes = torch.max(neb.view(neb.size()[0], -1), dim=0)[0]
        neb_nonzero_indices = torch.nonzero(neb_nodes > 1e-4).squeeze(1).to(torch.int).to(ego.device)
        # neb_nonzero_indices = torch.nonzero(torch.max(neb.view(neb.size()[0], -1), dim=0)[0]).squeeze(1).to(torch.int).to(ego.device)
        if neb_nonzero_indices.size()[0] > 0:
            ego_res = ego

            # build_start = time.time()
            # 构建相关图
            indices, values, sc_ego, sc_neb = self.build_graph(ego, neb, neb_confidence_map, neb_nonzero_indices, neb_point_cloud_range)
            # build_end = time.time()
            # print("建图时间：", build_end - build_start)
            
            # 距离位置编码
            ego = self.dis_enc(ego, sc_ego)
            neb = self.dis_enc(neb, sc_neb)  
            # 角度位置编码 
                    
            # c h w ->  h*w c
            h = torch.cat((ego.view(ego.size()[0], -1).t(), neb.view(neb.size()[0], -1).t()), dim=0)
            
            # 边编码 cat{pe(values), x_j - x_i}
            # x_src = h[indices[0]]
            # x_des = h[indices[1]]
            # e = x_src-x_des
            # e = dim_expand(values, 4)
            # e = self.edge_enc(torch.cat((x_src-x_des, values), dim=1))
            # e = torch.ones_like(x_src)
            # e = dim_expand(values, h.size()[1]//10)
            # e = self.edge_enc(e)
            # e = values[:,1].unsqueeze(1)
            e = values[:,:]
            e = self.edge_enc(e)
            # e = values[:,:1]
            
            g = dgl.DGLGraph((indices[0], indices[1]))
            
            # convnets
            for conv in self.layers:
                h, e = conv(g, h, e)
            # g.ndata['h'] = h

            # 提取出ego特征
            _, H, W = ego.size()
            x = h[:H*W,:]
            x = x.t().view(ego.size())
             # Residual connection.之前的图卷积计算中, 为了减少计算量没有包含self_loop, 在此处重新加上x以补充信息
            # x = x + ego_res
            # x = self.layernorm(x)
            # mswin_start = time.time()
            # mswin提取多尺度注意力增强特征
            # c, h, w -> b=1, l=1, h, w, c
            # x = x.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 4, 2)
            # x = self.mswin(x) + x
            # x = self.ff(x) + x
            # # # 1 1 h w c -> c h w
            # x = x.squeeze(0).squeeze(0).permute(2, 0, 1)

        # mswin_end = time.time()
        # print("多尺度注意力时间：", mswin_end - mswin_start)
        else:
            x = ego
        # inf_end = time.time()
        # print("推理时间：", inf_end - inf_start)

        return x

    # 建立邻接矩阵
    def build_graph(self, ego, neb, neb_confidence_map, neb_nonzero_indices, neb_point_cloud_range):
        
        # 构建稀疏图
        # 计算用于寻找neb邻接ego节点的超参数
        C_ego, H_ego, W_ego = ego.size()
        C_neb, H_neb, W_neb = neb.size()
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
        MAX_NEB = int(4)
        indices = -1 * torch.ones((neb_num * MAX_NEB, 2), device=ego.device).to(torch.int) # 初始化indices值为-1，后续若没有赋值，则排除
        values = torch.zeros((neb_num * MAX_NEB, 4), device=ego.device).to(torch.float)
        
        # 计算neb节点与ego节点交叉部分面积，作为边权值
        # values-[real_distanse, overlap/S_ego],
        # indices-[neb_idx(source), ego_idx(target)], torch_geometric中默认消息传递方向: soure_2_targer, 这样即可实现将neb消息融合到ego
        cal_overlap_dis(MAX_NEB, self.search_range, neb_nonzero_indices.to(torch.int), sc_neb, sc_ego, H_ego, W_ego, H_neb, W_neb, indices, values) 
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

        # 归一化处理、仅保留值为正的边特征
        # values = F.softmax(values, dim=1)
        # values = F.relu(values)

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
            mask = F.dropout(torch.ones_like(indices[0]).to(torch.float), 0.5, self.training) * (1-0.5)
            mask =  (mask > 1e-5) | (mask < -1e-5 )
            # mask = torch.
            values = values[mask, :]
            indices = indices[:, mask]


        # 根据距离线性层计算定位误差值
        dis = values[:,0].clone()
        delta_xyz = self.delta_xyz(dis.unsqueeze(1)).squeeze()

        delta_div_dis = delta_xyz / (dis + 1e-7) 
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
        values[:,0] = values[:,0] + delta_xyz
        value_neb[:,0] = values[:,0] / self.search_range
        values[:, 0] = values[:,0] / self.search_range
        

        
        # overlap_ratio计算
        ego_area = sc_ego[0]*sc_ego[1]
        neb_area = sc_neb[0]*sc_neb[1]
        value_neb[:,1] = values[:,1]*neb_area/ego_area
        # neb--> ego overlap_ratio误差修正
        delta_div_n = delta_xyz**2 /neb_area
        values[:,1] = (values[:,1] + delta_div_n) / (1 + delta_div_n)
        # ego--> neb overlap_ratio误差修正
        delta_div_n = delta_xyz**2 /ego_area
        value_neb[:,1] = (value_neb[:,1] + delta_div_n) / (1 + delta_div_n)
        
        # sin与cos重新计算
        values[:,2] = (values[:,2] + delta_div_dis) / (1 +  delta_div_dis)
        values[:,3] = (values[:,3] +  delta_div_dis) / (1 +  delta_div_dis)
        value_neb[:,2] = values[:,2]
        value_neb[:,3] = -values[:,3]
        values = torch.cat((values, value_neb), dim=0)
        indices = torch.cat((indices, torch.stack([indices[1], indices[0]])), dim=1)    # 2, n


        # 添加self_loop
        loop_back = torch.unique(indices[0])
        N_nodes = loop_back.size()[0]
        values = torch.cat((values, torch.cat((\
                                torch.zeros((N_nodes, 1), device=ego.device), \
                                torch.ones((N_nodes, 1), device=ego.device),\
                                torch.zeros((N_nodes, 1), device=ego.device), \
                                torch.ones((N_nodes, 1), device=ego.device),\
                                    ), dim=1)), dim=0)
        indices = torch.cat((indices, loop_back.repeat(2, 1)), dim = 1)
        
        # 添加索引最大的边及其权值, 以保证索引能取到最大值, 并且保证至少有两条边
        values = torch.cat((values, torch.Tensor([[0.0, 1, 0 ,1]]).to(values.device)), dim=0)
        indices = torch.cat((indices, torch.Tensor([[H_ego*W_ego+H_neb*W_neb-1], [H_ego*W_ego+H_neb*W_neb-1]]).to(indices.device)), dim=1)

        values = torch.cat((values, torch.Tensor([[0.0, 1, 0, 1]]).to(values.device)), dim=0)
        indices = torch.cat((indices, torch.Tensor([[0], [0]]).to(indices.device)), dim=1)


        
        return indices.to(torch.long), values, sc_ego, sc_neb

        