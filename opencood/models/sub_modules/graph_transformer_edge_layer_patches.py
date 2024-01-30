import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer with edge features
    u == src == neb
    v == des == ego
    
"""

"""
    Util functions
"""

def tp_src_dot_dst(src_field, dst_field, re_pe, out_field):
    def func(edges):
        return {out_field: (edges.dst[dst_field] * (edges.src[src_field] + edges.data[re_pe]))}
    return func

def tp_src_mul_edge(src_field, edge_weight, re_pe, out_field):
    def func(edges):
        return {out_field: (edges.data[edge_weight] * (edges.src[src_field] + edges.data[re_pe]) )}
    return func

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

# Improving implicit attention scores with explicit edge features, if available
def relative_pe_attn(attn, relative_pe):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {attn: (edges.data[attn] + edges.data[relative_pe])}
    return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func

# cross atteion in ego and neb squares
def src_cross_dst(src_field, dst_field, out_field):
    def func(edges):
        _, N, _, _  = edges.src[src_field].size()
        # nodes, j, m ,c -> nodes, 1, j, m ,c -> nodes, j, j, m, c
        k =  edges.src[src_field].unsqueeze(1).repeat(1, N, 1, 1, 1)
        # nodes, i, m, c -> nodes, i, 1, m, c -> nodes, i, i, m, c
        q =  edges.src[dst_field].unsqueeze(2).repeat(1, 1, N, 1, 1)

        #  nodes, i, j, m, c
        return {out_field: k*q}
    return func


# Improving implicit attention scores with explicit edge features, if available
def cossdomain_relative_pe_attn(attn, relative_pe):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        _, N, _, _ = edges.data[relative_pe].size()
        return {attn: (edges.data[attn] + edges.data[relative_pe].unsqueeze(1).repeat(1, N, 1, 1, 1))}
    return func

# To copy edge features to be passed to FFN_e
def out_edge_features_patches(edge_feat, relative_pe):
    def func(edges):
        edge = torch.sum(edges.data[edge_feat], dim=2)
        return {'e_out': edge + edges.data[relative_pe]}
    return func

# patch wise message aggragation
def patch_src_mul_edge(edge_feat, src_field):
    def func(edges):
        C = edges.src[src_field].size()[-1]
        # atten = edges.data[edge_feat].squeeze(-1)
        x = edges.data[edge_feat].repeat(1, 1, 1, 1, C)
        y = edges.src[src_field]
        Vh = torch.einsum('e i j m o, e j m c -> e i m c',
                          edges.data[edge_feat], \
                            edges.src[src_field])
        return {src_field: Vh}
    return func

def sum_keys(in_field, out_field):
    def func(edges):
        return{out_field: torch.sum(edges.data[in_field], 2)}
    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, window_size, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.relative_indices = get_relative_distances(window_size) + \
                                window_size - 1
        # 每种相对位置分别学习偏置
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1,
                                                        2 * window_size - 1))
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.relativepe_e_k = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.relativepe_e_v = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.relativepe_e_k = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.relativepe_e_v = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score

        # 这里的计算要改, 现在只是区域间对应位置之间做点-点乘积, 而不是整个区域间的交叉注意力
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # add relative postion encoding for atten score
        g.apply_edges(relative_pe_attn('score', 'proj_e'))
        
        # Use available edge features to modify the scores
        # g.apply_edges(imp_exp_attn('score', 'proj_e'))
        
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        
        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def propagate_attention_typical(self, g):

        # add relative pe to Q_h
        # g.apply_edges(fn.u_add_e('K_h', 'k_e', 'K_h'))

        # Compute attention score
        g.apply_edges(tp_src_dot_dst('K_h', 'Q_h', 'k_e', 'score')) #, edges)
        # g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # add relative postion encoding for atten score
        # g.apply_edges(relative_pe_attn('score', 'proj_e'))
        
        # Use available edge features to modify the scores
        # g.apply_edges(imp_exp_attn('score', 'proj_e'))
        
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        
        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, tp_src_mul_edge('V_h', 'score', 'v_e', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def propagate_attention_patches(self, g):
        # Compute attention score

        # 这里的计算要改, 现在只是区域间对应位置之间做点-点乘积, 而不是整个区域间的交叉注意力
        # g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        # nodes i m c -> nodes i j m c
        g.apply_edges(src_cross_dst('K_h', 'Q_h', 'score')) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # add relative postion encoding for atten score
        # g.apply_edges(relative_pe_attn('score', 'proj_e'))
        
        # add crossdomain relative position encode
        g.apply_edges(cossdomain_relative_pe_attn('score', 'proj_e'))

        # sum scores across channel,
        # nodes i j m c
        # g.edata['score'] = torch.sum(g.edata['score'], dim=-1)

        # t = self.pos_embedding[self.relative_indices[:, :, 0],
        #                                self.relative_indices[:, :, 1]]

        # apply indomian relative position encode
        g.edata['score'] = g.edata['score'] + self.pos_embedding[self.relative_indices[:, :, 0], \
                                                                self.relative_indices[:, :, 1]] \
                                                    .unsqueeze(-1).unsqueeze(-1)
        
        # Use available edge features to modify the scores
        # g.apply_edges(imp_exp_attn('score', 'proj_e'))
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features_patches('score', 'proj_e'))

        # prepare for softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        g.update_all(patch_src_mul_edge('score', 'V_h'), fn.sum('V_h', 'wV'))

        # sum of attention values after exp
        eids = g.edges()
        # g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, sum_keys('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e):
        
        num_nodes, N, C = h.size()
        
        Q_h = self.Q(h)
        
        
        # 在这里加入deformable_attetion, 对h进行refine后,再生成k, v 并与q融合
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, N, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, N, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, N, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, N, self.num_heads, self.out_dim)
        

        # T5 type
        self.propagate_attention(g)
        # self.propagate_attention_patches(g)

        # Typical type
        # k_e = self.relativepe_e_k(e)
        # v_e = self.relativepe_e_v(e)
        # g.edata['k_e'] = k_e.view(-1, N, self.num_heads, self.out_dim)
        # g.edata['v_e'] = v_e.view(-1, N, self.num_heads, self.out_dim)
        # self.propagate_attention_typical(g)
        
        # sum(exp(atten_j) * values_j) / sum(exp(atten_j))
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
        e_out = g.edata['e_out']
        
        return h_out, e_out


"""
    If neb node feature have same directtion with ego,  Enhance feature
"""

def src_cosim_dst(field, out_field):
    def func(edges):
        scale = torch.norm(edges.src[field], p=2) * torch.norm(edges.dst[field], p=2) + 1e-6
        dot = edges.src[field] * edges.dst[field]
        # scale_dst = torch.sqrt(edges.dst[field])
        return {out_field: dot / scale}
    return func

def compute_diff(h_field, diff_field):
    def func(edges):
        return{diff_field: F.relu(edges.src[h_field] - edges.dst[h_field])}
    return func

def dst_cosim_diff(dst_field, diff_field, repe, out_field):
    def func(edges):
        scale = torch.norm(edges.data[diff_field], p=2, dim=-1, keepdim= True) * \
                torch.norm(edges.dst[dst_field], p=2, dim=-1, keepdim= True) + 1e-6
        dot = edges.dst[dst_field] * edges.data[diff_field]
        return{out_field: dot * scale + edges.data[repe]}
    return func

def dst_mul_diff(dst_field, diff_field, repe, out_field):
    def func(edges):
        return{out_field: edges.dst[dst_field] * edges.data[diff_field] + edges.data[repe]}
    return func

def score_mul_diffv(score_field, diffv_field, out_field):
    def func(edges):
        return{out_field: edges.data[score_field] * edges.data[diffv_field]}
    return func

def get_relative_distances(window_size):
    indices = torch.tensor(np.array(
        [[x, y] for x in range(window_size) for y in range(window_size)]))
    z = indices[None, :, :]
    k = indices[:, None, :]
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, window_size, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        # self.window_size = window_size
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, window_size, num_heads, use_bias)
        self.enhance = EnhanceLayer(in_dim, out_dim//num_heads, num_heads)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h, e):
        num_nodes, N, C = h.size()
        
        h_in1 = h # for first residual connection
        e_in1 = e # for first residual connection
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)
        
        # enhance codirect feature
        # h_diff_out = self.enhance(g, h, e)
        # h_attn_out = h_attn_out + h_diff_out
        
        h = h_attn_out.view(-1, N, self.out_channels)
        e = e_attn_out.view(-1, N, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        if self.num_heads > 1:
            h = self.O_h(h)
            e = self.O_e(e)

        if self.residual:
            h = h_in1 + h # residual connection
            e = e_in1 + e # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h.permute(0, 2, 1)).permute(0, 2, 1)
            e = self.batch_norm1_e(e.permute(0, 2, 1)).permute(0, 2, 1)

        h_in2 = h # for second residual connection
        e_in2 = e # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h # residual connection       
            e = e_in2 + e # residual connection  

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h.permute(0, 2, 1)).permute(0, 2, 1)
            e = self.batch_norm2_e(e.permute(0, 2, 1)).permute(0, 2, 1)           

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)


class EnhanceLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0):
        super().__init__()
        self.dropout=dropout
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        self.repe_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # nn.init.kaiming_normal_(self.p)
    
    def propagate_attention(self, g):
        # compute diff from src to dst, only positive delta are saved
        g.apply_edges(compute_diff('h', 'diff'))
    
        # genetare k and v vector from diff
        g.edata['k_diff'] = self.K(g.edata['diff'])
        g.edata['v_diff'] = self.V(g.edata['diff'])
        # g.edata['k_diff'] = g.edata['diff']
        # g.edata['v_diff'] = g.edata['diff']
        N = g.ndata['h'].size()[1]
        g.edata['k_diff'] = g.edata['k_diff'].view(-1, N, self.num_heads, self.out_dim)
        g.edata['v_diff'] = g.edata['v_diff'].view(-1, N, self.num_heads, self.out_dim)
        
        
        # compute attention score
        g.apply_edges(dst_cosim_diff('Q_h', 'k_diff', 'repe_e', 'score'))
        # x = g.edata['k_diff']
        # softmax
        # g.apply_edges(exp('score'))
        
        # apply score to v_diff
        g.apply_edges(score_mul_diffv('score', 'v_diff', 'msg'))
        
        # select maximum from msg
        g.update_all(fn.copy_edge('msg', 'msg'), fn.max('msg', 'h_out'))
        # g.update_all(fn.copy_edge('v_diff', 'v_diff'), fn.max('v_diff', 'h_out'))
        
        # compute the sum weight of neighbors, used to scale for more accurate h_out
        # g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
        
    
    def forward(self, g, h, e):
        num_nodes, N, C = h.size()
        
        Q_h = self.Q(h)
        # Q_h = h
        repe_e = self.repe_e(e)
        
        g.ndata['h'] = h
        g.ndata['Q_h'] = Q_h.view(-1, N, self.num_heads, self.out_dim)
        g.edata['repe_e'] = repe_e.view(-1, N, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        h_out = g.ndata['h_out']
        # h_out = g.ndata['h_out'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
             
        return h_out
