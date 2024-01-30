"""
This class is about swap fusion applications (also known as Fused Axial Attention)
"""
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from opencood.models.sub_modules.base_transformer import \
    FeedForward, PreNormResidual


# swap attention -> max_vit
class HGTAttention(nn.Module):
    """
    Unit Attention class. Todo: mask is not added yet.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        The agent can be different views, timestamps or vehicles.
    """

    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7,
            num_types=2,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        # self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for t in range(num_types):
            self.k_linears.append(nn.Linear(dim, dim))
            self.q_linears.append(nn.Linear(dim, dim))
            self.v_linears.append(nn.Linear(dim, dim))
            self.a_linears.append(nn.Linear(dim, dim))
        
        # 边的相关关系种类数为智能体种类数的平方
        num_relations=num_types*num_types
        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations, self.heads, dim_head, dim_head))
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations, self.heads, dim_head, dim_head))
        
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)


    def to_qkv(self, x, types):
        # x: (b, h, w, w_h, w_w, l, c) x: (B,H,W,L,C)
        # types: (B,L)
        q_batch = []
        k_batch = []
        v_batch = []

        for b in range(x.shape[0]):
            q_list = []
            k_list = []
            v_list = []

            for i in range(x.shape[-2]):
                # (h, w, w_h, w_w, l, c)  (H,W,1,C)
                q_list.append(
                    self.q_linears[types[b, i]](x[b, :, :, :, :, i, :].unsqueeze(2)))
                k_list.append(
                    self.k_linears[types[b, i]](x[b, :, :, :, :, i, :].unsqueeze(2)))
                v_list.append(
                    self.v_linears[types[b, i]](x[b, :, :, :, :, i, :].unsqueeze(2)))
            # (1,H,W,L,C)
            q_batch.append(torch.cat(q_list, dim=-2).unsqueeze(0))
            k_batch.append(torch.cat(k_list, dim=-2).unsqueeze(0))
            v_batch.append(torch.cat(v_list, dim=-2).unsqueeze(0))
        # (B,H,W,L,C)
        q = torch.cat(q_batch, dim=0)
        k = torch.cat(k_batch, dim=0)
        v = torch.cat(v_batch, dim=0)
        return q, k, v

    def get_relation_type_index(self, type1, type2):
        return type1 * self.num_types + type2

    def get_hetero_edge_weights(self, x, types):
        w_att_batch = []
        w_msg_batch = []

        for b in range(x.shape[0]):
            w_att_list = []
            w_msg_list = []

            for i in range(x.shape[-2]):
                w_att_i_list = []
                w_msg_i_list = []

                for j in range(x.shape[-2]):
                    e_type = self.get_relation_type_index(types[b, i],
                                                          types[b, j])
                    w_att_i_list.append(self.relation_att[e_type].unsqueeze(0))
                    w_msg_i_list.append(self.relation_msg[e_type].unsqueeze(0))
                w_att_list.append(torch.cat(w_att_i_list, dim=0).unsqueeze(0))
                w_msg_list.append(torch.cat(w_msg_i_list, dim=0).unsqueeze(0))

            w_att_batch.append(torch.cat(w_att_list, dim=0).unsqueeze(0))
            w_msg_batch.append(torch.cat(w_msg_list, dim=0).unsqueeze(0))

        # (B,M,L,L,C_head,C_head)
        w_att = torch.cat(w_att_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        w_msg = torch.cat(w_msg_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        return w_att, w_msg

    def forward(self, x, mask=None, type=0):
        # x shape: b, l, h, w, w_h, w_w, c
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.heads

        types = [[0] + (agent_size-1)*[type] for _ in range(batch)]
        
        # b, l, h, w, w_h, w_w, c -> b, h, w, w_h, w_w, l, c
        x = x.permute(0, 2, 3, 4, 5, 1, 6)
        
        # project for queries, keys, values
        qkv = self.to_qkv(x, types, batch, agent_size)
        
        # (B,M,L,L,C_head,C_head)， M means number of heads
        w_att, w_msg = self.get_hetero_edge_weights(x, types)
        
        
        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c',
                                          m=self.heads), (qkv))
        
        # flatten
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        
        # project for queries, keys, values
        qkv = self.to_qkv(x, types)
        
        # (B,M,L,L,C_head,C_head)
        w_att, w_msg = self.get_hetero_edge_weights(x, types)
        
        
        
        # old version
        
        # flatten
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float('inf'))

        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)


class HeteSwapFusionBlockMask(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention with
    mask enabled for multi-vehicle cooperation.
    """

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out,
                 num_types):
        super(HeteSwapFusionBlockMask, self).__init__()

        self.window_size = window_size

        self.window_attention = PreNormResidual(input_dim,
                                                HGTAttention(input_dim, dim_head,
                                                          drop_out,
                                                          agent_size,
                                                          window_size,
                                                          num_types=num_types))
        self.window_ffd = PreNormResidual(input_dim,
                                          FeedForward(input_dim, mlp_dim,
                                                      drop_out))
        
        self.grid_attention = PreNormResidual(input_dim,
                                              HGTAttention(input_dim, dim_head,
                                                        drop_out,
                                                        agent_size,
                                                        window_size,
                                                        num_types=num_types))
        self.grid_ffd = PreNormResidual(input_dim,
                                        FeedForward(input_dim, mlp_dim,
                                                    drop_out))

    def forward(self, x, mask, type):
        # x: b l c h w
        # mask: b h w 1 l
        # window attention -> grid attention
        mask_swap = mask

        # mask b h w 1 l -> b x y w1 w2 1 L
        mask_swap = rearrange(mask_swap,
                              'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        x = self.window_attention(x, mask=mask_swap)
        x = self.window_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')

        # grid attention
        mask_swap = mask
        mask_swap = rearrange(mask_swap,
                              'b (w1 x) (w2 y) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        x = self.grid_attention(x, mask=mask_swap)
        x = self.grid_ffd(x)
        x = rearrange(x, 'b l x y w1 w2 c -> b l c (w1 x) (w2 y)')

        return x


class HeteSwapFusionBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention.
    """

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out,
                 num_types):
        super(HeteSwapFusionBlock, self).__init__()
        # b = batch * max_cav
        self.block = nn.Sequential(
            Rearrange('b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, HGTAttention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (x w1) (y w2)'),

            Rearrange('b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, HGTAttention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (w1 x) (w2 y)'),
        )

    def forward(self, x, mask=None):
        # todo: add mask operation later for mulit-agents
        x = self.block(x)
        return x


class HeteSwapFusionEncoder(nn.Module):
    """
    Data rearrange -> swap block -> mlp_head
    """

    def __init__(self, args):
        super(HeteSwapFusionEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args['depth']

        # block related
        input_dim = args['input_dim']
        mlp_dim = args['mlp_dim']
        agent_size = args['agent_size']
        window_size = args['window_size']
        drop_out = args['drop_out']
        dim_head = args['dim_head']
        num_types = args['num_types']

        self.mask = False
        if 'mask' in args:
            self.mask = args['mask']

        for i in range(self.depth):
            if self.mask:
                block = HeteSwapFusionBlockMask(input_dim,
                                    mlp_dim,
                                    dim_head,
                                    window_size,
                                    agent_size,
                                    drop_out,
                                    num_types)

            else:
                block = HeteSwapFusionBlock(input_dim,
                                        mlp_dim,
                                        dim_head,
                                        window_size,
                                        agent_size,
                                        drop_out,
                                        num_types)
            self.layers.append(block)

        # mlp head
        self.mlp_head = nn.Sequential(
            Reduce('b m d h w -> b d h w', 'mean'),
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange('b h w d -> b d h w')
        )

    def forward(self, x, mask=None, type = 0):
        # x: B, L, C, H, W
        for stage in self.layers:
            x = stage(x, mask=mask, type=type)
        return self.mlp_head(x)


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = {'input_dim': 512,
            'mlp_dim': 512,
            'agent_size': 4,
            'window_size': 8,
            'dim_head': 4,
            'drop_out': 0.1,
            'depth': 2,
            'mask': True
            }
    block = HeteSwapFusionEncoder(args)
    block.cuda()
    test_data = torch.rand(1, 4, 512, 32, 32)
    test_data = test_data.cuda()
    mask = torch.ones(1, 32, 32, 1, 4)
    mask = mask.cuda()

    output = block(test_data, mask)
    print(output)
