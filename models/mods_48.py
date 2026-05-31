import torch
import torch.nn as nn
import warnings
import math
import torch.nn.functional as F
from einops import rearrange, repeat


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# class Dilated_Resblock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Dilated_Resblock, self).__init__()
#
#         sequence = list()
#         sequence += [
#             nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
#                       padding=1, dilation=(1, 1)),
#             nn.LeakyReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
#                       padding=2, dilation=(2, 2)),
#             nn.LeakyReLU(),
#             nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
#                       padding=3, dilation=(3, 3)),
#             nn.LeakyReLU(),
#             nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
#                       padding=2, dilation=(2, 2)),
#             nn.LeakyReLU(),
#             nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
#                       padding=1, dilation=(1, 1))
#
#         ]
#
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, x):
#         out = self.model(x) + x
#
#         return out


# class HFRM(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(HFRM, self).__init__()
#
#         self.conv_head = Depth_conv(in_channels, out_channels)
#
#         self.dilated_block_LH = Dilated_Resblock(out_channels, out_channels)
#         self.dilated_block_HL = Dilated_Resblock(out_channels, out_channels)
#
#         self.cross_attention0 = cross_attention(out_channels, num_heads=8)
#         self.dilated_block_HH = Dilated_Resblock(out_channels, out_channels)
#         self.conv_HH = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
#         self.cross_attention1 = cross_attention(out_channels, num_heads=8)
#
#         self.conv_tail = Depth_conv(out_channels, in_channels)
#
#
#     def forward(self, x):
#
#         b, c, h, w = x.shape
#
#         residual = x
#
#         x = self.conv_head(x)
#
#         x_HL, x_LH, x_HH = x[:b//3, ...], x[b//3:2*b//3, ...], x[2*b//3:, ...]
#
#         x_HH_LH = self.cross_attention0(x_LH, x_HH)
#         x_HH_HL = self.cross_attention1(x_HL, x_HH)
#
#         x_HL = self.dilated_block_HL(x_HL)
#         x_LH = self.dilated_block_LH(x_LH)
#
#         x_HH = self.dilated_block_HH(self.conv_HH(torch.cat((x_HH_LH, x_HH_HL), dim=1)))
#
#         out = self.conv_tail(torch.cat((x_HL, x_LH, x_HH), dim=0))
#
#         return out + residual

##################################HFEBlock####################################
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0) # B

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False) # B, C, 1
    # topk_values, topk_indices = distances.topk(k=4, largest=False)  # B, C, 1

    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)


    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )

    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)
    # indices = (
    #     torch.arange(0, topk_values.size(1))
    #     .unsqueeze(0)
    #     .repeat(batch_size, 1)
    #     .to(topk_values.device)
    # )
    # indices_selected = indices.masked_select(mask)
    # indices_selected = indices_selected.reshape(batch_size, num_matches)
    # filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    # return filtered_input_maps, filtered_candidate_maps
    return filtered_candidate_maps
def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, C, H*W)
    candidate_maps: (B, C, H*W)
    """
    distances = torch.cdist(input_maps, candidate_maps)  # B,C,C

    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)

class Matching(nn.Module):
    def __init__(self, dim=48, match_factor=1):
        super(Matching, self).__init__()
        self.num_matching = int(dim/match_factor)
    def forward(self, x, perception):
        b, c, h, w = x.size()
        x = x.flatten(2, 3)
        perception = perception.flatten(2, 3)
        # print('x, perception1', x.size(), perception.size())
        filtered_candidate_maps = neirest_neighbores_on_l2(x, perception, self.num_matching)
        # filtered_input_maps = filtered_input_maps.reshape(b, self.num_matching, h, w)
        filtered_candidate_maps = filtered_candidate_maps.reshape(b, self.num_matching, h, w)
        return filtered_candidate_maps

class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf//2, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out

class Matching_transformation(nn.Module):
    def __init__(self, dim=48, match_factor=1, ffn_expansion_factor=1, bias=True):
        super(Matching_transformation, self).__init__()
        self.num_matching = int(dim / match_factor)
        self.channel = dim
        hidden_features = int(self.channel * ffn_expansion_factor)
        self.matching = Matching(dim=dim, match_factor=match_factor)
        # self.matching = Matching(dim=dim)

        self.paconv =  PAConv(dim*2)

    def forward(self, x, perception):
        filtered_candidate_maps = self.matching(x, perception)
        # conv11 = self.conv11(concat)
        concat = torch.cat([x, filtered_candidate_maps], dim=1)
        out = self.paconv(concat)

        return out

class CMTAttention(nn.Module):
    def __init__(self, dim, num_heads, match_factor=4,ffn_expansion_factor=1,scale_factor=8, bias=True, attention_matching=True):
        super(CMTAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.matching = attention_matching
        if self.matching is True:
            self.matching_transformation = Matching_transformation(dim=dim,
                                                                   match_factor=match_factor,
                                                                   ffn_expansion_factor=ffn_expansion_factor,
                                                                   bias=bias)

    def forward(self, x, perception):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # perception = self.LayerNorm(perception)
        if self.matching is True:
            q = self.matching_transformation(q, perception)
        else:
            q = q
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim=48, match_factor=4, ffn_expansion_factor=1, bias=True, ffn_matching=True):
        super(FeedForward, self).__init__()
        self.num_matching = int(dim/match_factor)
        self.channel = dim
        self.matching = ffn_matching
        hidden_features = int(self.channel * ffn_expansion_factor)

        self.project_in = nn.Sequential(
            nn.Conv2d(self.channel, hidden_features, 1, bias=bias),
            nn.Conv2d(hidden_features, self.channel, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=bias)
        )
        if self.matching is True:
            self.matching_transformation = Matching_transformation(dim=dim,
                                                                   match_factor=match_factor,
                                                                   ffn_expansion_factor=ffn_expansion_factor,
                                                                   bias=bias)

        self.project_out = nn.Sequential(
            nn.Conv2d(self.channel, hidden_features, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features, self.channel, 1, bias=bias))

    def forward(self, x, perception):
        project_in = self.project_in(x)
        if perception is not None:
            out = self.matching_transformation(project_in, perception)
        else:
            out = project_in
        project_out = self.project_out(out)
        return project_out

class FeedForward_Restormer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=True):
        super(FeedForward_Restormer, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



class HFEBlock(nn.Module):
    def __init__(self, dim=48, num_heads=1, match_factor=4, ffn_expansion_factor=1, bias=True, attention_matching=True, ffn_matching=True, ffn_restormer=False):
        super(HFEBlock, self).__init__()
        self.dim =dim
        self.norm1 = LayerNorm2d(dim)
        self.attn = CMTAttention(dim=dim,
                              num_heads=num_heads,
                              match_factor=match_factor,
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias,
                              attention_matching=attention_matching)
        self.norm2 = LayerNorm2d(dim)
        self.ffn_restormer = ffn_restormer
        if self.ffn_restormer is False:
            self.ffn = FeedForward(dim=dim,
                                   match_factor=match_factor,
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias,
                                   ffn_matching=ffn_matching)
        else:
            self.ffn = FeedForward_Restormer(dim=dim,
                                             ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias)
        self.LayerNorm = LayerNorm2d(dim)

    def forward(self, x, perception):
        # print(" input_HFE_HF shape:", x.shape)
        # print(" input_HFE_perception1 shape:", perception.shape)
        percetion = self.LayerNorm(perception)
        # print(" input_HFE_perception2 shape:", perception.shape)
        x = x + self.attn(self.norm1(x), percetion)
        # print(" HFE_attn_x shape:", x.shape)
        if self.ffn_restormer is False:
            x = x + self.ffn(self.norm2(x), percetion)
        else:
            x = x + self.ffn(self.norm2(x))
        # print(" HFE_fmt_x shape:", x.shape)
        return x

# class Frequency_fusion(nn.Module):
#     def __init__(self, in_c=3, dim=48):
#         super(Frequency_fusion, self).__init__()
#         self.channel = in_c
#         self.conv11 = nn.Conv2d(3 * self.channel, dim, 1, 1)
#         self.dwconv = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1,
#                                 groups=dim)
#
#     def forward(self, feature1, feature2, feature3):
#         concat = torch.cat([feature1, feature2, feature3], dim=1)
#         conv11 = self.conv11(concat)
#         dwconv1, dwconv2 = self.dwconv(conv11).chunk(2, dim=1)
#         b, c, h, w = dwconv1.size()
#         dwconv1 = dwconv1.flatten(2, 3)
#         dwconv1 = F.softmax(dwconv1, dim=1)
#         dwconv1 = dwconv1.reshape(b, c, h, w)
#         perception = torch.mul(dwconv1, conv11) + dwconv2
#
#         return perception

class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

class HFRM(nn.Module):
    def __init__(self, in_channels, out_channels, wf=32, n_h_blocks=1):
        super().__init__()
        self.in_proj = Depth_conv(3, wf)
        self.h_out_conv = Depth_conv(wf, out_channels)

        self.h_fusion = SKFF(wf, height=3, reduction=8)
        self.h_blk = nn.Sequential(*[HFEBlock(wf, match_factor=1, ffn_expansion_factor=1) for _ in range(n_h_blocks)])

    def forward(self, x, perception):
        b, c, h, w = x.shape
        x_LH, x_HL, x_HH = x[:, :c // 3, ...], x[:, c // 3:2 * c // 3, ...], x[:, 2 * c // 3:, ...]
        x_LH = self.in_proj(x_LH)  # [B, wf, H, W]
        x_HL = self.in_proj(x_HL)
        x_HH = self.in_proj(x_HH)

        x_h = self.h_fusion([x_LH, x_HL, x_HH])
        for h_layer in self.h_blk:
            x_h = h_layer(x_h, perception)
        x_h = self.h_out_conv(x_h)

        return x_h



