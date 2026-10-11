import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.GPFCA_checkpoint import GPFCA
from .kaconv.kaconv.fastkanconv import FastKANConvLayer as ConvKAN
import math
#from .layers import activations, normalizations
from utils import get_padding_2d, LearnableSigmoid_2d
from pesq import pesq
from joblib import Parallel, delayed
#from torchvision.ops.deform_conv import DeformConv2d
from SEMamba.models.DyTmamba_block import TFFMambaBlock as TFFMambaBlock
from SEMamba.models.DyTmamba_block import TFBMambaBlock as TFBMambaBlock
from models.MulMask_checkpoint import SRU
class ATTConvActNorm(nn.Module):
    def __init__(
            self,
            in_chan: int = 1,
            out_chan: int = 1,
            kernel_size: int = -1,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            padding: int = None,
            norm_type: str = None,
            act_type: str = None,
            n_freqs: int = -1,
            xavier_init: bool = False,
            bias: bool = True,
            is2d: bool = False,
            *args,
            **kwargs,
    ):
        super(ATTConvActNorm, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.n_freqs = n_freqs
        self.xavier_init = xavier_init
        self.bias = bias

        if self.padding is None:
            self.padding = 0  # if self.stride > 1 else "same"

        if kernel_size > 0:
            conv = nn.Conv2d if is2d else nn.Conv1d

            self.conv = conv(
                in_channels=self.in_chan,
                out_channels=self.out_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
            if self.xavier_init:
                nn.init.xavier_uniform_(self.conv.weight)
        else:
            self.conv = nn.Identity()

        self.act = activations.get(self.act_type)()
        self.norm = normalizations.get(self.norm_type)(
            (self.out_chan, self.n_freqs) if self.norm_type == "LayerNormalization4D" else self.out_chan
        )

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        output = self.act(output)
        output = self.norm(output)
        return output

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class MultiHeadSelfAttention2D(nn.Module):
    def __init__(
            self,
            in_chan: int,
            n_freqs: int,
            n_head: int = 4,
            hid_chan: int = 4,
            act_type: str = "prelu",
            norm_type: str = "LayerNormalization4D",
            dim: int = 3,
            *args,
            **kwargs,
    ):
        super(MultiHeadSelfAttention2D, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type
        self.dim = dim

        # assert self.in_chan % self.n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Keys.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Values.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.in_chan // self.n_head,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )

        self.attn_concat_proj = ATTConvActNorm(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.n_freqs,
            is2d=True,
        )

    def forward(self, x: torch.Tensor):
        # if self.dim == 4:
        #     x = x.transpose(-2, -1).contiguous()

        batch_size, _, time, freq = x.size()
        residual = x

        all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
        all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
        all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B', E, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]

        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        V = V.transpose(1, 2)  # [B', T, C/n_head, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
        emb_dim = Q.shape[-1]  # C*F/n_head

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim ** 0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]
        V = V.reshape(old_shape)  # [B', T, C/n_head, F]
        V = V.transpose(1, 2)  # [B', C/n_head, T, F]
        emb_dim = V.shape[1]  # C/n_head

        x = V.view([self.n_head, batch_size, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F]
        x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F]

        x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]
        x = self.attn_concat_proj(x)  # [B, C, T, F]

        x = x + residual

        # if self.dim == 4:
        #     x = x.transpose(-2, -1).contiguous()

        return x


class BiasNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(BiasNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # 可学习的逐通道偏置 (b)
        self.bias = nn.Parameter(torch.zeros(num_features))

        # 可学习的标量参数 (γ)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 确保输入是 2D 或更高维度 (batch_size, num_features, ...)
        if x.dim() < 2:
            raise ValueError("Input must have at least 2 dimensions.")

        # 计算 x - b
        x_centered = x - self.bias.view(1, -1, *([1] * (x.dim() - 2)))

        # 计算 RMS[x - b] (在通道维度上计算均方根值)
        rms = torch.sqrt(torch.mean(x_centered ** 2, dim=1, keepdim=True) + self.eps)

        # 归一化并应用缩放
        x_norm = x / rms
        x_norm = x_norm * torch.exp(self.gamma)

        return x_norm


class DWConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1, 1)
    ):
        super().__init__()

        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
                                                        stride=1, padding=1, bias=False, groups=in_ch),
                                              nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                        kernel_size=1,
                                                        stride=1, padding=0, bias=False)
                                              )
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=in_ch
        )
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        offset = self.offset_generator(x)

        if self.offset_clamp:
            offset = torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])
        x = self.dcn(x, offset)

        x = self.pwconv(x)
        x = self.act(x)
        return x


class Deform_Embedding(nn.Module):

    def __init__(self,
                 in_chans=64,
                 embed_dim=64,
                 patch_size=3,
                 stride=1,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1, 1)):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
            offset_clamp=offset_clamp
        )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x

class DyTanh(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_features,  1))  # 可学习缩放因子
        self.beta = nn.Parameter(torch.zeros(1, num_features,  1))  # 可学习偏移因子
        self.eps = eps  # 数值稳定性

    def forward(self, x):
        # 动态调整输入分布
        x = torch.tanh(self.alpha * x + self.beta)
        return x
class TRA(nn.Module):
    """Temporal Recurrent Attention (时间递归注意力)"""

    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(2 * channels, 2 * channels, batch_first=True)
        self.att_fc = nn.Linear(2 * channels, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """输入: (B,C,T,F), 输出: (B,C,T,F)"""
        B, C, T, frequency = x.shape
        # 计算能量 (B,C,T,F) -> (B,C,T)
        avg_energy = x.pow(2).mean(dim=-1)  # (B,C,T)
        max_energy = x.pow(2).max(dim=-1)[0] # (B,C,T)
        # 合并多尺度特征
        zt = torch.cat([avg_energy, max_energy], dim=1)  # (B,2C,1,T)
        zt = zt.squeeze(2).transpose(1, 2)  # (B,T,2C)

        # 后续GRU处理
        at, _ = self.att_gru(zt)  # (B,T,4C)
        at = self.att_fc(at)  # (B,T,C)
        at = self.att_act(at).transpose(1, 2)  # (B,C,T)

        # 扩展频率维度
        At = at.unsqueeze(-1).expand(-1, -1, -1, T)  # (B,C,T,F)
        return At


class FRA(nn.Module):
    """Frequency Recurrent Attention (对通道维度C求均值)"""

    def __init__(self, out_channels):
        super().__init__()
        # 第一层卷积：输入1通道（因对C求均值后通道数为1），输出out_channels
        self.conv_prelu = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=(3, 1), padding=(1, 0), dilation=(1, 1)),
            nn.PReLU()
        )
        # 第二层卷积：保持频率维度动态性
        self.conv_sigmoid = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (5, 1), padding=(4, 0), dilation=(2, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """输入: (B, C, T, F), 输出: (B, C, T, F)"""
        B, C, T, frequecy = x.shape
        # 对通道维度C求均值 -> (B, 1, T, F)
        zf = torch.mean(x, dim=1, keepdim=True)
        # 通过卷积生成频率注意力
        af = self.conv_prelu(zf)  # (B, F, T, F)
        af = self.conv_sigmoid(af)  # (B, F, T, F)
        return af


class cTFA(nn.Module):
    """Channel-wise Temporal-Frequency Attention"""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.tra = TRA(channels)
        self.fra = FRA(out_channels)
        self.norm = DyTanh(channels)
        self.post_norm = DyTanh(channels)
    def forward(self, x):
        """输入/输出: (B,C,T,F)"""
        residual = x
        # 获取时空注意力
        x = self.norm(x)
        At = self.tra(x)  # (B,C,T,F)
        Af = self.fra(x)  # (B,C,T,F)
        # 融合注意力 (矩阵乘法对F维度求和)
        # 归一化注意力权重
        At = F.softmax(At, dim=2)  # 时间维度归一化
        Af = F.softmax(Af, dim=3)  # 频率维度归一化
        attention = torch.einsum('bctt,bctf->bctf', At, Af)
        x = x * attention + residual
        # 特征加权
        return self.post_norm(x)
# class TRA(nn.Module):
#     """Temporal Recurrent Attention"""
#
#     def __init__(self, channels):
#         super().__init__()
#         self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
#         self.att_fc = nn.Linear(channels * 2, channels)
#         self.att_act = nn.Sigmoid()
#
#     def forward(self, x):
#         """x: (B,C,T,F)"""
#         b,c,t,f = x.shape
#         zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
#
#         at = self.att_gru(zt.transpose(1, 2))[0]
#         at = self.att_fc(at).transpose(1, 2)
#         at = self.att_act(at)
#         At = at[..., None]  # (B,C,T,1)
#         return  At.repeat(1,1,1,t) #/Amt      #用均方根缩小能量系数的动态范围#
# class FRA(nn.Module):
#     """Frequency Recurrent Attention"""
#
#     def __init__(self, channels=1,out_chan=64):
#         super().__init__()
#         self.att_conv_prelu = nn.Sequential(nn.Conv2d(channels, out_chan, (3,1),1,(1,0)),
#                                             nn.PReLU(out_chan))
#         self.att_conv = nn.Conv2d(out_chan, out_chan,(3,1),1,(1,0))
#         # self.att_act = nn.Sigmoid()
#
#     def forward(self, x):
#         """x: (B,C,T,F)"""
#         b,c,t,f = x.shape
#         zf = torch.mean(x.pow(2), dim=-3) # (B,1,T,F)
#         # print(zf.shape)
#         af = self.att_conv_prelu(zf.unsqueeze(1))
#         af = torch.sigmoid(self.att_conv(af))
#         return af #/Amt      #用均方根缩小能量系数的动态范围#
# class cTFA(nn.Module):
#     """Channel-wise Temporal Frequency Attention"""
#
#     def __init__(self, channels,out_chan):
#         super().__init__()
#         self.FITA = TRA(channels)
#         self.FDTA = FRA(out_chan=out_chan)
#     def forward(self, x):
#         """x: (B,C,T,F)"""
#         x_FITA = self.FITA(x)
#         x_FDTA = self.FDTA(x)
#         # print(x_FITA.shape)
#         # print(x_FDTA.shape)
#         x_map = torch.einsum('bcft,bctf->bctf', x_FITA, x_FDTA)  # 逐元素乘
#         # print(x_map.shape)
#         x = x * x_map
#         return x
class DS_DDAB(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DS_DDAB, self).__init__()
        self.h = h
        self.depth = depth
        # self.Deform_Embedding = Deform_Embedding(in_chans=h.dense_channel, embed_dim=h.dense_channel)
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h.dense_channel * (i + 1), h.dense_channel * (i + 1), kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, dilation=(dil, 1)), groups=h.dense_channel * (i + 1),
                          bias=True),
                nn.Conv2d(in_channels=h.dense_channel * (i + 1), out_channels=h.dense_channel, kernel_size=1, padding=0,
                          stride=1, groups=1,
                          bias=True),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel),
                Transpose(-1, -2),
            MultiHeadSelfAttention2D(h.dense_channel, n_freqs=1, n_head=4, hid_chan=4,
                                     act_type="prelu", norm_type="LayerNormalization4D", dim=4),
                nn.BatchNorm2d(h.dense_channel, affine=True),
                Transpose(-1, -2),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):

        skip = x
        for i in range(self.depth):
            # if i == 0:
            #     x = self.Deform_Embedding(x)
            #     x = self.dense_block[i](x)
            # else:
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x
class DS_DDB(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DS_DDB, self).__init__()
        self.h = h
        self.depth = depth
        # self.Deform_Embedding = Deform_Embedding(in_chans=h.dense_channel, embed_dim=h.dense_channel)
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h.dense_channel * (i + 1), h.dense_channel * (i + 1), kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, dilation=(dil, 1)), groups=h.dense_channel * (i + 1),
                          bias=True),
                nn.Conv2d(in_channels=h.dense_channel * (i + 1), out_channels=h.dense_channel, kernel_size=1, padding=0,
                          stride=1, groups=1,
                          bias=True),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):

        skip = x
        for i in range(self.depth):
            # if i == 0:
            #     x = self.Deform_Embedding(x)
            #     x = self.dense_block[i](x)
            # else:
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel, n_head=4, att_hid_chan=4, n_freq=1, ori_freq=201):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        # cTFA(h.dense_channel,h.dense_channel)
        )
        self.SRU = SRU(h.dense_channel, group_num=4, gate_treshold=0.5, fre=h.n_fft//2+1)
        self.dense_block = DS_DDB(h, depth=4)  # [b, h.dense_channel, ndim_time, h.n_fft//2+1]
        # self.TFA1 = cTFA(h.dense_channel,h.dense_channel)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            # cTFA(h.dense_channel, h.dense_channel)
        )
        # self.fp_layer = nn.Sequential(nn.BatchNorm2d(h.dense_channel),
        #                               FreMLP(h.dense_channel)
        #                               )
    def forward(self, x):
        x = self.dense_conv_1(x)   # [b, 64, T, F]

        x = self.dense_block(x)  # [b, 64, T, F]
        x = self.SRU(x)
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        # x = self.TFA2(x)
        return x
class Attn_Residual(nn.Module):
    def __init__(self, fn, scale_factor=1):
        super(Attn_Residual, self).__init__()
        self.fn = fn
        self.scale_factor = scale_factor

    def forward(self, x):
        return x + self.fn(x) * self.scale_factor
class DenseEncoder1(nn.Module):
    def __init__(self, h, in_channel, n_head=4, att_hid_chan=4, n_freq=1):
        super(DenseEncoder1, self).__init__()
        self.h = h


        # self.CE_1 = nn.Sequential(
        #     nn.Conv2d(in_channel, h.dense_channel // 8, 1),
        # #     nn.InstanceNorm2d(h.dense_channel // 8, affine=True),
        # #     nn.PReLU(h.dense_channel // 8),
        # #     nn.Conv2d(h.dense_channel // 8, h.dense_channel // 8, (3, 1), padding=(1, 0),groups=h.dense_channel // 8),
        # #     nn.InstanceNorm2d(h.dense_channel // 8, affine=True),
        # #     nn.PReLU(h.dense_channel // 8),
        # # cTFA(h.dense_channel // 8,h.dense_channel // 8)
        #     # Transpose(-1, -2),
        #     # MultiHeadSelfAttention2D(h.dense_channel // 8, n_freq, n_head=n_head, hid_chan=att_hid_chan,
        #     #                          act_type="prelu",
        #     #                          norm_type="LayerNormalization4D", dim=4),
        #     # nn.BatchNorm2d(h.dense_channel // 8),
        #     # Transpose(-1, -2),
        # )
        self.CE_2 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel // 4, 1),
            nn.InstanceNorm2d(h.dense_channel // 4, affine=True),
            nn.PReLU(h.dense_channel // 4),
            nn.Conv2d(h.dense_channel // 4, h.dense_channel // 4, (7, 1), padding=(3, 0),groups=h.dense_channel // 4),
            nn.InstanceNorm2d(h.dense_channel // 4, affine=True),
            nn.PReLU(h.dense_channel // 4),
        cTFA(h.dense_channel // 4,h.dense_channel // 4)
            # Transpose(-1, -2),
            # MultiHeadSelfAttention2D(h.dense_channel // 4, n_freq, n_head=n_head, hid_chan=att_hid_chan,
            #                          act_type="prelu",
            #                          norm_type="LayerNormalization4D", dim=4),
            # nn.BatchNorm2d(h.dense_channel // 4),
            # Transpose(-1, -2),
        )
        self.CE_3 = nn.Sequential(
            nn.Conv2d(h.dense_channel // 4, h.dense_channel//2, 1),
            nn.InstanceNorm2d(h.dense_channel // 2, affine=True),
            nn.PReLU(h.dense_channel // 2),
            nn.Conv2d(h.dense_channel // 2, h.dense_channel // 2, (13, 1), padding=(6, 0),groups=h.dense_channel // 2),
            nn.InstanceNorm2d(h.dense_channel // 2, affine=True),
            nn.PReLU(h.dense_channel // 2),
        cTFA(h.dense_channel // 2,h.dense_channel // 2)
            # Transpose(-1, -2),
            # MultiHeadSelfAttention2D(h.dense_channel // 2, n_freq, n_head=n_head, hid_chan=att_hid_chan,
            #                          act_type="prelu",
            #                          norm_type="LayerNormalization4D", dim=4),
            # nn.BatchNorm2d(h.dense_channel // 2),
            # Transpose(-1, -2),
        )

        self.CE_4 = nn.Sequential(
            nn.Conv2d(h.dense_channel // 2, h.dense_channel, 1),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, h.dense_channel, (23, 1), padding=(11, 0),groups=h.dense_channel),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        cTFA(h.dense_channel,h.dense_channel)
            # Transpose(-1, -2),
            # MultiHeadSelfAttention2D(h.dense_channel, n_freq, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
            #                          norm_type="LayerNormalization4D", dim=4),
            # nn.BatchNorm2d(h.dense_channel),
            # Transpose(-1, -2),
        )
        self.dense_block = DS_DDB(h, depth=4)  # [b, h.dense_channel, ndim_time, h.n_fft//2+1]
        self.dense_conv_1 = nn.Sequential((self.CE_2), (self.CE_3), (self.CE_4))
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        # print(x.shape)

        x = self.dense_block(x)  # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]

        return x

class PhaMaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaMaskDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.SP_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        # self.fp_layer = nn.Sequential(FreMLP(h.dense_channel),
        #                                nn.InstanceNorm2d(h.dense_channel, affine=True),
        #                                nn.PReLU(h.dense_channel))
        self.mask_pha = nn.Sequential(
            nn.Conv2d(h.dense_channel, 1, (1, 1)),
            nn.Tanh())
        self.SRU = SRU(h.dense_channel, group_num=4, gate_treshold=0.5, fre=h.n_fft//4)

    def forward(self, x):
        x = self.dense_block(x)

        # x = self.dysample(x)
        x = self.SRU(x)
        x = self.SP_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        pha = torch.atan2(x_i, x_r)
        # x = x * self.fp_layer(x)
        mask_pha = self.mask_pha(x)

        return pha, mask_pha

class PMaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PMaskDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.SP_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        # self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        # self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        # self.fp_layer = nn.Sequential(FreMLP(h.dense_channel),
        #                                nn.InstanceNorm2d(h.dense_channel, affine=True),
        #                                nn.PReLU(h.dense_channel))
        self.mask_pha = nn.Sequential(
            nn.Conv2d(h.dense_channel, 1, (1, 1)),
            nn.Tanh())
        self.SRU = SRU(h.dense_channel, group_num=4, gate_treshold=0.5, fre=h.n_fft//4)

    def forward(self, x):
        x = self.dense_block(x)

        # x = self.dysample(x)
        x = self.SRU(x)
        x = self.SP_conv(x)
        # x_r = self.phase_conv_r(x)
        # x_i = self.phase_conv_i(x)
        # pha = torch.atan2(x_i, x_r)
        # x = x * self.fp_layer(x)
        mask_pha = self.mask_pha(x)

        return mask_pha
class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
                                      # nn.InstanceNorm2d(h.dense_channel, affine=True),
                                      # nn.PReLU(h.dense_channel)
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        # self.fp_layer = nn.Sequential(nn.BatchNorm2d(h.dense_channel),
        #                               FreMLP(h.dense_channel)
        #                               )
        # self.final_conv = nn.Sequential(nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
        #     nn.InstanceNorm2d(out_channel, affine=True),
        #     nn.PReLU(out_channel),
        #     nn.Conv2d(out_channel, out_channel, (1, 1))
        # )
        self.SRU = SRU(h.dense_channel, group_num=4, gate_treshold=0.5, fre=h.n_fft//4)
        self.lsigmoid = LearnableSigmoid_2d(h.n_fft//2+1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.SRU(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x
class FreMLP(nn.Module):

    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.SRU = SRU(h.dense_channel, group_num=4, gate_treshold=0.5, fre=h.n_fft//4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            # cTFA(h.dense_channel, h.dense_channel)
        )
        # self.fp_layer = nn.Sequential(nn.BatchNorm2d(h.dense_channel),
        #                               FreMLP(h.dense_channel)
        #                               )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.SRU(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x

    # def __init__(self, h, out_channel=1):
    #     super(PhaseDecoder, self).__init__()
    #     self.dense_block = DS_DDB(h, depth=4)
    #     # self.dysample = DySample(h.dense_channel)
    #     self.phase_conv = nn.Sequential(
    #         nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
    #         nn.InstanceNorm2d(h.dense_channel, affine=True),
    #          nn.PReLU(h.dense_channel),
    #         nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
    #         # nn.InstanceNorm2d(out_channel, affine=True),
    #         # nn.PReLU(out_channel),
    #         # nn.Conv2d(out_channel, out_channel, (1, 1))
    #     )
    #     self.lsigmoid = LearnableSigmoid_2d(h.n_fft // 2 + 1, beta=h.beta)
    #
    # def forward(self, x):
    #     x = self.dense_block(x)
    #     # x = self.dysample(x)
    #     x = self.phase_conv(x)
    #     # x_r = self.phase_conv_r(x)
    #     # x_i = self.phase_conv_i(x)
    #     # x = torch.atan2(x_i, x_r)
    #     return x
# class MaskDecoder(nn.Module):
#     def __init__(self, h, out_channel=1,ori_channel=2):
#         super(MaskDecoder, self).__init__()
#         # self.dense_block = DS_DDB(h, depth=4)
#         self.CR_1 = nn.Sequential(
#             # MultiHeadSelfAttention2D(h.dense_channel , 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4),
#             # nn.BatchNorm2d(h.dense_channel),
#             nn.Conv2d(h.dense_channel, h.dense_channel // 2, (1, 23), padding=(0, 11)),
#             nn.InstanceNorm2d(h.dense_channel // 2, affine=True),
#             nn.PReLU(h.dense_channel // 2),
#             # MultiHeadSelfAttention2D(h.dense_channel // 2, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4),
#             # nn.BatchNorm2d(h.dense_channel // 2),
#         )
#         self.CR_2 = nn.Sequential(
#             nn.Conv2d(h.dense_channel // 2, h.dense_channel // 4, (1, 13), padding=(0, 6)),
#             nn.InstanceNorm2d(h.dense_channel // 4, affine=True),
#             nn.PReLU(h.dense_channel // 4),
#             # MultiHeadSelfAttention2D(h.dense_channel // 4, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4),
#             # nn.BatchNorm2d(h.dense_channel // 4),
#         )
#         self.CR_3 = nn.Sequential(
#             nn.Conv2d(h.dense_channel // 4, h.dense_channel // 8, (1, 7), padding=(0, 3)),
#             nn.InstanceNorm2d(h.dense_channel // 8, affine=True),
#             nn.PReLU(h.dense_channel // 8),
#             # MultiHeadSelfAttention2D(h.dense_channel // 8, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4),
#             # nn.BatchNorm2d(h.dense_channel // 8),
#         )
#         # self.dense_block = DS_DDB(h, depth=4)  # [b, h.dense_channel, ndim_time, h.n_fft//2+1]
#         self.CR_4 = nn.Sequential(
#             nn.Conv2d(h.dense_channel // 8, ori_channel, (1, 3), padding=(0, 1)),
#             nn.InstanceNorm2d(ori_channel, affine=True),
#             nn.PReLU(ori_channel),
#             # MultiHeadSelfAttention2D(ori_channel, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4)
#         )
#         self.dense_block = nn.Sequential(self.CR_1, self.CR_2, self.CR_3, self.CR_4)
#         self.mask_conv = nn.Sequential(
#             nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
#             nn.InstanceNorm2d(h.dense_channel, affine=True),
#             nn.PReLU(h.dense_channel),
#
#         )
#         self.final_conv = nn.Conv2d(ori_channel, out_channel, (1, 1))
#         self.lsigmoid = LearnableSigmoid_2d(h.n_fft // 2 + 1, beta=h.beta)
#
#     def forward(self, x):
#         x = self.mask_conv(x)
#         x = self.dense_block(x)
#         # x = self.dysample(x)
#         # x = self.mask_conv(x)
#         # print(x.shape)
#         x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
#         x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
#         return x
#
#
# class PhaseDecoder(nn.Module):
#     def __init__(self, h, out_channel=1,ori_channel=2):
#         super(PhaseDecoder, self).__init__()
#         # self.dense_block = DS_DDB(h, depth=4)
#         self.CR_1 = nn.Sequential(
#             # MultiHeadSelfAttention2D(h.dense_channel , 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4),
#             # nn.BatchNorm2d(h.dense_channel),
#             nn.Conv2d(h.dense_channel, h.dense_channel // 2, (1, 23), padding=(0, 11)),
#             nn.InstanceNorm2d(h.dense_channel // 2, affine=True),
#             nn.PReLU(h.dense_channel // 2),
#             # MultiHeadSelfAttention2D(h.dense_channel // 2, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4),
#             # nn.BatchNorm2d(h.dense_channel // 2),
#         )
#         self.CR_2 = nn.Sequential(
#             nn.Conv2d(h.dense_channel // 2, h.dense_channel // 4, (1, 13), padding=(0, 6)),
#             nn.InstanceNorm2d(h.dense_channel // 4, affine=True),
#             nn.PReLU(h.dense_channel // 4),
#             # MultiHeadSelfAttention2D(h.dense_channel // 4, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4),
#             # nn.BatchNorm2d(h.dense_channel // 4),
#         )
#         self.CR_3 = nn.Sequential(
#             nn.Conv2d(h.dense_channel // 4, h.dense_channel // 8, (1, 7), padding=(0, 3)),
#             nn.InstanceNorm2d(h.dense_channel // 8, affine=True),
#             nn.PReLU(h.dense_channel // 8),
#             # MultiHeadSelfAttention2D(h.dense_channel // 8, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4),
#             # nn.BatchNorm2d(h.dense_channel // 8),
#         )
#         # self.dense_block = DS_DDB(h, depth=4)  # [b, h.dense_channel, ndim_time, h.n_fft//2+1]
#         self.CR_4 = nn.Sequential(
#             nn.Conv2d(h.dense_channel // 8, ori_channel, (1, 3), padding=(0, 1)),
#             nn.InstanceNorm2d(ori_channel, affine=True),
#             nn.PReLU(ori_channel),
#             # MultiHeadSelfAttention2D(ori_channel, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu",
#             #                          norm_type="LayerNormalization4D", dim=4)
#         )
#         self.dense_block = nn.Sequential(self.CR_1, self.CR_2, self.CR_3, self.CR_4)
#         # self.dysample = DySample(h.dense_channel)
#         self.phase_conv = nn.Sequential(
#             nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
#             nn.InstanceNorm2d(h.dense_channel, affine=True),
#             nn.PReLU(h.dense_channel)
#         )
#         self.phase_conv_r = nn.Conv2d(ori_channel, out_channel, (1, 1))
#         self.phase_conv_i = nn.Conv2d(ori_channel, out_channel, (1, 1))
#
#     def forward(self, x):
#         x = self.phase_conv(x)
#         x = self.dense_block(x)
#         # x = self.dysample(x)
#         # x = self.phase_conv(x)
#         x_r = self.phase_conv_r(x)
#         x_i = self.phase_conv_i(x)
#         x = torch.atan2(x_i, x_r)
#         return x


class TS_BLOCK(nn.Module):
    def __init__(self, h):
        super(TS_BLOCK, self).__init__()
        self.h = h
        self.time = GPFCA(h.dense_channel)
        self.freq = GPFCA(h.dense_channel)
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
        # self.cTFA = cTFA(h.dense_channel, h.dense_channel)
        # self.cTFA2 = cTFA(h.dense_channel, h.dense_channel)
        # self.fp = nn.Sequential(nn.BatchNorm2d(h.dense_channel),
        #                         FreMLP(h.dense_channel),
        #                         nn.Sigmoid())
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        # x = x.reshape(b, c, t, f).contiguous()
        # x = self.cTFA(x)
        # x = x.reshape(b * f, t, c).contiguous()
        x = self.time(x) + x * self.beta
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        # x = x.reshape(b, c, t, f).contiguous()
        # x = self.cTFA2(x)
        # x = x.reshape(b * t, f, c).contiguous()
        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        # x = x * self.fp(x)
        return x
class TS_BLOCK(nn.Module):
    def __init__(self, h):
        super(TS_BLOCK, self).__init__()
        self.h = h
        self.time = GPFCA(h.dense_channel)
        self.freq = GPFCA(h.dense_channel)
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
        # self.cTFA = cTFA(h.dense_channel, h.dense_channel)
        # self.cTFA2 = cTFA(h.dense_channel, h.dense_channel)
        # self.fp = nn.Sequential(nn.BatchNorm2d(h.dense_channel),
        #                         FreMLP(h.dense_channel),
        #                         nn.Sigmoid())
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        # x = x.reshape(b, c, t, f).contiguous()
        # x = self.cTFA(x)
        # x = x.reshape(b * f, t, c).contiguous()
        x = self.time(x) + x * self.beta
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        # x = x.reshape(b, c, t, f).contiguous()
        # x = self.cTFA2(x)
        # x = x.reshape(b * t, f, c).contiguous()
        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        # x = x * self.fp(x)
        return x

class Transformer_BLOCK(nn.Module):
    def __init__(self, h):
        super(Tf_BLOCK, self).__init__()
        self.h = h
        # self.time_transformer = TransformerBlock(dim=h.dense_channel, n_heads=4)
        # self.freq_transformer = TransformerBlock(dim=h.dense_channel, n_heads=4)
        self.time_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x



class AdaptiveDecoder(nn.Module):
    def __init__(self, h, num_features, f_types,device='cuda:0'):  # out_channel=[M,P,C]
        super(AdaptiveDecoder, self).__init__()
        self.h = h

        self.dense_block = DS_DDB(h, depth=4)
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        # self.initial_conv = nn.Sequential(
        #     nn.ConvTranspose2d(ori_channel, ori_channel, (1, 3), (1, 2)),
        #     nn.InstanceNorm2d(ori_channel, affine=True),
        #     nn.PReLU(ori_channel)
        # )
        self.mask_conv = nn.Conv2d(h.dense_channel, 1, (1, 1))
        # self.mask_act = nn.Sequential(BiasNorm(1),nn.Softmax(dim=1))
        self.mask_act = LearnableSigmoid_2d(num_features)
        # self.LSigmoid2d = LearnableSigmoid_2d(num_features)
        self.weights = [nn.Parameter(torch.ones(1, h.dense_channel,  1, 1)).to(device),
                        nn.Parameter(torch.ones(1, h.dense_channel,  1, 1)).to(device)]
        # self.weights = [w.to(device) for w in self.weights]
        if f_types == "C":
            self.complex_spconv = nn.Sequential(
                nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.complex_conv = nn.Sequential(self.complex_spspconv,
            nn.Conv2d(h.dense_channel, 2, (1, 2)))
            # self.BiasNorm = BiasNorm(2)
        elif f_types == "P":
            self.phase_spconv = nn.Sequential(
                nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.phase_conv = nn.modules.ModuleList([nn.Conv2d(h.dense_channel, 1, (1, 1)) for i in range(2)])
            # self.BiasNorm = [BiasNorm(1).cuda(), BiasNorm(1).cuda()]
        elif f_types == "CP":
            self.complex_spconv = nn.Sequential(
                nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.complex_conv = nn.Sequential(self.complex_spconv,
                                              nn.Conv2d(h.dense_channel, 2, (1, 2)))
            self.phase_spconv = nn.Sequential(
                nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.phase_conv = nn.modules.ModuleList([nn.Conv2d(h.dense_channel, 1, (1, 1)) for i in range(2)])
            self.weights.append(nn.Parameter(torch.ones(1, h.dense_channel,  1, 1)).cuda())

    def forward(self, x, f_types):

        if f_types == "C":
            device = x.device
            # self.weights = [w.to(device) for w in self.weights]
            x = self.dense_block(x)
            x = self.initial_conv(x)
            x_mask = self.mask_act(self.mask_conv(x ).permute(0, 3, 2, 1).squeeze(-1)).permute(0, 2,
                                                                                                                1).unsqueeze(
                1)
            x_complex = self.complex_conv(x )
            return x_mask, x_complex
        if f_types == "P":
            device = x.device
            # print(x.shape)
            self.weights = [w.to(device) for w in self.weights]

            x = self.dense_block(x)
            x_p = self.phase_spconv(x * self.weights[1])
            x = self.initial_conv(x * self.weights[0])
            # print(x.shape)
            # x = self.initial_conv(x)
            x_mask = self.mask_act(self.mask_conv(x).permute(0, 3, 2, 1).squeeze(-1)).permute(0, 2, 1).unsqueeze(1)
            # x_p = x* self.weights[1]
            x_r = (self.phase_conv[0](x_p))
            x_i = (self.phase_conv[1](x_p))
            x_pha = torch.atan2(x_i, x_r)
            return x_mask, x_pha
        if f_types == "CP":
            device = x.device
            self.weights = [w.to(device) for w in self.weights]
            x = self.dense_block(x)
            x_complex = self.complex_conv(x*self.weights[2])
            x_p = self.phase_spconv(x*self.weights[1])
            x = self.initial_conv(x*self.weights[0])
            x_mask = self.mask_act(self.mask_conv(x ).permute(0, 3, 2, 1).squeeze(-1)).permute(0, 2, 1).unsqueeze(1)

            x_r = self.phase_conv[0](x_p)
            x_i = self.phase_conv[1](x_p)
            x_pha = torch.atan2(x_i, x_r)
            return x_mask, x_complex, x_pha


# class LKFCA_Net(nn.Module):
#     def __init__(self, h, num_tsblock=4):
#         super(LKFCA_Net, self).__init__()
#         self.h = h
#         self.num_tsblock = num_tsblock
#         self.dense_encoder = DenseEncoder(h, in_channel=2)
#         self.LKFCAnet = nn.ModuleList([])
#         for i in range(num_tsblock):
#             self.LKFCAnet.append(TS_BLOCK(h))
#         self.dense_decoder = AdaptiveDecoder(h, h.n_fft // 2 + 1, f_types="P")
#         # self.mask_decoder = MaskDecoder(h, out_channel=1)
#         # self.phase_decoder = PhaseDecoder(h, out_channel=1)
#
#     def forward(self, noisy_mag, noisy_pha):  # [B, F, T]
#         noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
#         noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
#         x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]
#         x = self.dense_encoder(x)
#
#         for i in range(self.num_tsblock):
#             x = self.LKFCAnet[i](x)
#         mask, phase = self.dense_decoder(x, f_types="P")
#         denoised_mag = (noisy_mag * mask).permute(0, 3, 2, 1).squeeze(-1)
#         denoised_pha = phase.permute(0, 3, 2, 1).squeeze(-1)
#         denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
#                                     denoised_mag * torch.sin(denoised_pha)), dim=-1)
#
#         return denoised_mag, denoised_pha, denoised_com

# class LKFCA_Net(nn.Module):
#     def __init__(self, h, num_tsblock=4):
#         super(LKFCA_Net, self).__init__()
#         self.h = h
#         self.num_tsblock = num_tsblock
#         self.dense_encoder = DenseEncoder(h, in_channel=4)
#         self.LKFCAnet = nn.ModuleList([])
#         for i in range(num_tsblock):
#             self.LKFCAnet.append(TS_BLOCK(h))
#         self.dense_decoder = AdaptiveDecoder(h, h.n_fft // 2 + 1, f_types="CP")
#         # self.mask_decoder = MaskDecoder(h, out_channel=1)
#         # self.phase_decoder = PhaseDecoder(h, out_channel=1)
#
#     def forward(self, noisy_mag, noisy_pha):  # [B, F, T]
#         noisy_com = torch.stack((noisy_mag * torch.cos(noisy_pha),
#                                  noisy_mag * torch.sin(noisy_pha)), dim=-1).permute(0, 3, 2, 1)
#         noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
#         noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]

#         x = torch.cat((noisy_mag, noisy_pha,noisy_com), dim=1)  # [B, 2, T, F]
#         x = self.dense_encoder(x)
#
#         for i in range(self.num_tsblock):
#             x = self.LKFCAnet[i](x)
#         mask, com, phase = self.dense_decoder(x, f_types="CP")
#         denoised_mag = (noisy_mag * mask).permute(0, 3, 2, 1).squeeze(-1)
#         denoised_pha = phase.permute(0, 3, 2, 1).squeeze(-1)
#         denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
#                                     denoised_mag * torch.sin(denoised_pha)), dim=-1) + com.permute(0, 3, 2, 1)
#
#         return denoised_mag, denoised_pha, denoised_com
# class LKFCA_Net(nn.Module):
#     def __init__(self, h, num_tsblock=4):
#         super(LKFCA_Net, self).__init__()
#         self.h = h
#         self.num_tsblock = num_tsblock
#         self.dense_encoder = DenseEncoder(h, in_channel=2)
#         self.LKFCAnet = nn.ModuleList([])
#         for i in range(num_tsblock):
#             self.LKFCAnet.append(TS_BLOCK(h))
#         self.mask_decoder = MaskDecoder(h, out_channel=1)
#         self.phase_decoder = PhaseDecoder(h, out_channel=1)
#         # self.phasemask_decoder = PhaMaskDecoder(h, out_channel=1)
#         # self.p = nn.Parameter(torch.zeros(1,  1, 1), requires_grad=True)
#     def forward(self, noisy_mag, noisy_pha): # [B, F, T]
#         noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
#         noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
#         x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
#         x = self.dense_encoder(x)
#
#         for i in range(self.num_tsblock):
#             x = self.LKFCAnet[i](x)
#
#         denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
#         denoised_pha = (self.phase_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
#         # masking_pha = (self.phasemask_decoder(x)*noisy_pha).permute(0, 3, 2, 1).squeeze(-1)
#         # denoised_pha = denoised_pha + self.p*masking_pha
#         denoised_com = torch.stack((denoised_mag*torch.cos(denoised_pha),
#                                     denoised_mag*torch.sin(denoised_pha)), dim=-1)
#
#         return denoised_mag, denoised_pha, denoised_com

class LKFCA_Net(nn.Module):
    def __init__(self, h, num_tsblock=4):
        super(LKFCA_Net, self).__init__()
        self.h = h
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.LKFCAnet = nn.ModuleList([])
        # self.InteractionModlue = nn.ModuleList([])
        # for i in range(4):
        #     self.InteractionModlue.append(Interaction(h,h.dense_channel*2))
        for i in range(4):
            self.LKFCAnet.append(TS_BLOCK(h))
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)
        self.TSFMamba = nn.ModuleList([TFFMambaBlock(h) for _ in range(1)])
        self.TSBMamba = nn.ModuleList([TFBMambaBlock(h) for _ in range(1)])

    def forward(self, noisy_mag, noisy_pha):  # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]
        x = self.dense_encoder(x)
        x, x_f1, x_f2 = self.TSFMamba[0](x)

        for i in range(4):
            x = self.LKFCAnet[i](x)
        x = self.TSBMamba[0](x,x_f1,x_f2)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = (self.phase_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
                                    denoised_mag * torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com
class DBD_LKFCA_Net(nn.Module):
    def __init__(self, h, num_tsblock=6):
        super(DBD_LKFCA_Net, self).__init__()
        self.h = h
        self.depth = h.depth
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.LKFCAnet = nn.ModuleList([])
        # self.InteractionModlue = nn.ModuleList([])
        # for i in range(4):
        #     self.InteractionModlue.append(Interaction(h,h.dense_channel*2))
        for i in range(self.depth):
            self.LKFCAnet.append(TS_BLOCK(h))
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        #self.mag_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)
        self.phamask_decoder = PMaskDecoder(h, out_channel=1)
        self.TSFMamba = nn.ModuleList([TFFMambaBlock(h) for _ in range(1)])
        self.TSBMamba = nn.ModuleList([TFBMambaBlock(h) for _ in range(1)])
        self.w_pha1 = nn.Parameter(torch.ones(1,  1, 1), requires_grad=True)
        self.w_pha = nn.Parameter(torch.zeros(1,  1, 1), requires_grad=True)
        #self.w_mag1 = nn.Parameter(torch.ones(1,  1, 1), requires_grad=True)
        #self.w_mag = nn.Parameter(torch.zeros(1,  1, 1), requires_grad=True)
    def forward(self, noisy_mag, noisy_pha):  # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]
        x = self.dense_encoder(x)
        x, x_f1, x_f2 = self.TSFMamba[0](x)

        for i in range(self.depth):
            x = self.LKFCAnet[i](x)
        x = self.TSBMamba[0](x,x_f1,x_f2)
        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        #denoised_mag = (noisy_mag * self.mask_decoder(x) * self.w_mag1 + self.mag_decoder(x)* self.w_mag).permute(0, 3, 2, 1).squeeze(-1)
        # denoised_pha = (self.phase_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha, mask_pha = self.phase_decoder(x),self.phamask_decoder(x)
        denoised_pha = (denoised_pha * self.w_pha1  + (mask_pha * torch.pi + noisy_pha) * self.w_pha).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
                                    denoised_mag * torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com
def phase_losses(phase_r, phase_g, h):
    dim_freq = h.n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq),
                                                                                     diagonal=2) - torch.eye(
        dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time),
                                                                                      diagonal=2) - torch.eye(
        dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):
    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        h.sampling_rate)
                                     for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score
