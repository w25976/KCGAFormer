import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
from mmseg.models.builder import BACKBONES
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init, trunc_normal_init)

class DWKonvLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0,
                 **norm_kwargs):
        super(DWKonvLayer, self).__init__()

        self.inputdim = input_dim
        self.outdim = output_dim
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        # 使用深度可分离卷积替代传统卷积
        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=input_dim // groups,  # 深度可分离卷积
                                                   bias=False) for _ in range(groups)])

        # 使用深度可分离卷积
        self.spline_conv = nn.ModuleList([conv_class((grid_size + spline_order) * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=input_dim // groups,  # 深度可分离卷积
                                                     bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_dwkonv(self, x, group_index):
        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))

        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        target = x.shape[1:] + self.grid.shape
        grid = self.grid.view(*list([1 for _ in range(self.ndim + 1)] + [-1, ])).expand(target).contiguous().to(
            x.device)

        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., :-(k + 1)]
            right_intervals = grid[..., k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                    ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
        bases = bases.contiguous()
        bases = bases.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](bases)
        x = self.prelus[group_index](self.layer_norm[group_index](base_output + spline_output))

        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_dwkonv(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class DWKonv2DLayer(DWKonvLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm2d,
                 **norm_kwargs):
        super(DWKonv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                            input_dim, output_dim,
                                            spline_order, kernel_size,
                                            groups=groups, padding=padding, stride=stride, dilation=dilation,
                                            ndim=2,
                                            grid_size=grid_size, base_activation=base_activation,
                                            grid_range=grid_range, dropout=dropout, **norm_kwargs)


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dkonv = DWKonv2DLayer(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)  # Replaced DWConv with DWKonv2DLayer
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dkonv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class Conv2Attention(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DWKonv2DLayer(dim)  # Replaced nn.Conv2d with DWKonv2DLayer
        self.conv0_1 = DWKonv2DLayer(dim, dim, kernel_size=(1, 11), padding=(0, 3), groups=dim)  # DWKonv2DLayer instead of nn.Conv2d
        self.conv0_2 = DWKonv2DLayer(dim, dim, kernel_size=(11, 1), padding=(3, 0), groups=dim)  # DWKonv2DLayer instead of nn.Conv2d

        self.conv1_1 = DWKonv2DLayer(dim, dim, kernel_size=(1, 13), padding=(0, 5), groups=dim)  # DWKonv2DLayer instead of nn.Conv2d
        self.conv1_2 = DWKonv2DLayer(dim, dim, kernel_size=(13, 1), padding=(5, 0), groups=dim)  # DWKonv2DLayer instead of nn.Conv2d

        self.conv2_1 = DWKonv2DLayer(dim, dim, kernel_size=(1, 21), padding=(0, 10), groups=dim)  # DWKonv2DLayer instead of nn.Conv2d
        self.conv2_2 = DWKonv2DLayer(dim, dim, kernel_size=(21, 1), padding=(10, 0), groups=dim)  # DWKonv2DLayer instead of nn.Conv2d

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class KonvAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_conv_atten = Conv2Attention(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_conv_atten(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(BaseModule):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = KonvAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W

@BACKBONES.register_module()
class KCGAFormer(BaseModule):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super(KCGAFormer, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 1 else 3,
                                                stride=4, in_chans=embed_dims[i - 1],
                                                embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([
                Block(dim=embed_dims[i],
                      mlp_ratio=mlp_ratios[i],
                      drop=drop_rate,
                      drop_path=dpr[cur + j],
                      norm_cfg=norm_cfg)
                for j in range(depths[i])
            ])
            cur += depths[i]
            setattr(self, f'patch_embed{i}', patch_embed)
            setattr(self, f'block{i}', block)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.02)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, 1)
            elif isinstance(m, nn.ModuleList):
                for module in m:
                    module._init_weights()

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i}')
            block = getattr(self, f'block{i}')

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            outs.append(x)
        return outs

