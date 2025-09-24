# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam2_utils import DropPath, get_clones, LayerNorm2d


class MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
        out_levels = 4,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        self.num_layers = num_layers
        self.out_levels = out_levels
        assert stride**num_layers == total_stride
        # self.encoder = nn.Sequential()
        self.encoder_blocks = nn.ModuleList()
        self.out_projs = nn.ModuleList()
        mask_in_chans, mask_out_chans = 1, 1
        for i in range(self.num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            encoder_block = nn.Sequential()
            encoder_block.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            encoder_block.append(LayerNorm2d(mask_out_chans))
            encoder_block.append(activation())
            self.encoder_blocks.append(encoder_block)
            if i >= self.num_layers-self.out_levels:
                self.out_projs.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
            mask_in_chans = mask_out_chans

    def forward(self, x):
        outs = [] 
        out_num = 0
        for i in range(self.num_layers):
            x = self.encoder_blocks[i](x)
            if i >= self.num_layers-self.out_levels:
                out = self.out_projs[out_num](x)
                out_num += 1
                outs.append(out)
        return outs


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        position_encoding,
        stages = 4,
        in_dim=256,  # in_dim of pix_feats

    ):
        super().__init__()
        # 1. mask encoder
        self.mask_downsampler = mask_downsampler
        self.position_encoding = position_encoding
        self.stages = stages
        self.out_dim = out_dim
        self.pix_feat_projs = nn.ModuleList()
        self.fusers = nn.ModuleList()
        self.out_projs = nn.ModuleList()
        for stage in range(stages): 
            # 2.1. projector for feature
            pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
            self.pix_feat_projs.append(pix_feat_proj)
            # 2.2. fuse feature and encoded mask
            fuser_layer = CXBlock(dim = 256, kernel_size=7, padding=3,
                                  layer_scale_init_value=1e-6, use_dwconv=True,)
            fuser = Fuser(layer = fuser_layer, num_layers=2)
            self.fusers.append(fuser)
            # 2.3. output projector 
            out_proj = nn.Identity()
            if out_dim != in_dim:
                out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
            self.out_projs.append(out_proj)
    def forward(
        self,
        stage_pix_feats: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        mask_levels = self.mask_downsampler(masks)

        assert len(mask_levels) == len(stage_pix_feats) == self.stages
        stage_outs = {"maskmem_features": [], "maskmem_pos_enc": [], "maskmem_feat_size": []}
        for i, (mask, pix_feat) in enumerate(zip(mask_levels, stage_pix_feats)):
            ## Fuse pix_feats and downsampled masks
            # in case the visual features are on CPU, cast them to CUDA
            pix_feat = pix_feat.to(mask.device)
            x = self.pix_feat_projs[i](pix_feat)
            x = x + mask
            x = self.fusers[i](x)
            x = self.out_projs[i](x)
            pos = self.position_encoding(x).to(x.dtype)

            stage_outs["maskmem_features"].append(x)
            stage_outs["maskmem_pos_enc"].append(pos)
            stage_outs["maskmem_feat_size"].append(torch.tensor([x.shape[-2], x.shape[-1]]))

        return stage_outs