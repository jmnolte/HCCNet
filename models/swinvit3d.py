from collections.abc import Sequence
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from monai.networks.blocks import PatchEmbed, UnetrBasicBlock
from monai.networks.nets.swin_unetr import BasicLayer, PatchMerging, PatchMergingV2
from monai.utils import look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        num_classes: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.use_v2 = use_v2
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        if self.use_v2:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
            if self.use_v2:
                layerc = UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                if i_layer == 0:
                    self.layers1c.append(layerc)
                elif i_layer == 1:
                    self.layers2c.append(layerc)
                elif i_layer == 2:
                    self.layers3c.append(layerc)
                elif i_layer == 3:
                    self.layers4c.append(layerc)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward_features(self, x, normalize=True):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        if self.use_v2:
            x = self.layers1c[0](x.contiguous())
            x = self.layers2c[0](x.contiguous())
            x = self.layers3c[0](x.contiguous())
            x = self.layers4c[0](x.contiguous())
        else:
            x = self.layers1[0](x.contiguous())
            x = self.layers2[0](x.contiguous())
            x = self.layers3[0](x.contiguous())
            x = self.layers4[0](x.contiguous())
        b, c, d, h, w = x.size()
        x = x.view(b, d * h * w, c)
        x = self.norm(x) # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1) # B C
        return x

    def forward(self, x):
        x = self.forward_features(x)
        out = self.head(x)
        return out

def _swinvit(
        progress: bool = False,
        **kwargs: Any
    ) -> SwinTransformer:
        
    model = SwinTransformer(
        window_size=(7, 7, 7),
        patch_size=(4, 4, 4),
        num_heads=(3, 6, 12, 24),
        mlp_ratio=4,
        drop_path_rate=0.1,
        patch_norm=True,
        **kwargs)
    return model

def swinvit_tiny(progress: bool = False, **kwargs: Any) -> SwinTransformer:
    return _swinvit(embed_dim=48, depths=(2, 2, 6, 2), pretrained=pretrained, progress=progress, **kwargs)

def swinvit_small(progress: bool = False, **kwargs: Any) -> SwinTransformer:
    return _swinvit(embed_dim=48, depths=(2, 2, 18, 2), pretrained=pretrained, progress=progress, **kwargs)

def swinvit_base(progress: bool = False, **kwargs: Any) -> SwinTransformer:
    return _swinvit(embed_dim=96, depths=(2, 2, 6, 2), pretrained=pretrained, progress=progress, **kwargs)

def swinvit_large(progress: bool = False, **kwargs: Any) -> SwinTransformer:
    return _swinvit(embed_dim=96, depths=(2, 2, 18, 2), pretrained=pretrained, progress=progress, **kwargs)