# spiketransformer.py
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module import *
from module.time_aug import TimeShuffle, TimeMask, TimeMix


class SpikeDrivenTransformer(nn.Module):
    """
    Spike-Driven Transformer with post-encoding temporal augmentations:
    - TimeShuffle: permute T steps
    - TimeMask:    zero-out a fraction of T steps
    - TimeMix:     TS-SNN-style temporal shift + residual mix

    Augmentations are applied immediately AFTER the spiking patch stem (MS_SPS)
    outputs spike frames shaped [T, B, C, H, W]. They are active only in training
    (unless TimeMix is configured to apply in eval too).
    """
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        pooling_stat="1111",
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
        # --- TimeShuffle ---
        time_shuffle_prob: float = 0.0,
        time_shuffle_per_channel: bool = False,
        time_shuffle_same_perm: bool = False,
        # --- TimeMask ---
        time_mask_prob: float = 0.0,
        time_mask_ratio: float = 0.25,
        time_mask_per_channel: bool = False,
        time_mask_same_mask: bool = False,
        # --- TimeMix (TS-SNN style) ---
        time_mix_prob: float = 0.0,
        time_mix_alpha: float = 0.5,
        time_mix_groups: int = 32,
        time_mix_random_split: bool = True,
        time_mix_apply_in_eval: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        # stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # Spiking Patch Stem (encoder to spikes)
        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        # Backbone blocks
        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                )
                for j in range(depths)
            ]
        )

        setattr(self, "patch_embed", patch_embed)
        setattr(self, "block", blocks)

        # Classification head (multi-step spiking + linear)
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        # --- Post-encoding temporal augmentations ---
        self.time_shuffle = TimeShuffle(
            prob=time_shuffle_prob,
            same_perm_across_batch=time_shuffle_same_perm,
            per_channel=time_shuffle_per_channel,
        )
        self.time_mask = TimeMask(
            prob=time_mask_prob,
            ratio=time_mask_ratio,
            same_mask_across_batch=time_mask_same_mask,
            per_channel=time_mask_per_channel,
            mask_value=0.0,
        )
        self.time_mix = TimeMix(
            prob=time_mix_prob,
            alpha=time_mix_alpha,
            groups=time_mix_groups,
            random_split=time_mix_random_split,
            apply_in_eval=time_mix_apply_in_eval,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        block = getattr(self, "block")
        patch_embed = getattr(self, "patch_embed")

        # x becomes spike frames: [T, B, C, H, W]
        x, _, hook = patch_embed(x, hook=hook)

        # === POST-ENCODING AUGMENTATIONS (order matches your figure) ===
        x = self.time_shuffle(x)   # Shuffle branch
        x = self.time_mask(x)      # Masking branch
        x = self.time_mix(x)       # Mixing (TS-SNN residual shift)
        # ===============================================================

        # Backbone
        for blk in block:
            x, _, hook = blk(x, hook=hook)

        # Global average over spatial dims
        x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None):
        # Ensure [T, B, C, H, W]
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x, hook = self.forward_features(x, hook=hook)

        # Temporal spiking head and classifier
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if not self.TET:
            x = x.mean(0)  # average over T
        return x, hook


@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(**kwargs)
    model.default_cfg = _cfg()
    return model
