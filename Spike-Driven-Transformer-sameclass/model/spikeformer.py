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


class SpikeDrivenTransformer(nn.Module):
    """
    SpikeDrivenTransformer with a non-invasive PatchDropout hook + optional
    label-driven post-encoding augmentation.

    Goals:
    - Keep EXACT behavior / shapes so PatchMix, PatchShuffle, IPMix, LayerMix, etc. do not break.
    - Expose a standard place where PatchDropout *would* be applied (after patch_embed, before blocks).
    - Expose a standard place where post-encoding augmentations that need labels (y) can be applied.
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

        # === Optional PatchDropout module (token-level)
        patchdrop_layer=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        # Optional PatchDropout layer (post-encoding tokens)
        self.patchdrop_layer = patchdrop_layer
        self._pd_warned = False  # to avoid spamming prints

        # NEW: optional post-encoding augmentation that uses labels
        # Expected signature: aug(x, y) with
        #   x: [T, B, C, H, W], y: [B]
        # Default None = no post-encoding augmentation.
        self.postenc_aug = None

        # stochastic depth decay rule across layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # Patch embedding module (spiking patch sampler / stem)
        self.patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        # Backbone "transformer-like" spiking blocks
        # These blocks operate on dense feature maps (T, B, C, H, W)
        self.block = nn.ModuleList([
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
        ])

        # Classification head (temporal spiking + linear head)
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        else:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, y=None, hook=None):
        """
        x is expected to be (T, B, C, H, W)

        Steps:
        1. patch_embed -> spiking patch features (T, B, C_embed, H_p, W_p)
        2. OPTIONAL post-encoding PatchDropout (token-level, zeroing dropped patches)
        3. OPTIONAL label-driven post-encoding augmentation (e.g. ClassMixPostEncoder)
        4. run MS_Block_Conv blocks
        5. spatial pool
        """

        # 1. Spiking patch embedding
        x, _, hook = self.patch_embed(x, hook=hook)
        # x: (T, B, C_embed, H_p, W_p)
        T, B, C, H, W = x.shape

        # 2. Post-encoding PatchDropoutTokens (only if mask_output=True)
        if (
            self.patchdrop_layer is not None
            and self.training
            and getattr(self.patchdrop_layer, "mask_output", False)
        ):
            # Flatten spatial grid -> tokens: (B, T, N, C)
            N = H * W
            x_tokens = (
                x.permute(1, 0, 3, 4, 2)   # (B, T, H, W, C)
                  .reshape(B, T, N, C)      # (B, T, N, C)
            )

            # Token-level PatchDropout (zeros dropped tokens, keeps length N)
            x_tokens = self.patchdrop_layer(x_tokens)  # (B, T, N, C)

            # Tokens back -> grid: (T, B, C, H, W)
            x = (
                x_tokens.reshape(B, T, H, W, C)   # (B, T, H, W, C)
                        .permute(1, 0, 4, 2, 3)   # (T, B, C, H, W)
                        .contiguous()
            )

            if not self._pd_warned:
                print(
                    "[PatchDropout] Applied post-encoding PatchDropoutTokens on "
                    f"features: T={T}, B={B}, H={H}, W={W}, N={N}",
                    flush=True,
                )
                self._pd_warned = True

        elif self.patchdrop_layer is not None and self.training and not self._pd_warned:
            # If a sequence-shortening PatchDropout is passed (mask_output=False),
            # we skip it to preserve the (T,B,C,H,W) grid structure.
            print(
                "[PatchDropout] patchdrop_layer is set but mask_output=False, "
                "so it is NOT applied inside SpikeDrivenTransformer to preserve grid shape. "
                "Use --pd-mask-output for post-encoding PatchDropout on SDT.",
                flush=True,
            )
            self._pd_warned = True

        # 2b. NEW: label-driven post-encoding augmentation (your idea)
        # x: (T, B, C, H, W), y: (B,) with class indices.
        if self.postenc_aug is not None and y is not None and self.training:
            x = self.postenc_aug(x, y)

        # 3. Backbone blocks (unchanged)
        for blk in self.block:
            x, _, hook = blk(x, hook=hook)
            # x stays (T, B, C_embed, H_p, W_p)

        # 4. Global spatial pooling over H_p, W_p
        x = x.flatten(3).mean(3)  # -> (T, B, C_embed)

        return x, hook

    def forward(self, x, y=None, hook=None):
        """
        Accepts either:
        - (B, C, H, W)
        - (B, T, C, H, W)

        Internally we always work as (T, B, C, H, W).
        Then:
          forward_features -> (T, B, C_embed)
          head_lif        -> (T, B, C_embed)
          head (linear)   -> (T, B, num_classes)
          temporal avg    -> (B, num_classes) if not TET

        y:
          Optional labels tensor [B] used only by post-encoding augmentation.
          If y is None, no label-driven augmentation is applied.
        """

        # Normalize input shape to (T, B, C, H, W)
        if len(x.shape) < 5:
            # Input was (B, C, H, W). Repeat across time steps.
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)  # -> (T, B, C, H, W)
        else:
            # Input was (B, T, C, H, W). Move T to dim 0.
            x = x.transpose(0, 1).contiguous()              # -> (T, B, C, H, W)

        # Backbone forward (can use labels y for post-encoding aug)
        x, hook = self.forward_features(x, y=y, hook=hook)
        # x: (T, B, C_embed)

        # Spiking multi-step head
        x = self.head_lif(x)  # still (T, B, C_embed)
        if hook is not None:
            hook["head_lif"] = x.detach()

        # Final linear head
        x = self.head(x)      # (T, B, num_classes)

        # Average across time unless TET wants per-step
        if not self.TET:
            x = x.mean(0)     # (B, num_classes)

        return x, hook


@register_model
def sdt(**kwargs):
    """
    Factory function. We keep the same name `sdt` so existing code keeps working.

    Code that does sdt(patchdrop_layer=...) will attach the optional PatchDropout hook.
    You can also attach a label-driven post-encoding augmentation later via:

        model.postenc_aug = YourAugModule(...)
    """
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
