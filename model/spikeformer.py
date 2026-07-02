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


class TemporalJitter(nn.Module):
    """
    Post-encoding temporal circular shift (wrap-around) on (T, B, C, H, W).

    Global mode: one shift k shared by the whole batch.
    Per-sample mode: each sample b gets its own shift k_b.
    Wrap-around keeps the total activity and the per-frame spike-rate
    distribution unchanged.
    """

    def __init__(self, max_shift: int = 1, p: float = 0.5, per_sample: bool = False):
        super().__init__()
        self.max_shift = int(max_shift)
        self.p = float(p)
        self.per_sample = bool(per_sample)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self.max_shift <= 0 or x.dim() != 5:
            return x

        if self.p < 1.0 and torch.rand((), device=x.device).item() > self.p:
            return x

        T, B, C, H, W = x.shape
        if T <= 1:
            return x

        if not self.per_sample:
            k = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,), device=x.device).item())
            if k == 0:
                return x
            return torch.roll(x, shifts=k, dims=0)

        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (B,), device=x.device)
        t = torch.arange(T, device=x.device).view(T, 1)
        idx = (t - shifts.view(1, B)) % T
        idx = idx.view(T, B, 1, 1, 1).expand(T, B, C, H, W)
        return torch.gather(x, dim=0, index=idx)


class CenterPatchMinLift(nn.Module):
    """
    Post-encoding CenterPatch MinLift augmentation on (T, B, C, H, W) spiking
    features. It is applied after patch embedding and before the transformer
    blocks, in the same place as TemporalJitter / LocalTimeShuffle.

    A centered spatial patch of the feature grid is lifted toward the maximum
    activation inside that patch:

        patch_aug = (1 - alpha) * patch + alpha * M

    where M is the max activation over the patch spatial extent. Low
    activations in the center region are pulled toward the strongest local
    response while the surrounding features are left untouched.

    share_time:
        True  -> one max and one alpha per sample, shared across all time
                 steps, so the same lift is applied to every frame and the
                 temporal spike structure stays coherent.
        False -> each time step is lifted independently, matching the
                 per-frame behavior of the original pre-encoding version.

    Shapes:
        input : (T, B, C, H, W)
        output: (T, B, C, H, W)

    Notes:
        This module runs under torch.no_grad() and returns a fresh tensor on
        the apply path, consistent with the other post-encoding augmentations
        in this file. On the skip path (probability 1 - p) it returns the
        input unchanged, so gradients still flow to patch_embed on those
        batches.
    """

    def __init__(
        self,
        patch_frac: float = 0.5,
        alpha_range=(0.3, 0.7),
        p: float = 0.5,
        share_time: bool = True,
    ):
        super().__init__()
        self.patch_frac = float(patch_frac)
        self.alpha_min = float(alpha_range[0])
        self.alpha_max = float(alpha_range[1])
        self.p = float(p)
        self.share_time = bool(share_time)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            return x

        if self.p < 1.0 and torch.rand((), device=x.device).item() > self.p:
            return x

        T, B, C, H, W = x.shape

        ph = max(1, int(round(H * self.patch_frac)))
        pw = max(1, int(round(W * self.patch_frac)))

        y0 = (H - ph) // 2
        x0 = (W - pw) // 2
        y1 = y0 + ph
        x1 = x0 + pw

        # Work on a fresh tensor so we never modify the patch_embed output
        # in place. Consistent with TemporalJitter / LocalTimeShuffle.
        out = x.clone()
        patch = out[:, :, :, y0:y1, x0:x1]  # (T, B, C, ph, pw)

        if self.share_time:
            # One max per (sample, channel), one alpha per sample, shared over
            # time so every frame gets the same lift.
            M = patch.amax(dim=(0, 3, 4), keepdim=True)          # (1, B, C, 1, 1)
            alpha = torch.empty(
                (1, B, 1, 1, 1), device=x.device, dtype=x.dtype
            ).uniform_(self.alpha_min, self.alpha_max)           # (1, B, 1, 1, 1)
        else:
            # Per (time, sample, channel) max, per (time, sample) alpha.
            M = patch.amax(dim=(3, 4), keepdim=True)             # (T, B, C, 1, 1)
            alpha = torch.empty(
                (T, B, 1, 1, 1), device=x.device, dtype=x.dtype
            ).uniform_(self.alpha_min, self.alpha_max)           # (T, B, 1, 1, 1)

        out[:, :, :, y0:y1, x0:x1] = (1.0 - alpha) * patch + alpha * M
        return out


class SpikeDrivenTransformer(nn.Module):
    """
    SpikeDrivenTransformer with a non-invasive PatchDropout hook.

    Option A goals:
    - Keep EXACT behavior / shapes so PatchMix, PatchShuffle, IPMix, LayerMix, etc. do not break.
    - Expose a standard place where PatchDropout *would* be applied (after patch_embed, before blocks).
    - Allow the training script to pass in a PatchDropoutTokens module later via `patchdrop_layer=...`,
      without forcing other code to change how it calls the model.

    We DO NOT actually modify the tensor `x` with PatchDropout here because the current backbone
    (MS_SPS + MS_Block_Conv) expects dense (T,B,C,H,W) feature maps. Real PatchDropout from the paper
    drops tokens and shortens the sequence before attention. That requires refactoring the blocks to
    operate on token lists instead of spatial grids. Here we only install a hook + debug print.
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


        # === NEW (Option A): optional PatchDropout module
        # This is expected to be something like PatchDropoutTokens from the paper.
        # Default None keeps behavior identical to the old code.
        patchdrop_layer=None,

        # === NEW: optional post-encoding CenterPatch MinLift module.
        # Default None keeps behavior identical to the old code. It can also be
        # attached from the training script as model.center_patch_min_lift = ...
        center_patch_min_lift=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode
        self.temporal_jitter = None
        self._tj_warned = False
        # Save (optional) PatchDropout layer.
        # We won't *use* it to change tensors yet, but we register it here so:
        # - training code can pass it in
        # - we have a stable attribute on the model
        self.patchdrop_layer = patchdrop_layer
        self._pd_warned = False  # to avoid spamming prints
        self.local_time_shuffle = None
        self._lts_warned = False
        # Post-encoding CenterPatch MinLift.
        self.center_patch_min_lift = center_patch_min_lift
        self._cpml_warned = False
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
        # NOTE: These blocks currently operate on dense feature maps,
        # shape ~ (T, B, C, H, W). They are NOT token-sequence transformer
        # blocks in (B, N, D) format.
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
            # Fallback just in case
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

    def forward_features(self, x, hook=None):
        """
        x is expected to be (T, B, C, H, W)

        Steps:
        1. patch_embed -> produce spiking patch features (T,B,C_embed,H_p,W_p)
        2. OPTIONAL post-encoding augmentations (LocalTimeShuffle, TemporalJitter,
           CenterPatchMinLift, PatchDropout)
        3. run MS_Block_Conv blocks
        4. spatial pool
        """

        # 1. Spiking patch embedding
        x, _, hook = self.patch_embed(x, hook=hook)
        # x: (T, B, C_embed, H_p, W_p)
        T, B, C, H, W = x.shape


        if (self.training and hasattr(self, "local_time_shuffle") and self.local_time_shuffle is not None):
            x = self.local_time_shuffle(x)

            if not self._lts_warned:
                print(
                    "[LocalTimeShuffle] Applied post-encoding LocalTimeShuffle on "
                    f"features: T={T}, B={B}, C={C}, H={H}, W={W}",
                    flush=True,
                )
                self._lts_warned = True
        if (self.training and getattr(self, "temporal_jitter", None) is not None):
            x = self.temporal_jitter(x)

            if not self._tj_warned:
                print(
                    "[TemporalJitter] Applied post-encoding TemporalJitter on "
                    f"features: T={T}, B={B}, C={C}, H={H}, W={W}",
                    flush=True,
                )
                self._tj_warned = True

        # Post-encoding CenterPatch MinLift on the spiking feature grid.
        if (self.training and getattr(self, "center_patch_min_lift", None) is not None):
            x = self.center_patch_min_lift(x)

            if not self._cpml_warned:
                print(
                    "[CenterPatchMinLift] Applied post-encoding CenterPatchMinLift on "
                    f"features: T={T}, B={B}, C={C}, H={H}, W={W}, "
                    f"patch_frac={self.center_patch_min_lift.patch_frac}, "
                    f"alpha=({self.center_patch_min_lift.alpha_min}, "
                    f"{self.center_patch_min_lift.alpha_max}), "
                    f"p={self.center_patch_min_lift.p}, "
                    f"share_time={self.center_patch_min_lift.share_time}",
                    flush=True,
                )
                self._cpml_warned = True

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
            # If someone passes a sequence-shortening PatchDropout (mask_output=False),
            # we skip it to avoid breaking the (T,B,C,H,W) grid structure.
            print(
                "[PatchDropout] patchdrop_layer is set but mask_output=False, "
                "so it is NOT applied inside SpikeDrivenTransformer to preserve grid shape. "
                "Use --pd-mask-output for post-encoding PatchDropout on SDT.",
                flush=True,
            )
            self._pd_warned = True

        # 3. Backbone blocks (unchanged)
        for blk in self.block:
            x, _, hook = blk(x, hook=hook)
            # x stays (T, B, C_embed, H_p, W_p)

        # 4. Global spatial pooling over H_p, W_p
        # x: (T, B, C_embed, H_p, W_p)
        x = x.flatten(3).mean(3)  # -> (T, B, C_embed)

        return x, hook

    def forward(self, x, hook=None):
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
        """

        # Normalize input shape to (T, B, C, H, W)
        if len(x.shape) < 5:
            # Input was (B, C, H, W). Repeat across time steps.
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)  # -> (T, B, C, H, W)
        else:
            # Input was (B, T, C, H, W). Move T to dim 0.
            x = x.transpose(0, 1).contiguous()              # -> (T, B, C, H, W)

        # Backbone forward
        x, hook = self.forward_features(x, hook=hook)
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

    This also means: code that does sdt(patchdrop_layer=...) will now attach the
    optional PatchDropout hook, while old code that doesn't pass it continues
    to behave identically. The same applies to center_patch_min_lift=...
    """
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model