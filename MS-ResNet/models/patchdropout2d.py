import math
import torch
import torch.nn as nn


class PostEncodingPatchDropout2D(nn.Module):
    """
    Post-encoding PatchDropout for CNN/SNN feature maps.

    - Input:  [T, B, C, H, W]  (SNN time stack)  OR  [B, C, H, W]
    - We drop *spatial patches* (blocks) on the encoded feature map.
    - A single mask is sampled per sample (B) and shared across time steps (T).
    - Uses "keep exactly K patches" (like token selection), but we mask-out instead of shortening the tensor.
    """

    def __init__(self, keep_rate: float = 1.0, patch_size: int = 4):
        super().__init__()
        assert 0.0 < keep_rate <= 1.0, "keep_rate must be in (0, 1]."
        assert patch_size >= 1, "patch_size must be >= 1."
        self.keep_rate = float(keep_rate)
        self.patch_size = int(patch_size)

    def forward(self, x):
        # Only active during training
        if (not self.training) or self.keep_rate >= 1.0:
            return x

        if x.dim() == 5:
            # [T, B, C, H, W]
            T, B, C, H, W = x.shape
        elif x.dim() == 4:
            # [B, C, H, W]
            B, C, H, W = x.shape
            T = None
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")

        ps = self.patch_size
        gh = int(math.ceil(H / ps))
        gw = int(math.ceil(W / ps))
        L = gh * gw

        keep = max(1, int(L * self.keep_rate))  # keep at least 1 patch

        # Sample exactly `keep` patch positions per sample
        scores = torch.rand(B, L, device=x.device)
        idx = torch.argsort(scores, dim=1)[:, :keep]  # [B, keep]

        mask = torch.zeros(B, L, device=x.device, dtype=x.dtype)
        mask.scatter_(1, idx, 1.0)
        mask = mask.view(B, 1, gh, gw)

        # Expand patch-grid mask to pixel-grid mask
        mask = mask.repeat_interleave(ps, dim=2).repeat_interleave(ps, dim=3)
        mask = mask[:, :, :H, :W]  # crop to exact H,W

        if x.dim() == 5:
            # broadcast over time and channels: [1, B, 1, H, W]
            mask = mask.unsqueeze(0)
        # Multiplicative masking
        return x * mask
