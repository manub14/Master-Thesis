# module/time_aug.py
import math
import torch
import torch.nn as nn

class TimeShuffle(nn.Module):
    """Shuffle along T for [T,B,C,H,W]. Train-only unless prob=0."""
    def __init__(self, prob: float = 0.0,
                 same_perm_across_batch: bool = False,
                 per_channel: bool = False):
        super().__init__()
        self.prob = float(prob)
        self.same_perm_across_batch = same_perm_across_batch
        self.per_channel = per_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.prob <= 0.0 or x.dim() != 5:
            return x
        if torch.rand(1, device=x.device).item() > self.prob:
            return x

        T, B, C, H, W = x.shape
        if self.per_channel:
            xc = x.permute(1, 2, 0, 3, 4).contiguous()  # [B,C,T,H,W]
            perms = torch.stack([torch.randperm(T, device=x.device)
                                 for _ in range(B * C)], dim=0)  # [B*C,T]
            xc = xc.reshape(B * C, T, H, W)
            idx = perms.view(B * C, T, 1, 1).expand(-1, -1, H, W)
            xc = torch.gather(xc, 1, idx)
            x = xc.view(B, C, T, H, W).permute(2, 0, 1, 3, 4).contiguous()
        else:
            if self.same_perm_across_batch:
                perm = torch.randperm(T, device=x.device)
                x = x[perm, ...]
            else:
                xb = x.permute(1, 0, 2, 3, 4).contiguous()  # [B,T,C,H,W]
                perms = torch.stack([torch.randperm(T, device=x.device)
                                     for _ in range(B)], dim=0)           # [B,T]
                idx = perms.view(B, T, 1, 1, 1).expand(B, T, C, H, W)
                xb = torch.gather(xb, 1, idx)
                x = xb.permute(1, 0, 2, 3, 4).contiguous()
        return x


class TimeMask(nn.Module):
    """
    Mask (zero-out) a fraction of timesteps along T for [T,B,C,H,W].
    By default uses per-sample random time indices. Train-only.
    """
    def __init__(self, prob: float = 0.0, ratio: float = 0.25,
                 same_mask_across_batch: bool = False,
                 per_channel: bool = False,
                 mask_value: float = 0.0):
        super().__init__()
        self.prob = float(prob)
        self.ratio = float(ratio)
        self.same_mask_across_batch = same_mask_across_batch
        self.per_channel = per_channel
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.prob <= 0.0 or x.dim() != 5:
            return x
        if torch.rand(1, device=x.device).item() > self.prob:
            return x

        T, B, C, H, W = x.shape
        k = max(1, int(math.ceil(self.ratio * T)))

        if self.per_channel:
            # independent mask per (b,c)
            mask = torch.ones(T, B, C, 1, 1, device=x.device, dtype=x.dtype)
            for bc in range(B * C):
                b = bc // C
                c = bc % C
                idx = torch.randperm(T, device=x.device)[:k]
                mask[idx, b, c, 0, 0] = 0
            x = x * mask + (1 - mask) * self.mask_value
        else:
            if self.same_mask_across_batch:
                idx = torch.randperm(T, device=x.device)[:k]
                x[idx, ...] = self.mask_value
            else:
                # per-sample
                for b in range(B):
                    idx = torch.randperm(T, device=x.device)[:k]
                    x[idx, b, ...] = self.mask_value
        return x


class TimeMix(nn.Module):
    """
    TS-SNN-style temporal shift + residual mix for [T,B,C,H,W].
    Split channels into groups: first part shifts +1 (left), middle shifts -1 (right),
    rest unchanged. Then Z' = alpha * Z + X. Train-only by default.
    See TS-SNN residual formulation (Eq.7).  :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, prob: float = 0.0, alpha: float = 0.5,
                 groups: int = 32,        # Ck in TS-SNN
                 random_split: bool = True,
                 apply_in_eval: bool = False):
        super().__init__()
        self.prob = float(prob)
        self.alpha = float(alpha)
        self.groups = int(max(1, groups))
        self.random_split = bool(random_split)
        self.apply_in_eval = bool(apply_in_eval)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = self.training or self.apply_in_eval
        if (not cond) or self.prob <= 0.0 or x.dim() != 5:
            return x
        if torch.rand(1, device=x.device).item() > self.prob:
            return x

        T, B, C, H, W = x.shape
        g = min(self.groups, C)
        cfold = max(1, C // g)

        if self.random_split and g >= 3:
            g1 = int(torch.randint(1, g - 1, (1,), device=x.device).item())
            g2 = int(torch.randint(g1 + 1, g, (1,), device=x.device).item())
        else:
            g1 = g // 3
            g2 = (2 * g) // 3
            g1 = max(1, min(g1, g - 2))
            g2 = max(g1 + 1, min(g2, g - 1))

        c1 = g1 * cfold
        c2 = g2 * cfold

        Z = torch.zeros_like(x)

        # +1 shift (bring future to present): Z[t, ...] = X[t+1, ...]
        if c1 > 0:
            Z[:-1, :, :c1, ...] = x[1:, :, :c1, ...]
        # -1 shift (bring past to present): Z[t, ...] = X[t-1, ...]
        if c2 > c1:
            Z[1:, :, c1:c2, ...] = x[:-1, :, c1:c2, ...]
        # 0 shift (keep)
        if C > c2:
            Z[:, :, c2:, ...] = x[:, :, c2:, ...]

        # Residual mixing with penalty factor alpha (TS-SNN Eq.7).  :contentReference[oaicite:2]{index=2}
        return self.alpha * Z + x
