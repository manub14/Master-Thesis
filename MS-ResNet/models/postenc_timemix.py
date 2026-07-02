# models/postenc_timemix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TimeMixConfig:
    p: float = 0.0                  # apply probability (0 disables)
    ck: int = 32                    # number of channel groups
    alpha: float = 0.5              # residual penalty factor
    split: str = "random"           # "random" | "fixed"
    fixed_g1: float = 0.25          # fraction of groups (if fixed)
    fixed_g2: float = 0.50          # fraction of groups (if fixed)
    learnable_alpha: bool = False
    layout: str = "TBCHW"           # "TBCHW" or "BTCHW"
    apply_in_eval: bool = False     # if True, also apply in eval


class PostEncTimeMix(nn.Module):
    """
    Post-encoding TimeMix (temporal shift + residual mix) for spike tensors.

    Expected spike tensor:
      - TBCHW: [T,B,C,H,W] (MS-ResNet default)
      - BTCHW: [B,T,C,H,W] (optional)

    Operation:
      Z = TemporalShift(X) using channel-group splits (g1,g2)
      Y = X + alpha * Z
    """
    def __init__(self, cfg: TimeMixConfig = TimeMixConfig()):
        super().__init__()
        assert cfg.layout in ("TBCHW", "BTCHW")
        assert cfg.split in ("random", "fixed")
        assert 0.0 <= cfg.p <= 1.0

        self.cfg = cfg
        if cfg.learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(cfg.alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(cfg.alpha)), persistent=False)

    def _pick_splits(self, ck: int, device: torch.device) -> tuple[int, int]:
        """
        Return (g1, g2) in group units.

        We allow:
          - ck == 2 -> (1, 2)  (so group2 reaches end; no "no-shift" group)
        """
        if ck <= 1:
            return 0, 0
        if ck == 2:
            return 1, 2

        if self.cfg.split == "fixed":
            g1 = int(round(self.cfg.fixed_g1 * ck))
            g2 = int(round(self.cfg.fixed_g2 * ck))
            g1 = max(1, min(g1, ck - 2))
            g2 = max(g1 + 1, min(g2, ck))   # allow g2 == ck
            return g1, g2

        # random
        g1 = int(torch.randint(1, ck - 1, (1,), device=device).item())      # [1, ck-2]
        g2 = int(torch.randint(g1 + 1, ck + 1, (1,), device=device).item()) # [g1+1, ck]
        return g1, g2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return x

        # only apply in training unless apply_in_eval
        if (not self.training) and (not self.cfg.apply_in_eval):
            return x

        if self.cfg.p == 0.0:
            return x

        if x.dim() != 5:
            # if someone passes BCHW, do nothing
            return x

        # probability gate (per forward call)
        if self.training and self.cfg.p < 1.0:
            if torch.rand((), device=x.device) >= self.cfg.p:
                return x

        # normalize to TBCHW internally
        orig_layout = self.cfg.layout
        if orig_layout == "BTCHW":
            x_tb = x.permute(1, 0, 2, 3, 4).contiguous()  # [T,B,C,H,W]
        else:
            x_tb = x

        T, B, C, H, W = x_tb.shape
        if T <= 1 or C <= 1:
            return x

        # channel grouping (exact ck groups; leftover tail is unshifted)
        ck = int(self.cfg.ck)
        ck = max(1, min(ck, C))

        if ck <= 1:
            return x  # nothing meaningful to split/shift

        cfold = C // ck
        if cfold == 0:
            ck = C
            cfold = 1

        grouped_C = ck * cfold  # <= C
        tail_start = grouped_C  # [tail_start:C] unchanged

        g1, g2 = self._pick_splits(ck, x.device)
        if g1 == 0 and g2 == 0:
            return x

        ch1 = g1 * cfold
        ch2 = min(g2 * cfold, grouped_C)

        z = x_tb.new_zeros(x_tb.shape)

        # left shift: group 1
        if ch1 > 0:
            z[:-1, :, :ch1] = x_tb[1:, :, :ch1]

        # right shift: group 2
        if ch2 > ch1:
            z[1:, :, ch1:ch2] = x_tb[:-1, :, ch1:ch2]

        # no shift: remaining grouped channels (if any)
        if ch2 < grouped_C:
            z[:, :, ch2:grouped_C] = x_tb[:, :, ch2:grouped_C]

        # tail (if any)
        if tail_start < C:
            z[:, :, tail_start:] = x_tb[:, :, tail_start:]

        alpha = self.alpha.to(dtype=x_tb.dtype)
        y_tb = x_tb + alpha * z

        # return in original layout
        if orig_layout == "BTCHW":
            return y_tb.permute(1, 0, 2, 3, 4).contiguous()
        return y_tb
