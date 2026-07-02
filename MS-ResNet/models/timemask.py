# models/timemask.py
from __future__ import annotations
import torch
import torch.nn as nn


class TimeMaskPostEncoding(nn.Module):
    """
    TimeMask for post-encoding spikes/features.

    Your MS-ResNet uses tensors shaped [T, B, C, H, W] throughout,
    so default layout is "TB".

    Masks contiguous time intervals on the time axis.
    """

    def __init__(
        self,
        p: float = 0.0,
        num_masks: int = 1,
        max_mask_frac: float = 0.25,
        min_mask_len: int = 1,
        mode: str = "zero",        # "zero" or "noise"
        noise_std: float = 0.05,
        layout: str = "TB",        # "TB" for MS-ResNet: [T,B,...]
        same_on_batch: bool = False,
        per_channel: bool = False,
        channel_groups: int = 1,
    ):
        super().__init__()
        assert 0.0 <= p <= 1.0
        assert num_masks >= 1
        assert 0.0 < max_mask_frac <= 1.0
        assert min_mask_len >= 1
        assert mode in ("zero", "noise")
        assert layout in ("BT", "TB")
        assert channel_groups >= 1

        self.p = float(p)
        self.num_masks = int(num_masks)
        self.max_mask_frac = float(max_mask_frac)
        self.min_mask_len = int(min_mask_len)
        self.mode = mode
        self.noise_std = float(noise_std)
        self.layout = layout
        self.same_on_batch = bool(same_on_batch)
        self.per_channel = bool(per_channel)
        self.channel_groups = int(channel_groups)

    def _to_bt(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.layout == "BT" else x.transpose(0, 1)

    def _from_bt(self, x_bt: torch.Tensor) -> torch.Tensor:
        return x_bt if self.layout == "BT" else x_bt.transpose(0, 1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p == 0.0:
            return x

        x_bt = self._to_bt(x)  # [B,T,...]
        if x_bt.dim() < 3:
            raise ValueError(f"Expected [T,B,...] or [B,T,...], got {tuple(x.shape)}")

        B, T = x_bt.shape[0], x_bt.shape[1]
        if T <= 1:
            return x

        # which samples to apply
        if self.same_on_batch:
            apply = (torch.rand((), device=x_bt.device) < self.p).expand(B)
        else:
            apply = (torch.rand((B,), device=x_bt.device) < self.p)

        if not apply.any():
            return x

        y = x_bt.clone()

        # channel dim at index 2 when present
        C = y.shape[2] if y.dim() >= 3 else 1

        if self.per_channel and C > 1:
            g = min(self.channel_groups, C)
            base = C // g
            rem = C % g
            group_slices = []
            st = 0
            for i in range(g):
                ln = base + (1 if i < rem else 0)
                group_slices.append((st, st + ln))
                st += ln
        else:
            group_slices = [(0, C)]

        for b in range(B):
            if not bool(apply[b].item()):
                continue

            for (c0, c1) in group_slices:
                time_mask = torch.zeros((T,), dtype=torch.bool, device=y.device)

                for _ in range(self.num_masks):
                    max_len = max(self.min_mask_len, int(round(self.max_mask_frac * T)))
                    max_len = min(max_len, T)

                    L = torch.randint(
                        low=self.min_mask_len,
                        high=max_len + 1,
                        size=(1,),
                        device=y.device
                    ).item()

                    t0 = 0 if (T - L) <= 0 else torch.randint(
                        low=0, high=T - L + 1, size=(1,), device=y.device
                    ).item()

                    time_mask[t0:t0 + L] = True

                if self.mode == "zero":
                    y[b, time_mask, c0:c1, ...] = 0
                else:
                    noise = torch.randn_like(y[b, time_mask, c0:c1, ...]) * self.noise_std
                    y[b, time_mask, c0:c1, ...] = noise

        return self._from_bt(y)
