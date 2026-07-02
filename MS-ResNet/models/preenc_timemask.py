# models/preenc_timemask.py
import torch
import torch.nn as nn


class PreEncodingTimeMask(nn.Module):
    """
    Pre-encoding TimeMask for MS-ResNet.

    Expected input:
        - layout == "TBCHW": [T, B, C, H, W]
        - layout == "BTCHW": [B, T, C, H, W]

    This masks contiguous temporal intervals BEFORE the first convolution,
    i.e. after the static image has been expanded along time but before any
    spike generation / temporal feature extraction in the network.

    Notes:
      - training-only by default
      - supports same-on-batch and per-channel-group masking
      - mode="zero" or mode="noise"
    """
    def __init__(self,
                 p=0.0,
                 num_masks=1,
                 max_mask_frac=0.25,
                 min_mask_len=1,
                 mode="zero",
                 noise_std=0.05,
                 layout="TBCHW",
                 same_on_batch=False,
                 per_channel=False,
                 channel_groups=1,
                 apply_in_eval=False):
        super().__init__()
        self.p = float(p)
        self.num_masks = int(num_masks)
        self.max_mask_frac = float(max_mask_frac)
        self.min_mask_len = int(min_mask_len)
        self.mode = str(mode)
        self.noise_std = float(noise_std)
        self.layout = str(layout)
        self.same_on_batch = bool(same_on_batch)
        self.per_channel = bool(per_channel)
        self.channel_groups = max(1, int(channel_groups))
        self.apply_in_eval = bool(apply_in_eval)

        if self.mode not in ["zero", "noise"]:
            raise ValueError("mode must be 'zero' or 'noise'")

        if self.layout not in ["TBCHW", "BTCHW"]:
            raise ValueError("layout must be 'TBCHW' or 'BTCHW'")

    def extra_repr(self):
        return (
            f"p={self.p}, num_masks={self.num_masks}, max_mask_frac={self.max_mask_frac}, "
            f"min_mask_len={self.min_mask_len}, mode={self.mode}, noise_std={self.noise_std}, "
            f"layout={self.layout}, same_on_batch={self.same_on_batch}, "
            f"per_channel={self.per_channel}, channel_groups={self.channel_groups}, "
            f"apply_in_eval={self.apply_in_eval}"
        )

    def _to_tbchw(self, x):
        if self.layout == "TBCHW":
            return x, False
        return x.transpose(0, 1).contiguous(), True

    def _from_tbchw(self, x, was_btchw):
        if was_btchw:
            return x.transpose(0, 1).contiguous()
        return x

    def _group_slices(self, c):
        g = max(1, min(self.channel_groups, c))
        bounds = torch.linspace(0, c, steps=g + 1).long().tolist()
        slices = []
        for i in range(g):
            s, e = bounds[i], bounds[i + 1]
            if e > s:
                slices.append((s, e))
        if not slices:
            slices = [(0, c)]
        return slices

    def _sample_interval(self, T, device):
        max_len = max(self.min_mask_len, int(round(T * self.max_mask_frac)))
        max_len = min(max_len, T)
        min_len = min(self.min_mask_len, max_len)

        if max_len <= 0:
            return 0, 0

        if min_len == max_len:
            L = min_len
        else:
            L = int(torch.randint(min_len, max_len + 1, (1,), device=device).item())

        if L >= T:
            return 0, T

        start = int(torch.randint(0, T - L + 1, (1,), device=device).item())
        return start, L

    def _apply_mask_region(self, out, t0, t1, b_slice, c_slice):
        if self.mode == "zero":
            out[t0:t1, b_slice, c_slice, :, :] = 0
        else:
            region = out[t0:t1, b_slice, c_slice, :, :]
            noise = torch.randn_like(region) * self.noise_std
            out[t0:t1, b_slice, c_slice, :, :] = noise

    def forward(self, x):
        if x is None:
            return x

        if x.dim() != 5:
            raise ValueError(
                f"PreEncodingTimeMask expects 5D input, got shape {tuple(x.shape)}"
            )

        if self.p <= 0.0 or self.num_masks <= 0 or self.max_mask_frac <= 0.0:
            return x

        if (not self.training) and (not self.apply_in_eval):
            return x

        x_tb, was_btchw = self._to_tbchw(x)
        T, B, C, H, W = x_tb.shape

        if T <= 1:
            return x

        out = x_tb.clone()
        channel_slices = self._group_slices(C)

        if self.same_on_batch:
            if float(torch.rand(1, device=x.device).item()) >= self.p:
                return x

            if self.per_channel:
                for cs, ce in channel_slices:
                    for _ in range(self.num_masks):
                        t0, L = self._sample_interval(T, x.device)
                        if L > 0:
                            self._apply_mask_region(out, t0, t0 + L, slice(None), slice(cs, ce))
            else:
                for _ in range(self.num_masks):
                    t0, L = self._sample_interval(T, x.device)
                    if L > 0:
                        self._apply_mask_region(out, t0, t0 + L, slice(None), slice(None))
        else:
            for b in range(B):
                if float(torch.rand(1, device=x.device).item()) >= self.p:
                    continue

                if self.per_channel:
                    for cs, ce in channel_slices:
                        for _ in range(self.num_masks):
                            t0, L = self._sample_interval(T, x.device)
                            if L > 0:
                                self._apply_mask_region(out, t0, t0 + L, slice(b, b + 1), slice(cs, ce))
                else:
                    for _ in range(self.num_masks):
                        t0, L = self._sample_interval(T, x.device)
                        if L > 0:
                            self._apply_mask_region(out, t0, t0 + L, slice(b, b + 1), slice(None))

        return self._from_tbchw(out, was_btchw)