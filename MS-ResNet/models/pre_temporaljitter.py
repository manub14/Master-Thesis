import torch
import torch.nn as nn


class PreTimeShuffle(nn.Module):
    """
    Pre-encoding temporal jitter / temporal circular shift.

    This is the MS-ResNet counterpart of the SDT temporal jitter:
      - sample k in [-max_shift, +max_shift]
      - circularly roll along time dimension
      - optionally use a different shift for each sample

    Expected input layout:
      - TBCHW  (default)
      - BTCHW  (supported via layout argument)

    Notes:
      - This is NOT the same as local/windowed time shuffling.
      - This module is meant to be attached as self.preenc_aug and applied
        before the first spiking convolution / encoding stage.
    """

    def __init__(self, p=0.5, max_shift=1, per_sample=False, layout="TBCHW"):
        super(PreTimeShuffle, self).__init__()
        self.p = float(p)
        self.max_shift = int(max_shift)
        self.per_sample = bool(per_sample)
        self.layout = str(layout).upper()

        if self.layout not in ["TBCHW", "BTCHW"]:
            raise ValueError("layout must be 'TBCHW' or 'BTCHW'")

    def extra_repr(self):
        return "p={}, max_shift={}, per_sample={}, layout={}".format(
            self.p, self.max_shift, self.per_sample, self.layout
        )

    def _to_tbchw(self, x):
        if self.layout == "TBCHW":
            return x, False
        # BTCHW -> TBCHW
        return x.permute(1, 0, 2, 3, 4).contiguous(), True

    def _from_tbchw(self, x, was_btchw):
        if was_btchw:
            return x.permute(1, 0, 2, 3, 4).contiguous()
        return x

    def _global_shift(self, x):
        # x: [T, B, C, H, W]
        k = int(torch.randint(
            low=-self.max_shift,
            high=self.max_shift + 1,
            size=(1,),
            device=x.device
        ).item())

        if k == 0:
            return x

        return torch.roll(x, shifts=k, dims=0)

    def _per_sample_shift(self, x):
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape

        shifts = torch.randint(
            low=-self.max_shift,
            high=self.max_shift + 1,
            size=(B,),
            device=x.device
        )

        if (shifts == 0).all():
            return x

        t = torch.arange(T, device=x.device).view(T, 1)          # [T, 1]
        idx = (t - shifts.view(1, B)) % T                        # [T, B]
        idx = idx.view(T, B, 1, 1, 1).expand(T, B, C, H, W)     # [T, B, C, H, W]

        return torch.gather(x, dim=0, index=idx)

    def forward(self, x):
        if (not self.training) or self.p <= 0.0 or self.max_shift <= 0:
            return x

        if x.dim() != 5:
            return x

        if torch.rand(1, device=x.device).item() > self.p:
            return x

        x, was_btchw = self._to_tbchw(x)

        if x.size(0) <= 1:
            return self._from_tbchw(x, was_btchw)

        if self.per_sample:
            x = self._per_sample_shift(x)
        else:
            x = self._global_shift(x)

        x = self._from_tbchw(x, was_btchw)
        return x


# Alias, so your naming stays flexible in later code
PreTemporalJitter = PreTimeShuffle