# models/postenc_patchmix.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PostEncPatchMix(nn.Module):
    """
    Post-encoding PatchMix for spike tensors.

    Supports inputs:
      - (B, T, C, H, W)  [default]
      - (T, B, C, H, W)  [set layout="TBCHW"]
      - (B, C, H, W)     [set layout="BCHW"]

    By default, it applies ONE spatial PatchMix transform consistently across ALL timesteps
    (i.e., time is folded into channels) which matches the "post-encoding patch-wise" spirit.
    """
    def __init__(
        self,
        patch_size: int = 16,
        p: float = 0.5,
        alpha: float = 0.2,
        beta: float = 0.2,
        layout: str = "BTCHW",          # "BTCHW", "TBCHW", "BCHW"
        mix_across_time: bool = True,   # fold T into channels => same patch mixing for all timesteps
        pad_to_multiple: bool = True,   # pad H/W so patching works cleanly
    ):
        super().__init__()
        assert patch_size > 0
        assert 0.0 <= p <= 1.0
        assert alpha > 0.0 and beta > 0.0
        assert layout in ("BTCHW", "TBCHW", "BCHW")

        self.patch_size = patch_size
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.layout = layout
        self.mix_across_time = mix_across_time
        self.pad_to_multiple = pad_to_multiple


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only augment during training
        if not self.training or self.p == 0.0:
            return x

        if x.dim() == 5:
            if self.layout == "TBCHW":
                x = x.permute(1, 0, 2, 3, 4).contiguous()  # -> BTCHW
            elif self.layout != "BTCHW":
                raise ValueError(f"layout={self.layout} incompatible with 5D input")

            b, t, c, h, w = x.shape
            if self.mix_across_time:
                x2 = x.view(b, t * c, h, w)
                x2 = self._patchmix_4d(x2)
                x = x2.view(b, t, c, h, w)
            else:
                # PatchMix each timestep independently (still within-image, not cross-batch)
                x2 = x.view(b * t, c, h, w)
                x2 = self._patchmix_4d(x2)
                x = x2.view(b, t, c, h, w)

            if self.layout == "TBCHW":
                x = x.permute(1, 0, 2, 3, 4).contiguous()  # back to TBCHW
            return x

        if x.dim() == 4:
            if self.layout != "BCHW":
                # allow BCHW even if user forgot to set it
                pass
            return self._patchmix_4d(x)

        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    def _pad_hw(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B, C, H, W)
        _, _, h, w = x.shape
        ps = self.patch_size
        pad_h = (ps - (h % ps)) % ps
        pad_w = (ps - (w % ps)) % ps
        if (pad_h == 0 and pad_w == 0) or (not self.pad_to_multiple):
            return x, 0, 0
        # pad last two dims: (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return x, pad_h, pad_w

    def _patchmix_4d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        dtype = x.dtype
        device = x.device

        x_pad, pad_h, pad_w = self._pad_hw(x)
        b, c, hp, wp = x_pad.shape
        ps = self.patch_size

        # unfold into non-overlapping patches: (B, C*ps*ps, N)
        patches = F.unfold(x_pad, kernel_size=ps, stride=ps)
        # (B, N, D)
        patches = patches.transpose(1, 2)
        b, n, d = patches.shape

        # per-sample random permutation (PatchMix shuffling)
        # perm: (B, N)
        perm = torch.rand((b, n), device=device).argsort(dim=1)
        shuffled = patches.gather(1, perm.unsqueeze(-1).expand(b, n, d))

        # patch-wise decide whether to mix
        mix_mask = (torch.rand((b, n), device=device) < self.p).unsqueeze(-1)  # (B, N, 1)

        # lambda per patch from Beta(alpha, beta)
        # clamp to avoid rare numerical issues for extreme params
        beta_dist = torch.distributions.Beta(
            torch.tensor(self.alpha, device=device, dtype=torch.float32),
            torch.tensor(self.beta, device=device, dtype=torch.float32),
        )
        lam = beta_dist.sample((b, n)).to(device=device, dtype=torch.float32).clamp_(0.0, 1.0).unsqueeze(-1)

        # mix (float) then cast back to original dtype
        patches_f = patches.to(torch.float32)
        shuffled_f = shuffled.to(torch.float32)
        mixed = lam * patches_f + (1.0 - lam) * shuffled_f
        out = torch.where(mix_mask, mixed, patches_f).to(dtype)

        # fold back
        out = out.transpose(1, 2)  # (B, D, N)
        x_out = F.fold(out, output_size=(hp, wp), kernel_size=ps, stride=ps)

        # crop padding back
        if pad_h or pad_w:
            x_out = x_out[:, :, : hp - pad_h, : wp - pad_w]
        return x_out


class _PostEncWrapper(nn.Module):
    """Wrap a network and apply a post-encoding augmentation before the net forward."""
    def __init__(self, net: nn.Module, aug: nn.Module):
        super().__init__()
        self.net = net
        self.aug = aug

    def forward(self, x, *args, **kwargs):
        x = self.aug(x)
        return self.net(x, *args, **kwargs)


def attach_postenc_aug(net: nn.Module, aug: Optional[nn.Module]) -> nn.Module:
    """Return net wrapped with aug, or net unchanged if aug is None."""
    if aug is None:
        return net
    return _PostEncWrapper(net, aug)
