import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class PreEncodingPatchMix2D(nn.Module):
    """
    Pre-encoding PatchMix on BCHW input images.

    - Single-image patch-level mixup
    - Same patch mixing is shared across channels within each sample
    - Training only by design (returns input unchanged in eval mode)
    """

    def __init__(
        self,
        p: float = 0.5,
        patch_size: int = 16,
        alpha: float = 0.2,
        beta: float = 0.2,
        pad_to_multiple: bool = True,
    ):
        super().__init__()
        self.p = float(p)
        self.patch_size = int(patch_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pad_to_multiple = bool(pad_to_multiple)

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"alpha and beta must be > 0, got alpha={self.alpha}, beta={self.beta}")

    def extra_repr(self) -> str:
        return (
            f"p={self.p}, patch_size={self.patch_size}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"pad_to_multiple={self.pad_to_multiple}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x

        if not isinstance(x, torch.Tensor):
            return x

        if x.ndim != 4:
            return x  # expect BCHW

        B, C, H, W = x.shape
        ps = self.patch_size

        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps

        if (pad_h != 0 or pad_w != 0):
            if not self.pad_to_multiple:
                raise ValueError(
                    f"PatchMix requires H and W divisible by patch_size when pad_to_multiple=False. "
                    f"Got H={H}, W={W}, patch_size={ps}"
                )
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        B, C, H2, W2 = x.shape
        gh, gw = H2 // ps, W2 // ps
        N = gh * gw

        # BCHW -> B, N, C, ps, ps
        patches = (
            x.reshape(B, C, gh, ps, gw, ps)
             .permute(0, 2, 4, 1, 3, 5)
             .contiguous()
             .reshape(B, N, C, ps, ps)
        )

        mixed = torch.empty_like(patches)

        beta_dist = Beta(
            torch.tensor(self.alpha, device=x.device, dtype=torch.float32),
            torch.tensor(self.beta, device=x.device, dtype=torch.float32),
        )

        for b in range(B):
            perm = torch.randperm(N, device=x.device)
            shuffled = patches[b, perm]  # (N, C, ps, ps)

            mix_mask = (torch.rand(N, device=x.device) < self.p)  # (N,)
            lam = beta_dist.sample((N,)).to(device=x.device, dtype=x.dtype)  # (N,)
            lam = torch.where(mix_mask, lam, torch.ones_like(lam))
            lam = lam.view(N, 1, 1, 1)

            mixed[b] = lam * patches[b] + (1.0 - lam) * shuffled

        # fold back
        out = (
            mixed.reshape(B, gh, gw, C, ps, ps)
                 .permute(0, 3, 1, 4, 2, 5)
                 .contiguous()
                 .reshape(B, C, H2, W2)
        )

        if pad_h != 0 or pad_w != 0:
            out = out[:, :, :H, :W]

        return out