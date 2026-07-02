import torch


class PreEncPatchDropout2D:
    """
    Pre-encoding PatchDropout for CNN / MS-ResNet training.

    This mirrors the Spike-Driven-Transformer version:
    - randomly keep a subset of non-overlapping spatial patches
    - zero out the dropped patches
    - apply during TRAINING only
    - use full image during validation / test

    Supports:
    - 4D input: (B, C, H, W)
    - 5D input: (B, T, C, H, W)   [same spatial mask for all time steps]

    Notes:
    - This is not token dropping inside a ViT. It is the input-space analogue,
      which is the correct pre-encoding version for MS-ResNet.
    """

    def __init__(self, patch_h=4, patch_w=None, keep=1.0, min_keep=None):
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w) if patch_w is not None else int(patch_h)
        self.keep = float(keep)
        self.min_keep = None if min_keep is None else float(min_keep)
        self._warned_shape = False

    def _sample_keep(self, device):
        if self.min_keep is not None:
            keep_rate = torch.empty(1, device=device).uniform_(self.min_keep, 1.0).item()
            return max(min(keep_rate, 1.0), 0.0)
        return max(min(self.keep, 1.0), 0.0)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # No-op
        if self.keep >= 1.0 and self.min_keep is None:
            return x

        is_5d = False
        if x.dim() == 4:
            # (B, C, H, W)
            B, C, H, W = x.shape
        elif x.dim() == 5:
            # (B, T, C, H, W)
            is_5d = True
            B, T, C, H, W = x.shape
        else:
            if not self._warned_shape:
                print(
                    f"[PreEncPatchDropout2D] Unsupported input dim={x.dim()}, skipping.",
                    flush=True
                )
                self._warned_shape = True
            return x

        ph, pw = self.patch_h, self.patch_w

        # Need exact tiling
        if H < ph or W < pw or (H % ph != 0) or (W % pw != 0):
            if not self._warned_shape:
                print(
                    f"[PreEncPatchDropout2D] HxW={H}x{W} not divisible by patch size "
                    f"{ph}x{pw}; skipping for this batch.",
                    flush=True
                )
                self._warned_shape = True
            return x

        device = x.device
        keep_rate = self._sample_keep(device)
        if keep_rate >= 1.0:
            return x

        grid_h = H // ph
        grid_w = W // pw
        num_patches = grid_h * grid_w
        num_keep = max(1, int(round(keep_rate * num_patches)))

        if num_keep >= num_patches:
            return x

        # Vectorized per-sample random subset without replacement
        rand = torch.rand(B, num_patches, device=device)
        keep_idx = torch.argsort(rand, dim=1)[:, :num_keep]  # (B, num_keep)

        # Build patch-grid mask: (B, num_patches)
        mask = torch.zeros(B, num_patches, device=device, dtype=x.dtype)
        mask.scatter_(1, keep_idx, 1.0)

        # Expand patch-grid mask to pixel mask: (B, 1, H, W)
        mask = mask.view(B, 1, grid_h, grid_w)
        mask = mask.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)

        if not is_5d:
            # (B, C, H, W) * (B, 1, H, W)
            x = x * mask
        else:
            # same spatial mask across all time steps
            # (B, T, C, H, W) * (B, 1, 1, H, W)
            mask = mask.unsqueeze(1)  # (B, 1, 1, H, W)
            x = x * mask

        return x