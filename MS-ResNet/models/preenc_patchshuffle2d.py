import torch
import torch.nn as nn


class PreEncodingPatchShuffle2D(nn.Module):
    """
    PatchShuffle for PRE-ENCODING / INPUT images.

    Supported input:
      - [B, C, H, W]
      - [B, T, C, H, W]   (supported for safety, though MS-ResNet uses BCHW input)

    Behavior:
      - train-only
      - Bernoulli(p) per sample
      - independent permutation per patch
      - same permutation shared across channels
      - border area that does not fit full patches is left unchanged
    """
    def __init__(self, p=0.05, patch_size=2, same_perm_on_batch=False):
        super().__init__()
        self.p = float(p)
        self.same_perm_on_batch = bool(same_perm_on_batch)

        if isinstance(patch_size, int):
            self.patch_h = int(patch_size)
            self.patch_w = int(patch_size)
        elif isinstance(patch_size, (tuple, list)):
            if len(patch_size) == 1:
                self.patch_h = int(patch_size[0])
                self.patch_w = int(patch_size[0])
            elif len(patch_size) == 2:
                self.patch_h = int(patch_size[0])
                self.patch_w = int(patch_size[1])
            else:
                raise ValueError("patch_size must be int, [P], or [Ph, Pw]")
        else:
            raise ValueError("patch_size must be int, tuple, or list")

    def extra_repr(self):
        return (
            f"p={self.p}, patch_size=({self.patch_h}, {self.patch_w}), "
            f"same_perm_on_batch={self.same_perm_on_batch}"
        )

    def forward(self, x):
        if (not self.training) or self.p <= 0.0:
            return x

        if not isinstance(x, torch.Tensor):
            return x

        orig_shape = x.shape

        # Support BCHW and BTCHW / TBCHW-like flattening if needed
        if x.dim() == 4:
            # [B, C, H, W]
            b, c, h, w = x.shape
            x_flat = x
            need_restore_5d = False
        elif x.dim() == 5:
            # [B, T, C, H, W]
            b, t, c, h, w = x.shape
            x_flat = x.reshape(b * t, c, h, w)
            need_restore_5d = True
        else:
            return x

        ph, pw = self.patch_h, self.patch_w
        if ph <= 1 and pw <= 1:
            return x

        grid_h = h // ph
        grid_w = w // pw
        if grid_h == 0 or grid_w == 0:
            return x

        h_main = grid_h * ph
        w_main = grid_w * pw
        p_area = ph * pw
        bflat = x_flat.shape[0]
        device = x_flat.device

        # Output copy so border stays untouched
        out = x_flat.clone()

        # Only the region covered by full patches
        x_main = x_flat[:, :, :h_main, :w_main]

        # [B, C, Hm, Wm]
        # -> [B, C, gh, ph, gw, pw]
        # -> [B, gh, gw, C, ph, pw]
        x_main = (
            x_main.view(bflat, c, grid_h, ph, grid_w, pw)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
        )

        # -> [B, gh, gw, C, ph*pw]
        x_patch = x_main.view(bflat, grid_h, grid_w, c, p_area)

        # Bernoulli per sample
        apply_mask = torch.rand(bflat, device=device) < self.p

        if apply_mask.any():
            if self.same_perm_on_batch:
                # one patch permutation field shared across batch
                rand = torch.rand(grid_h, grid_w, p_area, device=device)
                perm = rand.argsort(dim=-1).unsqueeze(0).expand(bflat, -1, -1, -1)
            else:
                # independent permutation field per sample
                rand = torch.rand(bflat, grid_h, grid_w, p_area, device=device)
                perm = rand.argsort(dim=-1)

            # same permutation across channels
            perm = perm.unsqueeze(3).expand(-1, -1, -1, c, -1)

            x_shuf = x_patch.gather(dim=-1, index=perm)

            mask = apply_mask.view(bflat, 1, 1, 1, 1)
            x_patch = torch.where(mask, x_shuf, x_patch)

        # Restore
        x_main = (
            x_patch.view(bflat, grid_h, grid_w, c, ph, pw)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(bflat, c, h_main, w_main)
        )

        out[:, :, :h_main, :w_main] = x_main

        if need_restore_5d:
            out = out.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4])

        return out