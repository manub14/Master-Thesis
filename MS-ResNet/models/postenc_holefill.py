# models/postenc_holefill.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PostEncHoleFill(nn.Module):
    """
    Post-encoding hole-fill augmentation for spike tensors.

    Supports:
        - spatial:
            fill a zero spike if all 8 spatial neighbours are 1
        - spatiotemporal:
            fill a zero spike if all 26 neighbours in (t, h, w) are 1

    Intended layout for MS-ResNet:
        TBCHW

    Also supports:
        BTCHW
        BCHW  -> treated as T=1

    Notes:
        - Works on spike-like tensors (usually 0/1 or >=0 values after mem_update)
        - Applies only during training
        - Can be applied per-sample with probability p
        - Does not remove or overwrite other aug methods
    """

    def __init__(self, mode="spatial", p=1.0, layout="TBCHW", time_steps=None):
        super(PostEncHoleFill, self).__init__()

        if mode is not None:
            mode = str(mode).lower()

        if mode not in (None, "spatial", "spatiotemporal"):
            raise ValueError(
                "Unsupported holefill mode: {}. Use one of: None, 'spatial', 'spatiotemporal'".format(mode)
            )

        self.mode = mode
        self.p = float(p)
        self.layout = str(layout).upper()
        self.time_steps = time_steps

    def extra_repr(self):
        return "mode={}, p={}, layout={}, time_steps={}".format(
            self.mode, self.p, self.layout, self.time_steps
        )

    def _to_btchw(self, x):
        """
        Convert supported layouts to BTCHW for internal processing.
        Returns:
            x_btchw, original_layout
        """
        if x.dim() == 4:
            # BCHW -> BTCHW with T=1
            return x.unsqueeze(1), "BCHW"

        if x.dim() != 5:
            return x, None

        if self.layout == "BTCHW":
            return x, "BTCHW"

        if self.layout == "TBCHW":
            return x.permute(1, 0, 2, 3, 4).contiguous(), "TBCHW"

        # fallback heuristic
        if self.time_steps is not None:
            if x.shape[1] == self.time_steps:
                return x, "BTCHW"
            if x.shape[0] == self.time_steps:
                return x.permute(1, 0, 2, 3, 4).contiguous(), "TBCHW"

        # default assume TBCHW because that is what you use in MS-ResNet
        return x.permute(1, 0, 2, 3, 4).contiguous(), "TBCHW"

    def _from_btchw(self, x, original_layout):
        if original_layout == "BCHW":
            return x.squeeze(1)
        if original_layout == "TBCHW":
            return x.permute(1, 0, 2, 3, 4).contiguous()
        return x

    def _spatial_fill(self, x_btchw):
        """
        x_btchw: [B, T, C, H, W]
        Fill a zero if all 8 neighbours in 2D are 1.
        """
        b, t, c, h, w = x_btchw.shape

        x_bin = (x_btchw > 0).to(x_btchw.dtype)

        x_bin_2d = x_bin.reshape(b * t * c, 1, h, w)
        x_orig_2d = x_btchw.reshape(b * t * c, 1, h, w)

        kernel = torch.ones((1, 1, 3, 3), dtype=x_btchw.dtype, device=x_btchw.device)
        kernel[:, :, 1, 1] = 0.0

        neigh = F.conv2d(x_bin_2d, kernel, padding=1)
        fill_mask = (x_bin_2d == 0) & (neigh == 8)

        x_filled_2d = torch.where(fill_mask, torch.ones_like(x_orig_2d), x_orig_2d)
        x_filled = x_filled_2d.reshape(b, t, c, h, w)
        return x_filled

    def _spatiotemporal_fill(self, x_btchw):
        """
        x_btchw: [B, T, C, H, W]
        Fill a zero if all 26 neighbours in 3D (T,H,W) are 1.
        Channel is treated independently.
        """
        b, t, c, h, w = x_btchw.shape

        x_bin = (x_btchw > 0).to(x_btchw.dtype)

        # [B, T, C, H, W] -> [B*C, 1, T, H, W]
        x_bin_3d = (
            x_bin.permute(0, 2, 1, 3, 4).contiguous().reshape(b * c, 1, t, h, w)
        )
        x_orig_3d = (
            x_btchw.permute(0, 2, 1, 3, 4).contiguous().reshape(b * c, 1, t, h, w)
        )

        kernel = torch.ones((1, 1, 3, 3, 3), dtype=x_btchw.dtype, device=x_btchw.device)
        kernel[:, :, 1, 1, 1] = 0.0

        neigh = F.conv3d(x_bin_3d, kernel, padding=1)
        fill_mask = (x_bin_3d == 0) & (neigh == 26)

        x_filled_3d = torch.where(fill_mask, torch.ones_like(x_orig_3d), x_orig_3d)

        x_filled = (
            x_filled_3d.reshape(b, c, t, h, w).permute(0, 2, 1, 3, 4).contiguous()
        )
        return x_filled

    def forward(self, x):
        if self.mode is None:
            return x

        if not self.training:
            return x

        if self.p <= 0.0:
            return x

        x_btchw, original_layout = self._to_btchw(x)
        if original_layout is None:
            return x

        if x_btchw.numel() == 0:
            return x

        if self.mode == "spatial":
            x_aug = self._spatial_fill(x_btchw)
        elif self.mode == "spatiotemporal":
            x_aug = self._spatiotemporal_fill(x_btchw)
        else:
            return x

        # per-sample application
        if self.p < 1.0:
            b = x_btchw.shape[0]
            apply_mask = (torch.rand(b, device=x_btchw.device) < self.p).view(b, 1, 1, 1, 1)
            x_btchw = torch.where(apply_mask, x_aug, x_btchw)
        else:
            x_btchw = x_aug

        return self._from_btchw(x_btchw, original_layout)