# models/holefill.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class HoleFillPostEncoding(nn.Module):
    """
    Post-encoding HoleFill augmentation for spike tensors.

    Supports:
        1) spatial
           Fill a zero spike if all 8 spatial neighbours are 1
           independently at each time step.

        2) spatiotemporal
           Fill a zero spike if all 26 neighbours in the local
           3D (time, height, width) neighbourhood are 1.

    Supported layouts:
        - TBCHW
        - BTCHW
        - BCHW (treated as T=1)

    Notes:
        - train-only by default unless apply_in_eval=True
        - applied per sample with probability p
        - channel dimension is processed independently
        - does not overwrite any other augmentation method
    """

    def __init__(
        self,
        p: float = 1.0,
        mode: str = "spatiotemporal",
        layout: str = "TBCHW",
        apply_in_eval: bool = False,
    ):
        super().__init__()

        self.p = float(p)
        self.mode = str(mode).lower()
        self.layout = str(layout).upper()
        self.apply_in_eval = bool(apply_in_eval)

        if self.mode not in ("spatial", "spatiotemporal"):
            raise ValueError(
                "HoleFillPostEncoding mode must be 'spatial' or 'spatiotemporal', "
                f"got: {mode}"
            )

        if self.layout not in ("TBCHW", "BTCHW", "BCHW"):
            raise ValueError(
                "HoleFillPostEncoding layout must be one of "
                "['TBCHW', 'BTCHW', 'BCHW'], "
                f"got: {layout}"
            )

    def extra_repr(self):
        return (
            f"p={self.p}, mode='{self.mode}', layout='{self.layout}', "
            f"apply_in_eval={self.apply_in_eval}"
        )

    def _to_btchw(self, x: torch.Tensor):
        """
        Convert input to BTCHW for internal processing.
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
        elif self.layout == "TBCHW":
            return x.permute(1, 0, 2, 3, 4).contiguous(), "TBCHW"
        elif self.layout == "BCHW":
            return x.unsqueeze(1), "BCHW"

        return x, None

    def _from_btchw(self, x: torch.Tensor, original_layout: str):
        """
        Restore original layout.
        """
        if original_layout == "BTCHW":
            return x
        elif original_layout == "TBCHW":
            return x.permute(1, 0, 2, 3, 4).contiguous()
        elif original_layout == "BCHW":
            return x.squeeze(1)
        return x

    def _fill_spatial(self, x_btchw: torch.Tensor) -> torch.Tensor:
        """
        Spatial hole fill:
        fill 0 if all 8 spatial neighbours are 1.

        x_btchw: [B, T, C, H, W]
        """
        b, t, c, h, w = x_btchw.shape

        x_bin = (x_btchw > 0).to(x_btchw.dtype)

        # process each (B,T,C) plane independently
        x2 = x_bin.reshape(b * t * c, 1, h, w)
        x_orig = x_btchw.reshape(b * t * c, 1, h, w)

        kernel = torch.ones((1, 1, 3, 3), device=x_btchw.device, dtype=x_btchw.dtype)
        kernel[:, :, 1, 1] = 0.0

        neigh_count = F.conv2d(x2, kernel, padding=1)
        fill_mask = (x2 == 0) & (neigh_count == 8)

        out = torch.where(fill_mask, torch.ones_like(x_orig), x_orig)
        return out.reshape(b, t, c, h, w)

    def _fill_spatiotemporal(self, x_btchw: torch.Tensor) -> torch.Tensor:
        """
        Spatiotemporal hole fill:
        fill 0 if all 26 neighbours in (T,H,W) are 1.

        x_btchw: [B, T, C, H, W]
        """
        b, t, c, h, w = x_btchw.shape

        x_bin = (x_btchw > 0).to(x_btchw.dtype)

        # [B, T, C, H, W] -> [B*C, 1, T, H, W]
        x3 = (
            x_bin.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(b * c, 1, t, h, w)
        )
        x_orig = (
            x_btchw.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(b * c, 1, t, h, w)
        )

        kernel = torch.ones((1, 1, 3, 3, 3), device=x_btchw.device, dtype=x_btchw.dtype)
        kernel[:, :, 1, 1, 1] = 0.0

        neigh_count = F.conv3d(x3, kernel, padding=1)
        fill_mask = (x3 == 0) & (neigh_count == 26)

        out = torch.where(fill_mask, torch.ones_like(x_orig), x_orig)

        out = (
            out.reshape(b, c, t, h, w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        return out

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if (not self.training) and (not self.apply_in_eval):
            return x

        if x is None:
            return x

        if self.p <= 0.0:
            return x

        x_btchw, original_layout = self._to_btchw(x)
        if original_layout is None:
            return x

        if x_btchw.dim() != 5:
            return x

        b = x_btchw.size(0)
        if b == 0:
            return x

        if self.mode == "spatial":
            x_filled = self._fill_spatial(x_btchw)
        else:
            x_filled = self._fill_spatiotemporal(x_btchw)

        if self.p < 1.0:
            apply_mask = (torch.rand(b, device=x_btchw.device) < self.p).view(b, 1, 1, 1, 1)
            x_btchw = torch.where(apply_mask, x_filled, x_btchw)
        else:
            x_btchw = x_filled

        return self._from_btchw(x_btchw, original_layout)