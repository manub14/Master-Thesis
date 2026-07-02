# models/frequency_encoding_aug.py

import torch
import torch.nn as nn


def make_default_fe_radii(height, width, time_steps):
    """
    Default radii schedule following the FEEL-SNN style:
    for 32x32, T=4  -> [16, 14, 12, 10]
    for 32x32, T=8  -> [16, 14, 12, 10, 8, 6, 4, 2]
    for 64x64, T=4  -> [32, 30, 28, 26]

    For other T, we keep the same descending-by-2 pattern and clamp at 1.
    """
    max_r = min(int(height), int(width)) // 2
    return [max(1, max_r - 2 * i) for i in range(int(time_steps))]


class FrequencyEncodingAug(nn.Module):
    """
    Frequency Encoding (FE) for static image inputs.

    Input:
        x: [B, C, H, W]

    Output:
        by default: [T, B, C, H, W]  (TBCHW)

    Method:
        1) FFT2
        2) fftshift so low-frequency is centered
        3) apply a centered box low-pass mask with radius r_t
        4) inverse FFT
        5) stack outputs across time
    """

    def __init__(self, T, radii=None, p=1.0, jitter=0, return_layout="TBCHW"):
        super(FrequencyEncodingAug, self).__init__()
        self.T = int(T)
        self.p = float(p)
        self.jitter = int(jitter)
        self.return_layout = str(return_layout).upper()

        if self.return_layout not in ("TBCHW", "BTCHW"):
            raise ValueError(
                "return_layout must be 'TBCHW' or 'BTCHW', got '{}'.".format(self.return_layout)
            )

        # radii may be None here; if so we infer them from H/W at forward time
        if radii is not None:
            radii = [int(r) for r in radii]
            if len(radii) != self.T:
                raise ValueError(
                    "Length of radii ({}) must match T ({}).".format(len(radii), self.T)
                )

        self.radii = radii
        self._mask_cache = {}

    def extra_repr(self):
        return "T={}, radii={}, p={}, jitter={}, return_layout={}".format(
            self.T, self.radii, self.p, self.jitter, self.return_layout
        )

    def _repeat_input(self, x):
        # x: [B, C, H, W] -> [T, B, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        if self.return_layout == "BTCHW":
            x_seq = x_seq.permute(1, 0, 2, 3, 4).contiguous()
        return x_seq

    def _resolve_radii(self, H, W, device):
        max_r = max(1, min(H, W) // 2)

        if self.radii is None:
            radii = make_default_fe_radii(H, W, self.T)
        else:
            radii = [max(1, min(max_r, int(r))) for r in self.radii]

        if self.training and self.jitter > 0:
            noise = torch.randint(
                low=-self.jitter,
                high=self.jitter + 1,
                size=(len(radii),),
                device=device,
            )
            radii = [
                max(1, min(max_r, radii[i] + int(noise[i].item())))
                for i in range(len(radii))
            ]
            # keep descending schedule
            radii = sorted(radii, reverse=True)

        return radii

    def _get_box_mask(self, H, W, radius, device, dtype):
        key = (int(H), int(W), int(radius), str(device), str(dtype))
        mask = self._mask_cache.get(key, None)
        if mask is not None:
            return mask

        mask = torch.zeros((1, 1, H, W), device=device, dtype=dtype)

        cy = H // 2
        cx = W // 2

        y0 = max(0, cy - int(radius))
        y1 = min(H, cy + int(radius) + 1)
        x0 = max(0, cx - int(radius))
        x1 = min(W, cx + int(radius) + 1)

        mask[:, :, y0:y1, x0:x1] = 1.0
        self._mask_cache[key] = mask
        return mask

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns:
            [T, B, C, H, W] by default
        """
        if x.dim() != 4:
            raise ValueError(
                "FrequencyEncodingAug expects a 4D tensor [B, C, H, W], got shape {}."
                .format(tuple(x.shape))
            )

        # training-time probability gate
        if self.training and self.p < 1.0:
            if torch.rand(1, device=x.device).item() > self.p:
                return self._repeat_input(x)

        x_in_dtype = x.dtype
        x = x.float()  # safest for torch.fft

        B, C, H, W = x.shape
        radii = self._resolve_radii(H, W, x.device)

        # [B, C, H, W] complex
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))

        frames = []
        for r in radii:
            mask = self._get_box_mask(H, W, r, x.device, x.dtype)  # real mask
            x_masked = x_fft * mask
            x_masked = torch.fft.ifftshift(x_masked, dim=(-2, -1))
            x_t = torch.fft.ifft2(x_masked, dim=(-2, -1)).real
            x_t = x_t.to(dtype=x_in_dtype)
            frames.append(x_t)

        x_seq = torch.stack(frames, dim=0)  # [T, B, C, H, W]

        if self.return_layout == "BTCHW":
            x_seq = x_seq.permute(1, 0, 2, 3, 4).contiguous()

        return x_seq


class PreEncodingFrequencyEncoding(nn.Module):
    """
    Wrapper used by MS_ResNet.py.

    This is the model-side pre-encoding FE module that:
      - accepts static input [B, C, H, W]
      - returns temporal input [T, B, C, H, W] (or BTCHW if requested)
      - respects apply_in_eval
      - falls back to direct repeat when FE is disabled during eval

    This makes it compatible with both:
      - train_amp.py attachment style via `core.preenc_fe = ...`
      - direct construction inside MS_ResNet.py
    """

    def __init__(
        self,
        T,
        radii=None,
        p=1.0,
        jitter=0,
        layout="TBCHW",
        apply_in_eval=False,
    ):
        super(PreEncodingFrequencyEncoding, self).__init__()
        self.T = int(T)
        self.apply_in_eval = bool(apply_in_eval)
        self.layout = str(layout).upper()

        if self.layout not in ("TBCHW", "BTCHW"):
            raise ValueError(
                "layout must be 'TBCHW' or 'BTCHW', got '{}'.".format(self.layout)
            )

        self.fe = FrequencyEncodingAug(
            T=self.T,
            radii=radii,
            p=p,
            jitter=jitter,
            return_layout=self.layout,
        )

    def extra_repr(self):
        return "T={}, layout={}, apply_in_eval={}".format(
            self.T, self.layout, self.apply_in_eval
        )

    def _repeat_input(self, x):
        return self.fe._repeat_input(x)

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns:
            [T, B, C, H, W] or [B, T, C, H, W]
        """
        if x.dim() != 4:
            raise ValueError(
                "PreEncodingFrequencyEncoding expects a 4D tensor [B, C, H, W], got shape {}."
                .format(tuple(x.shape))
            )

        # In eval, if not explicitly enabled, preserve original MS-ResNet behavior:
        # direct repeat across time with no FE transform.
        if (not self.training) and (not self.apply_in_eval):
            return self._repeat_input(x)

        return self.fe(x)