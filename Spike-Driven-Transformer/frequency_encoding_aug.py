# frequency_encoding_aug.py
import torch
from typing import List, Optional

def make_default_fe_radii(h: int, w: int, T: int) -> List[int]:
    """
    Paper-like decreasing radii schedule. For 32x32, T=4 -> [16, 14, 12, 10].
    For larger sizes, we scale similarly.
    """
    max_r = min(h, w) // 2
    if T <= 1:
        return [max_r]

    # Try to keep a gentle step like 2 for small images, scaled for larger.
    # This keeps "more->less" frequency over time without dropping too aggressively.
    if max_r <= 32:
        step = 2
    else:
        step = max(2, max_r // (T + 6))

    radii = [max_r - step * i for i in range(T)]
    # clamp to >= 1 and ensure non-increasing
    radii = [max(1, min(max_r, r)) for r in radii]
    radii = sorted(radii, reverse=True)
    return radii


def _box_lowpass_mask(h: int, w: int, r: int, device, dtype) -> torch.Tensor:
    """Square low-pass mask in shifted FFT domain."""
    cy, cx = h // 2, w // 2
    r = int(r)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, h)
    x0, x1 = max(cx - r, 0), min(cx + r + 1, w)

    mask = torch.zeros((h, w), device=device, dtype=dtype)
    mask[y0:y1, x0:x1] = 1.0
    return mask


class FrequencyEncodingAug(torch.nn.Module):
    """
    Frequency Encoding (FE) augmentation:
    Input:  [B,C,H,W]
    Output: [B,T,C,H,W]
    """
    def __init__(
        self,
        T: int,
        radii: Optional[List[int]] = None,
        p: float = 1.0,
        jitter: int = 0,
    ):
        super().__init__()
        self.T = int(T)
        self.radii = radii
        self.p = float(p)
        self.jitter = int(jitter)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W]
        returns: [B,T,C,H,W]
        """
        assert x.dim() == 4, f"FE expects [B,C,H,W], got {tuple(x.shape)}"
        B, C, H, W = x.shape
        device = x.device

        # prob gate (augmentation-style)
        if self.p < 1.0 and torch.rand((), device=device) > self.p:
            return x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)

        # pick radii
        if self.radii is None:
            radii = make_default_fe_radii(H, W, self.T)
        else:
            assert len(self.radii) == self.T, "radii must have length T"
            radii = list(map(int, self.radii))

        # jitter radii (optional)
        if self.jitter > 0:
            max_r = min(H, W) // 2
            radii_j = []
            for r in radii:
                delta = int(torch.randint(-self.jitter, self.jitter + 1, ()).item())
                radii_j.append(max(1, min(max_r, r + delta)))
            radii = sorted(radii_j, reverse=True)

        # FFT (vectorized over batch & channel)
        # torch.fft works best in float32/float64
        x_f = x.to(torch.float32)
        X = torch.fft.fft2(x_f, dim=(-2, -1))               # [B,C,H,W] complex
        Xs = torch.fft.fftshift(X, dim=(-2, -1))            # center low freq

        # build masks for each t (T,H,W) then broadcast to (1,1,T,H,W)
        masks = []
        for r in radii:
            masks.append(_box_lowpass_mask(H, W, r, device=device, dtype=x_f.dtype))
        M = torch.stack(masks, dim=0)                       # [T,H,W]
        M = M.view(1, 1, self.T, H, W)                      # [1,1,T,H,W]

        # apply each mask to spectrum -> inverse
        Xs = Xs.unsqueeze(2)                                # [B,C,1,H,W]
        Y = Xs * M                                          # [B,C,T,H,W]
        Y = torch.fft.ifftshift(Y, dim=(-2, -1))
        y = torch.fft.ifft2(Y, dim=(-2, -1)).real           # [B,C,T,H,W]

        # return [B,T,C,H,W]
        y = y.permute(0, 2, 1, 3, 4).contiguous()
        return y
