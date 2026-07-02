# models/center_patch_minlift.py

import torch
import torch.nn as nn


class CenterPatchMinLift(nn.Module):
    """
    Pre-encoding image augmentation for frame-based inputs [B, C, H, W].

    It takes the center patch and lifts all values in that patch toward the
    per-image, per-channel maximum inside the same patch:

        patch' = (1 - alpha) * patch + alpha * patch_max

    This increases the minimum intensity in the center patch while keeping the
    local maximum unchanged.

    Notes:
    - intended for standard image inputs, not DVS spike tensors
    - applied with probability p
    - alpha is sampled once per forward call, matching the simple SDT-style logic
    """

    def __init__(
        self,
        patch_frac=0.5,
        alpha_min=0.3,
        alpha_max=0.7,
        p=0.5,
        inplace=False,
    ):
        super(CenterPatchMinLift, self).__init__()

        self.patch_frac = float(patch_frac)
        self.alpha_min = float(min(alpha_min, alpha_max))
        self.alpha_max = float(max(alpha_min, alpha_max))
        self.p = float(p)
        self.inplace = bool(inplace)

    def extra_repr(self):
        return (
            "patch_frac={:.4f}, alpha_min={:.4f}, alpha_max={:.4f}, "
            "p={:.4f}, inplace={}".format(
                self.patch_frac,
                self.alpha_min,
                self.alpha_max,
                self.p,
                self.inplace,
            )
        )

    def _get_center_patch_bounds(self, h, w):
        ph = max(1, int(round(h * self.patch_frac)))
        pw = max(1, int(round(w * self.patch_frac)))

        ph = min(ph, h)
        pw = min(pw, w)

        y0 = (h - ph) // 2
        x0 = (w - pw) // 2
        y1 = y0 + ph
        x1 = x0 + pw
        return y0, y1, x0, x1

    def forward(self, x):
        if x is None:
            return x

        # only standard image tensors are supported here
        if x.dim() != 4:
            return x

        if not self.training:
            return x

        if self.p <= 0.0:
            return x

        if torch.rand(1, device=x.device).item() > self.p:
            return x

        b, c, h, w = x.shape
        y0, y1, x0, x1 = self._get_center_patch_bounds(h, w)

        out = x if self.inplace else x.clone()
        patch = out[:, :, y0:y1, x0:x1]  # [B, C, ph, pw]

        # per-image, per-channel max in the center patch
        patch_max = patch.amax(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # sample one alpha for the whole batch, same spirit as the SDT version
        alpha = torch.empty(
            1, device=out.device, dtype=out.dtype
        ).uniform_(self.alpha_min, self.alpha_max)

        patch_aug = (1.0 - alpha) * patch + alpha * patch_max
        out[:, :, y0:y1, x0:x1] = patch_aug
        return out