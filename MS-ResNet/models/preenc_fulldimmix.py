import torch
import torch.nn as nn


class PreEncFullDimMix(nn.Module):
    """
    Full-dimension mixup on 5D tensors.

    Supported layouts:
        - TBCHW : [T, B, C, H, W]  (default for MS-ResNet)
        - BTCHW : [B, T, C, H, W]

    This is the MS-ResNet version of the SDT 5D FullDimMix idea:
        y = alpha * x + (1 - alpha) * x_perm

    Notes:
        - Only acts on 5D tensors.
        - Leaves labels unchanged.
        - Does not interfere with other augmentations because it uses its own
          unique module and unique CLI args.
    """

    def __init__(self, alpha=0.5, p=0.0, layout="TBCHW", apply_in_eval=False):
        super(PreEncFullDimMix, self).__init__()
        self.alpha = float(alpha)
        self.p = float(p)
        self.layout = str(layout).upper()
        self.apply_in_eval = bool(apply_in_eval)

        if self.layout not in ("TBCHW", "BTCHW"):
            raise ValueError(
                "Unsupported layout '{}'. Expected 'TBCHW' or 'BTCHW'.".format(self.layout)
            )

    def extra_repr(self):
        return "alpha={}, p={}, layout='{}', apply_in_eval={}".format(
            self.alpha, self.p, self.layout, self.apply_in_eval
        )

    @staticmethod
    def _randperm_or_identity(n, device):
        if n > 1:
            return torch.randperm(n, device=device)
        return torch.arange(n, device=device)

    def forward(self, x):
        if x is None:
            return x

        if x.dim() != 5:
            return x

        if self.p <= 0.0:
            return x

        if (not self.training) and (not self.apply_in_eval):
            return x

        if torch.rand(1, device=x.device).item() > self.p:
            return x

        if self.layout == "TBCHW":
            # x: [T, B, C, H, W]
            T, B, C, H, W = x.shape
            perm_t = self._randperm_or_identity(T, x.device)
            perm_b = self._randperm_or_identity(B, x.device)
            perm_c = self._randperm_or_identity(C, x.device)
            perm_h = self._randperm_or_identity(H, x.device)
            perm_w = self._randperm_or_identity(W, x.device)

            x_perm = x.index_select(0, perm_t)
            x_perm = x_perm.index_select(1, perm_b)
            x_perm = x_perm.index_select(2, perm_c)
            x_perm = x_perm.index_select(3, perm_h)
            x_perm = x_perm.index_select(4, perm_w)

        else:
            # x: [B, T, C, H, W]
            B, T, C, H, W = x.shape
            perm_b = self._randperm_or_identity(B, x.device)
            perm_t = self._randperm_or_identity(T, x.device)
            perm_c = self._randperm_or_identity(C, x.device)
            perm_h = self._randperm_or_identity(H, x.device)
            perm_w = self._randperm_or_identity(W, x.device)

            x_perm = x.index_select(0, perm_b)
            x_perm = x_perm.index_select(1, perm_t)
            x_perm = x_perm.index_select(2, perm_c)
            x_perm = x_perm.index_select(3, perm_h)
            x_perm = x_perm.index_select(4, perm_w)

        return self.alpha * x + (1.0 - self.alpha) * x_perm