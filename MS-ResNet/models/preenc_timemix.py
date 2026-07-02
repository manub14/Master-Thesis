# models/preenc_timemix.py
# Python 3.7 compatible

import torch
import torch.nn as nn


class PreEncodingTimeMix(nn.Module):
    """
    TS-SNN-inspired temporal channel-group mixing for MS-ResNet pre-encoding.

    Expected input shape:
        [T, B, C, H, W]

    Behavior:
        - split channels into groups
        - choose 2 split points g1 < g2
        - first part uses t+1
        - middle part uses t-1
        - last part stays unchanged
        - residual combine: out = x + alpha * shifted

    Notes:
        - batch-consistent transform (same split for the whole batch)
        - stochastic application controlled by prob
        - safe for chaining with other pre-encoding augs through nn.Sequential
    """

    def __init__(
        self,
        prob=0.0,
        alpha=0.3,
        groups=32,
        random_split=True,
        apply_in_eval=False
    ):
        super(PreEncodingTimeMix, self).__init__()
        self.prob = float(prob)
        self.alpha = float(alpha)
        self.groups = int(groups)
        self.random_split = bool(random_split)
        self.apply_in_eval = bool(apply_in_eval)

    def extra_repr(self):
        return (
            "prob={:.3f}, alpha={:.3f}, groups={}, random_split={}, apply_in_eval={}"
            .format(
                self.prob,
                self.alpha,
                self.groups,
                self.random_split,
                self.apply_in_eval
            )
        )

    def _choose_split_points(self, num_groups, device):
        """
        Choose g1, g2 with 0 < g1 < g2 < num_groups
        """
        if num_groups < 3:
            return None, None

        if self.random_split:
            # choose two distinct internal boundaries from [1, num_groups - 1)
            candidates = torch.arange(1, num_groups, device=device)
            perm = candidates[torch.randperm(candidates.numel(), device=device)]
            g1 = int(torch.min(perm[0], perm[1]).item())
            g2 = int(torch.max(perm[0], perm[1]).item())
        else:
            g1 = max(1, num_groups // 3)
            g2 = max(g1 + 1, (2 * num_groups) // 3)
            if g2 >= num_groups:
                g2 = num_groups - 1
            if g1 >= g2:
                return None, None

        return g1, g2

    def _timemix_tbchw(self, x):
        """
        x: [T, B, C, H, W]
        """
        if x.dim() != 5:
            return x

        T, B, C, H, W = x.shape

        if T < 2:
            return x
        if C < 3:
            return x
        if self.groups <= 0:
            return x

        # clamp groups to channel count
        group_count = min(self.groups, C)
        fold = max(1, C // group_count)
        actual_groups = max(1, C // fold)

        if actual_groups < 3:
            return x

        g1, g2 = self._choose_split_points(actual_groups, x.device)
        if g1 is None or g2 is None:
            return x

        c1 = min(C, g1 * fold)
        c2 = min(C, g2 * fold)

        if c1 <= 0 or c2 <= c1:
            return x

        shifted = torch.zeros_like(x)

        # part 1: use t+1  (future -> current)
        shifted[:-1, :, :c1, :, :] = x[1:, :, :c1, :, :]

        # part 2: use t-1  (past -> current)
        shifted[1:, :, c1:c2, :, :] = x[:-1, :, c1:c2, :, :]

        # part 3: unchanged
        shifted[:, :, c2:, :, :] = x[:, :, c2:, :, :]

        out = x + self.alpha * shifted
        return out

    def forward(self, x):
        if x.dim() != 5:
            return x

        if self.prob <= 0.0:
            return x

        if (not self.training) and (not self.apply_in_eval):
            return x

        if torch.rand(1, device=x.device).item() > self.prob:
            return x

        return self._timemix_tbchw(x)