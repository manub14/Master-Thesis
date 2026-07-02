import torch
import torch.nn as nn


class PostEncodingTimeShuffle2D(nn.Module):
    """
    TS-SNN style Temporal Shift module for spike feature maps.

    Input shape:  [T, B, C, H, W]
    Output shape: [T, B, C, H, W]

    Steps:
      - Split channels into C_k groups (folding)
      - Choose two random split points g1 < g2
      - Shift first part left in time (+shift)
      - Shift middle part right in time (-shift)
      - Keep rest unchanged
      - Residual shift: X + alpha * Z
    """
    def __init__(self, p: float = 0.0, max_shift: int = 1, fold_k: int = 32, alpha: float = 0.3):
        super().__init__()
        self.p = float(p)
        self.max_shift = int(max_shift)
        self.fold_k = int(fold_k)
        self.alpha = float(alpha)

        if self.max_shift < 1:
            raise ValueError("max_shift must be >= 1")
        if self.fold_k < 1:
            raise ValueError("fold_k must be >= 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [T,B,C,H,W]
        if x.dim() != 5:
            return x

        if (not self.training) or (self.p <= 0.0):
            return x

        # apply with probability p
        if torch.rand((), device=x.device) > self.p:
            return x

        T, B, C, H, W = x.shape

        # Decide number of groups (C_k) and fold size (C_fold)
        groups = min(self.fold_k, C)
        if groups < 3:
            return x

        fold = C // groups
        if fold < 1:
            return x

        # Channels that are neatly covered by folding
        C_used = fold * groups

        # Random split points among channel-groups (avoid 0 and groups)
        # choose two distinct integers from [1, groups-1]
        idx = torch.randint(1, groups, (2,), device=x.device)
        if idx[0] == idx[1]:
            idx[1] = (idx[1] % (groups - 1)) + 1

        g1 = int(torch.min(idx).item())
        g2 = int(torch.max(idx).item())

        if not (0 < g1 < g2 < groups):
            return x

        c1 = g1 * fold
        c2 = g2 * fold

        # random shift magnitude in [1..max_shift]
        s = int(torch.randint(1, self.max_shift + 1, (1,), device=x.device).item())
        if s >= T:
            return x

        z = torch.zeros_like(x)

        # shift left (future -> present): z[t] = x[t+s] for t < T-s
        # part 1: [0:c1]
        if c1 > 0:
            z[:T - s, :, :c1, :, :] = x[s:, :, :c1, :, :]

        # shift right (past -> present): z[t] = x[t-s] for t >= s
        # part 2: [c1:c2]
        if c2 > c1:
            z[s:, :, c1:c2, :, :] = x[:T - s, :, c1:c2, :, :]

        # unchanged remainder: [c2:C_used]
        if C_used > c2:
            z[:, :, c2:C_used, :, :] = x[:, :, c2:C_used, :, :]

        # leftover channels (if C not divisible by groups) remain unchanged
        if C_used < C:
            z[:, :, C_used:, :, :] = x[:, :, C_used:, :, :]

        alpha = torch.tensor(self.alpha, device=x.device, dtype=x.dtype)
        return x + alpha * z
