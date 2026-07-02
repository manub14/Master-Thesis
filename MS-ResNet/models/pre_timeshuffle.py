import torch
import torch.nn as nn


class PreTimeShuffle(nn.Module):
    """
    TS-SNN-inspired pre-encoding temporal shift for MS-ResNet.

    Expected input shape:
        [T, B, C, H, W]

    Recommended insertion point:
        right after conv1/stem output, before the first residual stage.

    Behavior:
      - with probability p, sample two random split points g1 < g2 over channel groups
      - shift first part left in time
      - shift second part right in time
      - keep remaining part unchanged
      - zero-pad emptied borders
      - return residual blend: x + alpha * shifted
    """

    def __init__(
        self,
        p: float = 0.0,
        max_shift: int = 1,
        foldk: int = 32,
        alpha: float = 0.3,
        apply_in_eval: bool = False,
    ):
        super().__init__()
        self.p = float(p)
        self.max_shift = int(max_shift)
        self.foldk = int(foldk)
        self.alpha = float(alpha)
        self.apply_in_eval = bool(apply_in_eval)

    def extra_repr(self) -> str:
        return (
            f"p={self.p}, max_shift={self.max_shift}, "
            f"foldk={self.foldk}, alpha={self.alpha}, "
            f"apply_in_eval={self.apply_in_eval}"
        )

    @staticmethod
    def _sample_split_points(num_groups: int, device: torch.device):
        # Need 3 segments: [0:g1), [g1:g2), [g2:end)
        # So choose two distinct points from 1..num_groups-1
        idx = torch.randperm(num_groups - 1, device=device)[:2] + 1
        g1 = int(torch.min(idx).item())
        g2 = int(torch.max(idx).item())
        return g1, g2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None or x.dim() != 5:
            return x

        if self.p <= 0.0:
            return x

        if (not self.training) and (not self.apply_in_eval):
            return x

        if torch.rand(1, device=x.device).item() > self.p:
            return x

        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape

        if T < 2 or C < 3:
            return x

        # Effective channel folding factor
        foldk = min(self.foldk, C)
        if foldk < 3:
            return x

        # Use fixed-size channel groups; leftover tail stays in the final "no-shift" chunk
        cfold = max(1, C // foldk)
        num_groups = C // cfold
        if num_groups < 3:
            return x

        g1, g2 = self._sample_split_points(num_groups, x.device)

        c1 = min(g1 * cfold, C)
        c2 = min(g2 * cfold, C)

        max_valid_shift = min(self.max_shift, T - 1)
        if max_valid_shift < 1:
            return x

        shift = 1
        if max_valid_shift > 1:
            shift = int(
                torch.randint(
                    low=1,
                    high=max_valid_shift + 1,
                    size=(1,),
                    device=x.device,
                ).item()
            )

        z = torch.zeros_like(x)

        # Left shift: future -> current for the first chunk
        # z[t] = x[t + shift]
        z[:-shift, :, :c1, :, :] = x[shift:, :, :c1, :, :]

        # Right shift: past -> current for the second chunk
        # z[t] = x[t - shift]
        z[shift:, :, c1:c2, :, :] = x[:-shift, :, c1:c2, :, :]

        # No shift for the remaining channels
        z[:, :, c2:, :, :] = x[:, :, c2:, :, :]

        # Residual blend, following the TS-SNN idea
        out = x + self.alpha * z
        return out