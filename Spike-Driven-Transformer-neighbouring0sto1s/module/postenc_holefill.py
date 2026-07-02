# module/postenc_holefill.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class HoleFillPostEncoding2D(nn.Module):
    """
    Spatial post-encoding spike augmentation.

    Input:  x (T, B, C, H, W) with binary spikes (0/1).
    For each time step independently:
      If a location is 0 and all 8 neighbours (up, down, left, right,
      and 4 diagonals) are 1, flip it to 1.

    No-op in eval mode.
    """

    def __init__(self, prob: float = 1.0):
        super().__init__()
        self.prob = prob

        # 3x3 kernel with zero center ? counts the 8 neighbours
        kernel = torch.ones(1, 1, 3, 3)
        kernel[0, 0, 1, 1] = 0.0  # center = 0
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W)
        if (not self.training) or self.prob <= 0.0:
            return x

        # stochastic apply per forward (optional)
        if self.prob < 1.0 and torch.rand(1, device=x.device) > self.prob:
            return x

        if x.dim() != 5:
            raise ValueError(f"Expected (T, B, C, H, W), got {x.shape}")

        T, B, C, H, W = x.shape

        # treat each (T,B,C) map independently, conv over H,W
        x_flat = x.reshape(T * B * C, 1, H, W)        # (N, 1, H, W)
        x_float = x_flat.float()

        # sum of 8 neighbours (center weight is 0)
        neigh_sum = F.conv2d(x_float, self.kernel, padding=1)

        # "hole": center is 0 and all 8 neighbours are 1 ? sum = 8
        hole_mask = (x_flat == 0) & (neigh_sum == 8)

        x_flat_filled = torch.where(hole_mask, torch.ones_like(x_flat), x_flat)

        x_filled = x_flat_filled.reshape(T, B, C, H, W)
        return x_filled


class HoleFillPostEncoding3D(nn.Module):
    """
    Spatio-temporal post-encoding spike augmentation.

    Input:  x (T, B, C, H, W) with binary spikes (0/1).
    Treat spikes as a 3D volume over (T, H, W) per (B,C).
    For each voxel (t,h,w):
      If it is 0 and all its 26 neighbours in 3x3x3 (excluding itself)
      are 1, flip it to 1.

    No-op in eval mode.
    """

    def __init__(self, prob: float = 1.0):
        super().__init__()
        self.prob = prob

        # 3x3x3 kernel with zero center ? counts the 26 neighbours
        kernel3d = torch.ones(1, 1, 3, 3, 3)
        kernel3d[0, 0, 1, 1, 1] = 0.0  # center = 0
        self.register_buffer("kernel3d", kernel3d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W)
        if (not self.training) or self.prob <= 0.0:
            return x

        if self.prob < 1.0 and torch.rand(1, device=x.device) > self.prob:
            return x

        if x.dim() != 5:
            raise ValueError(f"Expected (T, B, C, H, W), got {x.shape}")

        T, B, C, H, W = x.shape

        # reshape to (N, 1, D, H, W) for conv3d, where D = T
        x_perm = x.permute(1, 2, 0, 3, 4)             # (B, C, T, H, W)
        x_flat = x_perm.reshape(B * C, 1, T, H, W)    # (N, 1, T, H, W)
        x_float = x_flat.float()

        # sum of 26 neighbours (3x3x3 minus center)
        neigh_sum = F.conv3d(x_float, self.kernel3d, padding=1)

        # "3D hole": center is 0, all 26 neighbours are 1 ? sum = 26
        hole_mask = (x_flat == 0) & (neigh_sum == 26)

        x_flat_filled = torch.where(hole_mask, torch.ones_like(x_flat), x_flat)

        x_perm_filled = x_flat_filled.reshape(B, C, T, H, W)
        x_filled = x_perm_filled.permute(2, 0, 1, 3, 4)  # back to (T, B, C, H, W)

        return x_filled
