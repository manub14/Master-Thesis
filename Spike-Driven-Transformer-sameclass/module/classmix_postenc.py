import torch
import torch.nn as nn


class ClassMixPostEncoder(nn.Module):
    """
    Post-encoding augmentation for Spike-Driven Transformer.

    Works on spike feature maps x of shape [T, B, C, H, W] and
    uses labels y of shape [B].

    Idea (your supervisor's):
    - If the whole batch has the same class -> "mix across batches":
        mix across images in the batch (batch dim).
    - Else -> "mix inside":
        mix across time steps (time dim) within each image.

    Mixing is done via elementwise selection between original and "partner"
    spikes, so outputs remain binary (0/1).
    """

    def __init__(self, p: float = 1.0):
        """
        Args:
            p: probability to apply the augmentation to a batch.
        """
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: spike features [T, B, C, H, W]
            y: labels [B] (long)

        Returns:
            Augmented x with same shape.
        """
        # No aug in eval mode
        if not self.training:
            return x

        # Stochastic on/off
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        T, B, C, H, W = x.shape
        device = x.device
        y = y.view(-1)

        # Edge case: batch of size 1 -> nothing to mix
        if B <= 1:
            return x

        # -------- Case 1: "batch has same class" -> mix across images in batch --------
        # Here I interpret your rule literally: all labels equal.
        if (y == y[0]).all():
            # Partner indices for each sample in batch
            perm = torch.randperm(B, device=device)

            # Avoid trivial identity permutation
            if (perm == torch.arange(B, device=device)).all():
                perm = torch.roll(perm, 1)

            partner = x[:, perm]  # [T, B, C, H, W]

            # Binary mask to choose between original and partner spikes
            # Shape [T, B, 1, H, W] -> broadcast over C
            mask = (torch.rand(T, B, 1, H, W, device=device) < 0.5)

            x = torch.where(mask, partner, x)

        # -------- Case 2: mixed labels -> mix inside each image across time steps --------
        else:
            # Permute time dimension
            perm_t = torch.randperm(T, device=device)
            partner = x[perm_t]  # [T, B, C, H, W]

            mask = (torch.rand(T, B, 1, H, W, device=device) < 0.5)
            x = torch.where(mask, partner, x)

        return x
