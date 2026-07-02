# models/patchshuffle2d.py
import torch
import torch.nn as nn


class PatchShufflePostEncoding2D(nn.Module):
    """
    Post-encoding PatchShuffle for spike tensors.

    Supports:
      - x: [T,B,C,H,W] when layout="TB"
      - x: [B,T,C,H,W] when layout="BT"
      - x: [B,C,H,W]   (no time dim)

    Args:
      p: probability to apply patch shuffle (0 disables)
      patch_size: spatial patch size (must divide H and W)
      layout: "TB" or "BT" for 5D inputs
      same_on_time: if True, use the SAME permutation across all timesteps (for each sample)
      same_on_batch: if True, use the SAME permutation across the whole batch (and time if same_on_time)
    """
    def __init__(
        self,
        p: float = 0.0,
        patch_size: int = 2,
        layout: str = "TB",
        same_on_time: bool = True,
        same_on_batch: bool = False,
        **kwargs,  # swallow extra args safely (prevents future clashes)
    ):
        super().__init__()
        self.p = float(p)
        self.patch_size = int(patch_size)
        self.layout = str(layout).upper()
        self.same_on_time = bool(same_on_time)
        self.same_on_batch = bool(same_on_batch)

        if self.layout not in ("TB", "BT"):
            raise ValueError(f"layout must be 'TB' or 'BT', got {self.layout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # disable in eval or if p==0
        if (not self.training) or self.p <= 0.0:
            return x

        if x.dim() == 4:
            # [B,C,H,W]
            return self._shuffle_4d(x)

        if x.dim() != 5:
            return x

        if self.layout == "TB":
            # [T,B,C,H,W] -> [B,T,C,H,W]
            x_bt = x.permute(1, 0, 2, 3, 4).contiguous()
            x_bt = self._shuffle_5d_bt(x_bt)
            return x_bt.permute(1, 0, 2, 3, 4).contiguous()

        # layout == "BT": [B,T,C,H,W]
        return self._shuffle_5d_bt(x)

    # -------------------------
    # Internals
    # -------------------------

    def _shuffle_5d_bt(self, x_bt: torch.Tensor) -> torch.Tensor:
        # x_bt: [B,T,C,H,W]
        B, T, C, H, W = x_bt.shape
        ps = self.patch_size

        if ps <= 1:
            return x_bt
        if (H % ps != 0) or (W % ps != 0):
            # can't patchify cleanly -> no-op
            return x_bt

        # Decide whether to apply shuffling
        if self.same_on_batch:
            # one Bernoulli for entire batch (either apply or skip)
            if torch.rand((), device=x_bt.device) >= self.p:
                return x_bt
            apply_mask = None  # apply to all
        else:
            # per-sample Bernoulli
            apply_mask = (torch.rand((B,), device=x_bt.device) < self.p)

            # if nobody selected, no-op
            if not bool(apply_mask.any()):
                return x_bt

        # If same_on_time: one permutation per sample (or one for all if same_on_batch)
        if self.same_on_time:
            # flatten time into batch for efficient gather after we build per-sample perms
            x_flat = x_bt.view(B * T, C, H, W)

            if self.same_on_batch:
                # one perm for all samples and all timesteps
                x_shuf = self._shuffle_4d_with_perm(x_flat, perm=None, per_sample=False)
                return x_shuf.view(B, T, C, H, W)

            # per-sample permutation, reused across T
            # build perms for B samples, then expand to B*T
            perms = self._make_perms(B, H, W, ps, device=x_bt.device)  # [B, nP]
            perms_bt = perms.repeat_interleave(T, dim=0)  # [B*T, nP]

            x_shuf = self._shuffle_4d_with_perm(x_flat, perm=perms_bt, per_sample=True)
            x_shuf = x_shuf.view(B, T, C, H, W)

            # apply only on selected samples
            if apply_mask is not None:
                x_out = x_bt.clone()
                x_out[apply_mask] = x_shuf[apply_mask]
                return x_out
            return x_shuf

        # else: per-time permutations (potentially different at each t)
        # treat each (b,t) as its own item
        N = B * T
        x_flat = x_bt.view(N, C, H, W)

        if self.same_on_batch:
            # one perm for all (b,t)
            x_shuf = self._shuffle_4d_with_perm(x_flat, perm=None, per_sample=False)
            return x_shuf.view(B, T, C, H, W)

        perms = self._make_perms(N, H, W, ps, device=x_bt.device)  # [B*T, nP]
        x_shuf = self._shuffle_4d_with_perm(x_flat, perm=perms, per_sample=True)
        x_shuf = x_shuf.view(B, T, C, H, W)

        # apply per-sample mask: if sample b not selected, keep all its timesteps
        if apply_mask is not None:
            x_out = x_bt.clone()
            x_out[apply_mask] = x_shuf[apply_mask]
            return x_out

        return x_shuf

    def _shuffle_4d(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        ps = self.patch_size

        if ps <= 1:
            return x
        if (H % ps != 0) or (W % ps != 0):
            return x

        if self.same_on_batch:
            if torch.rand((), device=x.device) >= self.p:
                return x
            # one perm for all samples
            return self._shuffle_4d_with_perm(x, perm=None, per_sample=False)

        # per-sample
        apply_mask = (torch.rand((B,), device=x.device) < self.p)
        if not bool(apply_mask.any()):
            return x

        perms = self._make_perms(B, H, W, ps, device=x.device)  # [B,nP]
        x_shuf = self._shuffle_4d_with_perm(x, perm=perms, per_sample=True)

        out = x.clone()
        out[apply_mask] = x_shuf[apply_mask]
        return out

    def _make_perms(self, N: int, H: int, W: int, ps: int, device) -> torch.Tensor:
        nH = H // ps
        nW = W // ps
        nP = nH * nW
        # random scores -> argsort gives random permutation per row
        scores = torch.rand((N, nP), device=device)
        return scores.argsort(dim=1)

    def _shuffle_4d_with_perm(self, x: torch.Tensor, perm: torch.Tensor = None, per_sample: bool = True) -> torch.Tensor:
        """
        x: [N,C,H,W]
        perm:
          - None => single random perm for all (per_sample=False)
          - [N,nP] => per item perm (per_sample=True)
        """
        N, C, H, W = x.shape
        ps = self.patch_size
        nH = H // ps
        nW = W // ps
        nP = nH * nW

        # patchify -> [N,C,nP,ps,ps]
        patches = (
            x.view(N, C, nH, ps, nW, ps)
             .permute(0, 1, 2, 4, 3, 5)
             .contiguous()
             .view(N, C, nP, ps, ps)
        )

        if perm is None:
            # one perm for all
            idx = torch.randperm(nP, device=x.device)
            patches = patches[:, :, idx, :, :]
        else:
            # perm: [N,nP]
            idx = perm.view(N, 1, nP, 1, 1).expand(N, C, nP, ps, ps)
            patches = torch.gather(patches, dim=2, index=idx)

        # unpatchify
        out = (
            patches.view(N, C, nH, nW, ps, ps)
                  .permute(0, 1, 2, 4, 3, 5)
                  .contiguous()
                  .view(N, C, H, W)
        )
        return out
