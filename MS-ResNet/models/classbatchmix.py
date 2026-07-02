# models/classbatchmix.py
import torch
import torch.nn as nn

class ClassBatchMixPostEncoding(nn.Module):
    """
    Post-encoding augmentation:

    For each sample i:
      - if there exists another sample j in the same mini-batch with the same class:
            x_i <- lam*x_i + (1-lam)*x_j
      - else:
            "mix inside" the sample:
               - if spikes with time dim: mix with time-shuffled version
               - else: mix with patch-shuffled version

    Supports x shapes:
      - [B, C, H, W]
      - [B, T, C, H, W]
      - [T, B, C, H, W]  (common in SNN code)
    """
    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 0.4,
        intra: str = "time",          # "time" | "patch" | "both"
        patch_size: int = 4,
        binarize: bool = False,       # if you really want spikes back to {0,1}
        clamp01: bool = True,
    ):
        super().__init__()
        assert 0.0 <= p <= 1.0
        assert alpha > 0.0
        assert intra in ["time", "patch", "both"]
        assert patch_size >= 1

        self.p = p
        self.alpha = alpha
        self.intra = intra
        self.patch_size = patch_size
        self.binarize = binarize
        self.clamp01 = clamp01

    # -----------------------
    # Utilities: shape handling
    # -----------------------
    def _to_B_first(self, x, y):
        """
        Returns (xB, restore_fn)
        where xB has batch dimension first.
        """
        if x.dim() == 4:
            # [B,C,H,W]
            return x, (lambda z: z)

        if x.dim() != 5:
            raise ValueError(f"Unsupported x.dim()={x.dim()}; expected 4 or 5.")

        B = y.shape[0] if y is not None else None

        # Try identify batch dim by matching size to len(y)
        if B is not None:
            if x.shape[0] == B:
                # [B,T,C,H,W]
                return x, (lambda z: z)
            if x.shape[1] == B:
                # [T,B,C,H,W] -> [B,T,C,H,W]
                return x.permute(1, 0, 2, 3, 4).contiguous(), (lambda z: z.permute(1, 0, 2, 3, 4).contiguous())

        # If y is None, assume [B,T,C,H,W]
        return x, (lambda z: z)

    def _sample_lam(self, n, device, dtype):
        # lam ~ Beta(alpha, alpha), shape [n,1,1,1,(1)]
        dist = torch.distributions.Beta(self.alpha, self.alpha)
        lam = dist.sample((n,)).to(device=device, dtype=dtype)
        return lam

    # -----------------------
    # Intra-sample partner creation
    # -----------------------
    def _time_shuffle_partner(self, xB):
        # xB: [B,T,C,H,W]
        B, T = xB.shape[0], xB.shape[1]
        perm = torch.randperm(T, device=xB.device)
        return xB[:, perm, ...]  # time-shuffled

    def _patch_shuffle_partner_4d(self, xB):
        # xB: [B,C,H,W]
        B, C, H, W = xB.shape
        ps = self.patch_size
        if H < ps or W < ps:
            # too small -> fallback to channel shuffle
            perm_c = torch.randperm(C, device=xB.device)
            return xB[:, perm_c, :, :]

        # split into patches and shuffle patch order
        hN = H // ps
        wN = W // ps
        Hc = hN * ps
        Wc = wN * ps
        x = xB[:, :, :Hc, :Wc]

        # [B,C,hN,ps,wN,ps] -> [B,C,hN,wN,ps,ps]
        x = x.view(B, C, hN, ps, wN, ps).permute(0, 1, 2, 4, 3, 5).contiguous()
        # flatten patches: [B,C,hN*wN,ps,ps]
        x = x.view(B, C, hN * wN, ps, ps)

        perm = torch.randperm(hN * wN, device=xB.device)
        x = x[:, :, perm, :, :]

        # restore
        x = x.view(B, C, hN, wN, ps, ps).permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, Hc, Wc)

        out = xB.clone()
        out[:, :, :Hc, :Wc] = x
        return out

    def _patch_shuffle_partner_5d(self, xB):
        # xB: [B,T,C,H,W] -> shuffle patches per time frame
        B, T, C, H, W = xB.shape
        flat = xB.view(B * T, C, H, W)
        shuf = self._patch_shuffle_partner_4d(flat)
        return shuf.view(B, T, C, H, W)

    def _intra_partner(self, xB):
        # xB may be 4d or 5d (B-first)
        if xB.dim() == 5:
            if self.intra == "time":
                return self._time_shuffle_partner(xB)
            if self.intra == "patch":
                return self._patch_shuffle_partner_5d(xB)
            # both: time-shuffle then patch-shuffle
            return self._patch_shuffle_partner_5d(self._time_shuffle_partner(xB))

        # 4d
        if self.intra in ["patch", "both"]:
            return self._patch_shuffle_partner_4d(xB)
        # "time" requested but no time dim -> fallback to patch shuffle anyway
        return self._patch_shuffle_partner_4d(xB)

    # -----------------------
    # Main
    # -----------------------
    def forward(self, x, y=None):
        if (not self.training) or (self.p == 0.0):
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        if y is None:
            # no labels -> only intra-sample mixing
            xB, restore = self._to_B_first(x, None)
            partner = self._intra_partner(xB)
            lam = self._sample_lam(xB.shape[0], xB.device, xB.dtype)
            while lam.dim() < xB.dim():
                lam = lam.view(-1, *([1] * (xB.dim() - 1)))
            out = lam * xB + (1.0 - lam) * partner
            if self.clamp01:
                out = out.clamp(0.0, 1.0)
            if self.binarize:
                out = torch.bernoulli(out)
            return restore(out)

        # labels provided
        if not torch.is_tensor(y):
            y = torch.tensor(y, device=x.device)
        else:
            y = y.to(device=x.device)

        xB, restore = self._to_B_first(x, y)

        B = y.shape[0]
        if xB.shape[0] != B:
            # safety fallback
            return restore(xB)

        out = xB.clone()
        used = torch.zeros(B, dtype=torch.bool, device=x.device)

        # group indices by class
        classes = torch.unique(y)
        for c in classes:
            idx = torch.nonzero(y == c, as_tuple=False).squeeze(1)
            n = idx.numel()
            if n <= 1:
                continue

            # pair within class: permute and avoid self-pair by rolling if needed
            perm = idx[torch.randperm(n, device=x.device)]
            if torch.all(perm == idx):
                perm = torch.roll(perm, shifts=1, dims=0)

            xi = xB[idx]
            xj = xB[perm]

            lam = self._sample_lam(n, xB.device, xB.dtype)
            while lam.dim() < xi.dim():
                lam = lam.view(-1, *([1] * (xi.dim() - 1)))

            mixed = lam * xi + (1.0 - lam) * xj
            out[idx] = mixed
            used[idx] = True

        # for samples with no same-class partner: intra-sample mix
        leftover = torch.nonzero(~used, as_tuple=False).squeeze(1)
        if leftover.numel() > 0:
            xl = xB[leftover]
            partner = self._intra_partner(xl)
            lam = self._sample_lam(leftover.numel(), xB.device, xB.dtype)
            while lam.dim() < xl.dim():
                lam = lam.view(-1, *([1] * (xl.dim() - 1)))
            out[leftover] = lam * xl + (1.0 - lam) * partner

        if self.clamp01:
            out = out.clamp(0.0, 1.0)
        if self.binarize:
            out = torch.bernoulli(out)

        return restore(out)