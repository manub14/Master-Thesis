import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- NEW (for postenc_aug hook / external post-encoding augs like
# PatchMix, TimeMix, HoleFill, LocalTimeShuffle, etc.) ----
import weakref
from typing import Optional

from models.pre_timeshuffle import PreTimeShuffle
from models.pre_temporaljitter import PreTemporalJitter
from models.patchdropout2d import PostEncodingPatchDropout2D
from models.timeshuffle2d import PostEncodingTimeShuffle2D
from models.patchshuffle2d import PatchShufflePostEncoding2D
# TimeMask (post-encoding)
from models.timemask import TimeMaskPostEncoding

# ---- NEW: Pre-encoding TimeMask ----
from models.preenc_timemask import PreEncodingTimeMask

# ---- NEW: Pre-encoding Frequency Encoding ----
from models.frequency_encoding_aug import PreEncodingFrequencyEncoding

# ---- NEW: Pre-encoding FullDimMix ----
from models.preenc_fulldimmix import PreEncFullDimMix


# ----------------------------
# Globals (same as your file)
# ----------------------------
thresh = 0.5
lens = 0.5
decay = 0.25
time_window = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_apply_preenc_timeshuffle(module, x):
    aug = getattr(module, "preenc_timeshuffle", None)
    if aug is None:
        return x
    return aug(x)


def _maybe_apply_preenc_aug(module, x):
    """
    Generic pre-encoding augmentation hook.

    This allows train_amp.py to attach:
        core.preenc_aug = some_module_or_chain

    without clashing with the existing built-in pre-encoding methods.
    """
    aug = getattr(module, "preenc_aug", None)
    if aug is None:
        return x
    return aug(x)


def _module_tree_contains_type(module, type_names):
    """
    Check whether module or any of its children has class name in type_names.
    Useful to avoid applying temporal jitter twice when it is already attached
    through self.preenc_aug.
    """
    if module is None:
        return False

    if isinstance(type_names, str):
        type_names = {type_names}
    else:
        type_names = set(type_names)

    if module.__class__.__name__ in type_names:
        return True

    for child in module.children():
        if _module_tree_contains_type(child, type_names):
            return True

    return False


def _maybe_apply_preenc_temporaljitter(module, x):
    aug = getattr(module, "preenc_tjitter", None)
    if aug is None:
        return x
    return aug(x)


def _maybe_apply_preenc_fulldimmix(module, x):
    aug = getattr(module, "preenc_fulldimmix", None)
    if aug is None:
        return x
    return aug(x)


# ----------------------------
# Spike function + membrane update
# ----------------------------
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (abs(input - thresh) < lens)
        temp = temp / (2 * lens)
        return grad_input * temp.float()


act_fun = ActFun.apply


class mem_update(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(x.device)
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)

        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * decay * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


# ----------------------------
# TDBN
# ----------------------------
class BatchNorm3d1(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0)
            nn.init.zeros_(self.bias)


class batch_norm_2d(nn.Module):
    """TDBN"""
    def __init__(self, num_features):
        super().__init__()
        self.bn = BatchNorm3d1(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class batch_norm_2d1(nn.Module):
    """TDBN (zero init gamma)"""
    def __init__(self, num_features):
        super().__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


# ----------------------------
# SNN conv
# ----------------------------
class Snn_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', marker='b'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        weight = self.weight
        h = (input.size()[3] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        w = (input.size()[4] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device)
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        return c1


# ----------------------------
# NEW: Pre-encoding TimeMix (TS-SNN-style, additive)
# ----------------------------
class PreEncodingTimeMix(nn.Module):
    """
    TS-SNN-style temporal channel-group mixing for pre-encoding input
    of shape [T, B, C, H, W].

    Behavior:
      - split channels into groups
      - choose two split points g1 < g2
      - first part uses t+1
      - middle part uses t-1
      - last part stays unchanged
      - residual combine: out = x + alpha * shifted

    This is additive and does not interfere with other methods.
    """

    def __init__(self,
                 p=0.0,
                 alpha=0.3,
                 groups=32,
                 random_split=True,
                 apply_in_eval=False):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.groups = int(groups)
        self.random_split = bool(random_split)
        self.apply_in_eval = bool(apply_in_eval)

    def _choose_split_points(self, num_groups, device):
        if num_groups < 3:
            return None, None

        if self.random_split:
            candidates = torch.arange(1, num_groups, device=device)
            if candidates.numel() < 2:
                return None, None
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

    def forward(self, x):
        # x: [T, B, C, H, W]
        if x.dim() != 5:
            return x

        if self.p <= 0.0:
            return x

        if (not self.training) and (not self.apply_in_eval):
            return x

        if torch.rand(1, device=x.device).item() > self.p:
            return x

        T, B, C, H, W = x.shape

        if T < 2 or C < 3:
            return x

        group_count = min(self.groups, C)
        if group_count <= 0:
            return x

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

        # future -> current
        shifted[:-1, :, :c1, :, :] = x[1:, :, :c1, :, :]

        # past -> current
        shifted[1:, :, c1:c2, :, :] = x[:-1, :, c1:c2, :, :]

        # current stays
        shifted[:, :, c2:, :, :] = x[:, :, c2:, :, :]

        out = x + self.alpha * shifted
        return out


# ----------------------------
# Post-encoding augmentation wrapper (NO CLASH)
# ----------------------------
class PostEncAug2D(nn.Module):
    """
    Applies post-encoding augs on spike tensor [T,B,C,H,W]:
      1) TimeMask
      2) TimeShuffle
      3) PatchShuffle
      4) PatchDropout

    Order rationale:
      - mask time first,
      - then shuffle time,
      - then shuffle spatial patches,
      - then patch-drop.
    """
    def __init__(self,
                 patchdrop_keep=1.0,
                 patchdrop_size=4,
                 tshift_p=0.0,
                 tshift_max=1,
                 tshift_fold_k=32,
                 tshift_alpha=0.3,
                 # TimeMask args
                 tmask_p=0.0,
                 tmask_num=1,
                 tmask_max_frac=0.25,
                 tmask_min_len=1,
                 tmask_mode="zero",
                 tmask_noise_std=0.05,
                 tmask_layout="TB",
                 tmask_same_on_batch=False,
                 tmask_per_channel=False,
                 tmask_channel_groups=1,
                 # PatchShuffle args
                 pshuf_p=0.0,
                 pshuf_size=4,
                 pshuf_layout="TB",
                 pshuf_per_time=False,
                 pshuf_same_on_batch=False):
        super().__init__()

        # TimeMask (post-encoding)
        self.tmask = TimeMaskPostEncoding(
            p=tmask_p,
            num_masks=tmask_num,
            max_mask_frac=tmask_max_frac,
            min_mask_len=tmask_min_len,
            mode=tmask_mode,
            noise_std=tmask_noise_std,
            layout=tmask_layout,              # MS-ResNet uses [T,B,...] so TB
            same_on_batch=tmask_same_on_batch,
            per_channel=tmask_per_channel,
            channel_groups=tmask_channel_groups,
        )

        # TimeShuffle
        self.tshift = PostEncodingTimeShuffle2D(
            p=tshift_p,
            max_shift=tshift_max,
            fold_k=tshift_fold_k,
            alpha=tshift_alpha
        )

        # PatchShuffle (spatial)
        self.pshuf = PatchShufflePostEncoding2D(
            p=pshuf_p,
            patch_size=pshuf_size,
            layout=pshuf_layout,
            same_on_time=(not pshuf_per_time),
            same_on_batch=pshuf_same_on_batch,
        )

        # PatchDropout (spatial drop)
        self.patchdrop = PostEncodingPatchDropout2D(
            keep_rate=patchdrop_keep,
            patch_size=patchdrop_size
        )

    def forward(self, x):
        # TimeMask
        x = self.tmask(x)

        # TimeShuffle
        x = self.tshift(x)

        # PatchShuffle
        x = self.pshuf(x)

        # PatchDropout (safe for 5D)
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            xb = x.reshape(T * B, C, H, W)
            xb = self.patchdrop(xb)
            x = xb.reshape(T, B, C, H, W)
        else:
            x = self.patchdrop(x)

        return x


# ----------------------------
# NEW: Hook wrapper to use train_amp.py-attached parent.postenc_aug
# ----------------------------
class PostEncApply(nn.Module):
    """
    If parent.postenc_aug is set (e.g., PatchMix / TimeMix / HoleFill /
    LocalTimeShuffle from train_amp.py), apply it.
    Otherwise, fall back to the built-in PostEncAug2D pipeline.

    We use weakref so the parent isn't registered as a submodule (no state_dict clutter).
    """
    def __init__(self, parent: Optional[nn.Module], fallback: nn.Module):
        super().__init__()
        self._parent_ref = weakref.ref(parent) if parent is not None else None
        self.fallback = fallback  # registered submodule

    def forward(self, x):
        parent = self._parent_ref() if self._parent_ref is not None else None
        aug = getattr(parent, "postenc_aug", None) if parent is not None else None
        if aug is not None:
            return aug(x)
        return self.fallback(x)


# ============================================================
# ImageNet-ish MS-ResNet blocks (18/34) with post-encoding aug
# ============================================================
class BasicBlock_18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,
                 patchdrop_keep=1.0, patchdrop_size=4,
                 tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
                 # TimeMask args
                 tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
                 tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
                 tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
                 # PatchShuffle args
                 pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
                 pshuf_per_time=False, pshuf_same_on_batch=False,
                 # NEW
                 parent: Optional[nn.Module] = None):
        super().__init__()

        # fallback pipeline (your existing composed augs)
        self.postenc_fallback = PostEncAug2D(
            patchdrop_keep=patchdrop_keep,
            patchdrop_size=patchdrop_size,
            tshift_p=tshift_p,
            tshift_max=tshift_max,
            tshift_fold_k=tshift_fold_k,
            tshift_alpha=tshift_alpha,
            # TimeMask
            tmask_p=tmask_p,
            tmask_num=tmask_num,
            tmask_max_frac=tmask_max_frac,
            tmask_min_len=tmask_min_len,
            tmask_mode=tmask_mode,
            tmask_noise_std=tmask_noise_std,
            tmask_layout=tmask_layout,
            tmask_same_on_batch=tmask_same_on_batch,
            tmask_per_channel=tmask_per_channel,
            tmask_channel_groups=tmask_channel_groups,
            # PatchShuffle
            pshuf_p=pshuf_p,
            pshuf_size=pshuf_size,
            pshuf_layout=pshuf_layout,
            pshuf_per_time=pshuf_per_time,
            pshuf_same_on_batch=pshuf_same_on_batch,
        )

        # NEW: if parent.postenc_aug is set (from train_amp), use it
        self.postenc = PostEncApply(parent, self.postenc_fallback)

        self.residual_function = nn.Sequential(
            mem_update(),
            self.postenc,  # <-- POST-ENCODING HERE (after spikes)
            Snn_Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            batch_norm_2d(out_channels),
            mem_update(),
            Snn_Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            batch_norm_2d1(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class ResNet_origin_18(nn.Module):
    def __init__(self, block, num_block, num_classes=1000, cifar_stem=False,
                 patchdrop_keep=1.0, patchdrop_size=4,
                 tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
                 # TimeMask args
                 tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
                 tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
                 tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
                 # PatchShuffle args
                 pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
                 pshuf_per_time=False, pshuf_same_on_batch=False,
                 # NEW: Pre-encoding TimeMask args
                 pre_tmask_enable=False,
                 pre_tmask_p=0.0,
                 pre_tmask_num=1,
                 pre_tmask_max_frac=0.25,
                 pre_tmask_min_len=1,
                 pre_tmask_mode="zero",
                 pre_tmask_noise_std=0.05,
                 pre_tmask_layout="TBCHW",
                 pre_tmask_same_on_batch=False,
                 pre_tmask_per_channel=False,
                 pre_tmask_channel_groups=1,
                 pre_tmask_apply_in_eval=False,
                 # NEW: Pre-encoding TimeShuffle args
                 pre_tshift_enable=False,
                 pre_tshift_p=0.0,
                 pre_tshift_max=1,
                 pre_tshift_foldk=32,
                 pre_tshift_alpha=0.3,
                 pre_tshift_apply_in_eval=False,
                 # NEW: Pre-encoding Temporal Jitter args
                 pre_tjitter_enable=False,
                 pre_tjitter_p=0.0,
                 pre_tjitter_max=1,
                 pre_tjitter_per_sample=False,
                 pre_tjitter_layout="TBCHW",
                 pre_tjitter_apply_in_eval=False,
                 # NEW: Pre-encoding TimeMix args
                 pre_tmix_enable=False,
                 pre_tmix_p=0.0,
                 pre_tmix_alpha=0.3,
                 pre_tmix_groups=32,
                 pre_tmix_random_split=True,
                 pre_tmix_apply_in_eval=False,
                 # NEW: Pre-encoding FullDimMix args
                 pre_fdmix_enable=False,
                 pre_fdmix_p=0.0,
                 pre_fdmix_alpha=0.5,
                 pre_fdmix_layout="TBCHW",
                 pre_fdmix_apply_in_eval=False,
                 # NEW: Pre-encoding Frequency Encoding args
                 pre_fe_enable=False,
                 pre_fe_p=1.0,
                 pre_fe_jitter=0,
                 pre_fe_radii=None,
                 pre_fe_apply_in_eval=False):
        super().__init__()

        self.time_window = time_window
        self.T = time_window

        # NEW: train_amp.py will set this (PatchMix / TimeMix / HoleFill /
        # LocalTimeShuffle / etc.)
        self.postenc_aug = None

        # NEW: generic pre-encoding hook set from train_amp.py
        self.preenc_aug = None

        # NEW: Pre-encoding Frequency Encoding (inside model, before direct repeat)
        self.preenc_fe = None
        if pre_fe_enable and float(pre_fe_p) > 0.0:
            self.preenc_fe = PreEncodingFrequencyEncoding(
                T=time_window,
                radii=pre_fe_radii,
                p=pre_fe_p,
                jitter=pre_fe_jitter,
                apply_in_eval=pre_fe_apply_in_eval,
            )

        # NEW: Pre-encoding TimeMask (inside model, before conv1)
        self.preenc_tmask = None
        if pre_tmask_enable and float(pre_tmask_p) > 0.0:
            self.preenc_tmask = PreEncodingTimeMask(
                p=pre_tmask_p,
                num_masks=pre_tmask_num,
                max_mask_frac=pre_tmask_max_frac,
                min_mask_len=pre_tmask_min_len,
                mode=pre_tmask_mode,
                noise_std=pre_tmask_noise_std,
                layout=pre_tmask_layout,
                same_on_batch=pre_tmask_same_on_batch,
                per_channel=pre_tmask_per_channel,
                channel_groups=pre_tmask_channel_groups,
                apply_in_eval=pre_tmask_apply_in_eval,
            )

        # NEW: Pre-encoding TimeShuffle (inside model, before conv1)
        self.preenc_timeshuffle = None
        if pre_tshift_enable and float(pre_tshift_p) > 0.0:
            self.preenc_timeshuffle = PreTimeShuffle(
                p=pre_tshift_p,
                max_shift=pre_tshift_max,
                per_sample=False,
                layout="TBCHW",
            )

        # NEW: Pre-encoding Temporal Jitter (native model support)
        self.preenc_tjitter = None
        if pre_tjitter_enable and float(pre_tjitter_p) > 0.0:
            try:
                self.preenc_tjitter = PreTemporalJitter(
                    p=pre_tjitter_p,
                    max_shift=pre_tjitter_max,
                    per_sample=pre_tjitter_per_sample,
                    layout=pre_tjitter_layout,
                    apply_in_eval=pre_tjitter_apply_in_eval,
                )
            except TypeError:
                self.preenc_tjitter = PreTemporalJitter(
                    p=pre_tjitter_p,
                    max_shift=pre_tjitter_max,
                    per_sample=pre_tjitter_per_sample,
                    layout=pre_tjitter_layout,
                )

        # NEW: Pre-encoding TimeMix (inside model, before conv1)
        self.preenc_tmix = None
        if pre_tmix_enable and float(pre_tmix_p) > 0.0:
            self.preenc_tmix = PreEncodingTimeMix(
                p=pre_tmix_p,
                alpha=pre_tmix_alpha,
                groups=pre_tmix_groups,
                random_split=pre_tmix_random_split,
                apply_in_eval=pre_tmix_apply_in_eval,
            )

        # NEW: Pre-encoding FullDimMix (inside model, after 5D expansion, before conv1)
        self.preenc_fulldimmix = None
        if pre_fdmix_enable and float(pre_fdmix_p) > 0.0:
            self.preenc_fulldimmix = PreEncFullDimMix(
                p=pre_fdmix_p,
                alpha=pre_fdmix_alpha,
                layout=pre_fdmix_layout,
                apply_in_eval=pre_fdmix_apply_in_eval,
            )

        k = 1
        self.in_channels = 64 * k

        if cifar_stem:
            self.conv1 = nn.Sequential(
                Snn_Conv2d(3, 64 * k, kernel_size=3, padding=1, bias=False, stride=1),
                batch_norm_2d(64 * k),
            )
            self.pool = nn.Identity()
        else:
            self.conv1 = nn.Sequential(
                Snn_Conv2d(3, 64 * k, kernel_size=7, padding=3, bias=False, stride=2),
                batch_norm_2d(64 * k),
            )
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(block, 64 * k, num_block[0], 2,
                                        patchdrop_keep, patchdrop_size,
                                        tshift_p, tshift_max, tshift_fold_k, tshift_alpha,
                                        tmask_p, tmask_num, tmask_max_frac, tmask_min_len,
                                        tmask_mode, tmask_noise_std, tmask_layout,
                                        tmask_same_on_batch, tmask_per_channel, tmask_channel_groups,
                                        pshuf_p, pshuf_size, pshuf_layout, pshuf_per_time, pshuf_same_on_batch)
        self.conv3_x = self._make_layer(block, 128 * k, num_block[1], 2,
                                        patchdrop_keep, patchdrop_size,
                                        tshift_p, tshift_max, tshift_fold_k, tshift_alpha,
                                        tmask_p, tmask_num, tmask_max_frac, tmask_min_len,
                                        tmask_mode, tmask_noise_std, tmask_layout,
                                        tmask_same_on_batch, tmask_per_channel, tmask_channel_groups,
                                        pshuf_p, pshuf_size, pshuf_layout, pshuf_per_time, pshuf_same_on_batch)
        self.conv4_x = self._make_layer(block, 256 * k, num_block[2], 2,
                                        patchdrop_keep, patchdrop_size,
                                        tshift_p, tshift_max, tshift_fold_k, tshift_alpha,
                                        tmask_p, tmask_num, tmask_max_frac, tmask_min_len,
                                        tmask_mode, tmask_noise_std, tmask_layout,
                                        tmask_same_on_batch, tmask_per_channel, tmask_channel_groups,
                                        pshuf_p, pshuf_size, pshuf_layout, pshuf_per_time, pshuf_same_on_batch)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2,
                                        patchdrop_keep, patchdrop_size,
                                        tshift_p, tshift_max, tshift_fold_k, tshift_alpha,
                                        tmask_p, tmask_num, tmask_max_frac, tmask_min_len,
                                        tmask_mode, tmask_noise_std, tmask_layout,
                                        tmask_same_on_batch, tmask_per_channel, tmask_channel_groups,
                                        pshuf_p, pshuf_size, pshuf_layout, pshuf_per_time, pshuf_same_on_batch)

        self.mem_update = mem_update()
        self.fc = nn.Linear(512 * block.expansion * k, num_classes)

    def set_input_encoder(self, encoder):
        self.preenc_fe = encoder

    def _make_layer(self, block, out_channels, num_blocks, stride,
                    patchdrop_keep, patchdrop_size,
                    tshift_p, tshift_max, tshift_fold_k, tshift_alpha,
                    tmask_p, tmask_num, tmask_max_frac, tmask_min_len,
                    tmask_mode, tmask_noise_std, tmask_layout,
                    tmask_same_on_batch, tmask_per_channel, tmask_channel_groups,
                    pshuf_p, pshuf_size, pshuf_layout, pshuf_per_time, pshuf_same_on_batch):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s,
                                patchdrop_keep, patchdrop_size,
                                tshift_p, tshift_max, tshift_fold_k, tshift_alpha,
                                tmask_p, tmask_num, tmask_max_frac, tmask_min_len,
                                tmask_mode, tmask_noise_std, tmask_layout,
                                tmask_same_on_batch, tmask_per_channel, tmask_channel_groups,
                                pshuf_p, pshuf_size, pshuf_layout, pshuf_per_time, pshuf_same_on_batch,
                                parent=self))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # NEW: pre-encoding Frequency Encoding before direct repeat
        if self.preenc_fe is not None:
            input = self.preenc_fe(x)
        else:
            input = torch.zeros(time_window, x.size(0), 3, x.size(2), x.size(3), device=x.device)
            input[:] = x

        # NEW: pre-encoding FullDimMix right after 5D expansion
        input = _maybe_apply_preenc_fulldimmix(self, input)

        # NEW: generic pre-encoding augmentation hook from train_amp.py
        input = _maybe_apply_preenc_aug(self, input)

        # NEW: pre-encoding TimeMask before first conv
        if self.preenc_tmask is not None:
            input = self.preenc_tmask(input)

        # NEW: pre-encoding TimeShuffle before first conv
        input = _maybe_apply_preenc_timeshuffle(self, input)

        # NEW: pre-encoding Temporal Jitter before first conv
        # Skip native tjitter if it is already present inside self.preenc_aug
        if not _module_tree_contains_type(getattr(self, "preenc_aug", None), {"PreTimeShuffle", "PreTemporalJitter"}):
            input = _maybe_apply_preenc_temporaljitter(self, input)

        # NEW: pre-encoding TimeMix before first conv
        if self.preenc_tmix is not None:
            input = self.preenc_tmix(input)

        out = self.conv1(input)

        if not isinstance(self.pool, nn.Identity):
            out = torch.stack([self.pool(out[t]) for t in range(time_window)], dim=0)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)

        out = self.mem_update(out)
        out = F.adaptive_avg_pool3d(out, (None, 1, 1))
        out = out.view(out.size(0), out.size(1), -1)  # [T,B,C]
        out = out.sum(dim=0) / out.size(0)            # [B,C]
        out = self.fc(out)
        return out


def resnet18(num_classes=1000, cifar_stem=False,
             patchdrop_keep=1.0, patchdrop_size=4,
             tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
             tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
             tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
             tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
             pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
             pshuf_per_time=False, pshuf_same_on_batch=False,
             # NEW: Pre-encoding TimeMask args
             pre_tmask_enable=False,
             pre_tmask_p=0.0,
             pre_tmask_num=1,
             pre_tmask_max_frac=0.25,
             pre_tmask_min_len=1,
             pre_tmask_mode="zero",
             pre_tmask_noise_std=0.05,
             pre_tmask_layout="TBCHW",
             pre_tmask_same_on_batch=False,
             pre_tmask_per_channel=False,
             pre_tmask_channel_groups=1,
             pre_tmask_apply_in_eval=False,
             # NEW: Pre-encoding TimeShuffle args
             pre_tshift_enable=False,
             pre_tshift_p=0.0,
             pre_tshift_max=1,
             pre_tshift_foldk=32,
             pre_tshift_alpha=0.3,
             pre_tshift_apply_in_eval=False,
             # NEW: Pre-encoding Temporal Jitter args
             pre_tjitter_enable=False,
             pre_tjitter_p=0.0,
             pre_tjitter_max=1,
             pre_tjitter_per_sample=False,
             pre_tjitter_layout="TBCHW",
             pre_tjitter_apply_in_eval=False,
             # NEW: Pre-encoding TimeMix args
             pre_tmix_enable=False,
             pre_tmix_p=0.0,
             pre_tmix_alpha=0.3,
             pre_tmix_groups=32,
             pre_tmix_random_split=True,
             pre_tmix_apply_in_eval=False,
             # NEW: Pre-encoding FullDimMix args
             pre_fdmix_enable=False,
             pre_fdmix_p=0.0,
             pre_fdmix_alpha=0.5,
             pre_fdmix_layout="TBCHW",
             pre_fdmix_apply_in_eval=False,
             # NEW: Pre-encoding Frequency Encoding args
             pre_fe_enable=False,
             pre_fe_p=1.0,
             pre_fe_jitter=0,
             pre_fe_radii=None,
             pre_fe_apply_in_eval=False):
    return ResNet_origin_18(BasicBlock_18, [2, 2, 2, 2],
                            num_classes=num_classes, cifar_stem=cifar_stem,
                            patchdrop_keep=patchdrop_keep, patchdrop_size=patchdrop_size,
                            tshift_p=tshift_p, tshift_max=tshift_max,
                            tshift_fold_k=tshift_fold_k, tshift_alpha=tshift_alpha,
                            tmask_p=tmask_p, tmask_num=tmask_num, tmask_max_frac=tmask_max_frac, tmask_min_len=tmask_min_len,
                            tmask_mode=tmask_mode, tmask_noise_std=tmask_noise_std, tmask_layout=tmask_layout,
                            tmask_same_on_batch=tmask_same_on_batch, tmask_per_channel=tmask_per_channel, tmask_channel_groups=tmask_channel_groups,
                            pshuf_p=pshuf_p, pshuf_size=pshuf_size, pshuf_layout=pshuf_layout,
                            pshuf_per_time=pshuf_per_time, pshuf_same_on_batch=pshuf_same_on_batch,
                            pre_tmask_enable=pre_tmask_enable,
                            pre_tmask_p=pre_tmask_p,
                            pre_tmask_num=pre_tmask_num,
                            pre_tmask_max_frac=pre_tmask_max_frac,
                            pre_tmask_min_len=pre_tmask_min_len,
                            pre_tmask_mode=pre_tmask_mode,
                            pre_tmask_noise_std=pre_tmask_noise_std,
                            pre_tmask_layout=pre_tmask_layout,
                            pre_tmask_same_on_batch=pre_tmask_same_on_batch,
                            pre_tmask_per_channel=pre_tmask_per_channel,
                            pre_tmask_channel_groups=pre_tmask_channel_groups,
                            pre_tmask_apply_in_eval=pre_tmask_apply_in_eval,
                            pre_tshift_enable=pre_tshift_enable,
                            pre_tshift_p=pre_tshift_p,
                            pre_tshift_max=pre_tshift_max,
                            pre_tshift_foldk=pre_tshift_foldk,
                            pre_tshift_alpha=pre_tshift_alpha,
                            pre_tshift_apply_in_eval=pre_tshift_apply_in_eval,
                            pre_tjitter_enable=pre_tjitter_enable,
                            pre_tjitter_p=pre_tjitter_p,
                            pre_tjitter_max=pre_tjitter_max,
                            pre_tjitter_per_sample=pre_tjitter_per_sample,
                            pre_tjitter_layout=pre_tjitter_layout,
                            pre_tjitter_apply_in_eval=pre_tjitter_apply_in_eval,
                            pre_tmix_enable=pre_tmix_enable,
                            pre_tmix_p=pre_tmix_p,
                            pre_tmix_alpha=pre_tmix_alpha,
                            pre_tmix_groups=pre_tmix_groups,
                            pre_tmix_random_split=pre_tmix_random_split,
                            pre_tmix_apply_in_eval=pre_tmix_apply_in_eval,
                            pre_fdmix_enable=pre_fdmix_enable,
                            pre_fdmix_p=pre_fdmix_p,
                            pre_fdmix_alpha=pre_fdmix_alpha,
                            pre_fdmix_layout=pre_fdmix_layout,
                            pre_fdmix_apply_in_eval=pre_fdmix_apply_in_eval,
                            pre_fe_enable=pre_fe_enable,
                            pre_fe_p=pre_fe_p,
                            pre_fe_jitter=pre_fe_jitter,
                            pre_fe_radii=pre_fe_radii,
                            pre_fe_apply_in_eval=pre_fe_apply_in_eval)


def resnet34(num_classes=1000, cifar_stem=False,
             patchdrop_keep=1.0, patchdrop_size=4,
             tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
             tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
             tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
             tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
             pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
             pshuf_per_time=False, pshuf_same_on_batch=False,
             # NEW: Pre-encoding TimeMask args
             pre_tmask_enable=False,
             pre_tmask_p=0.0,
             pre_tmask_num=1,
             pre_tmask_max_frac=0.25,
             pre_tmask_min_len=1,
             pre_tmask_mode="zero",
             pre_tmask_noise_std=0.05,
             pre_tmask_layout="TBCHW",
             pre_tmask_same_on_batch=False,
             pre_tmask_per_channel=False,
             pre_tmask_channel_groups=1,
             pre_tmask_apply_in_eval=False,
             # NEW: Pre-encoding TimeShuffle args
             pre_tshift_enable=False,
             pre_tshift_p=0.0,
             pre_tshift_max=1,
             pre_tshift_foldk=32,
             pre_tshift_alpha=0.3,
             pre_tshift_apply_in_eval=False,
             # NEW: Pre-encoding Temporal Jitter args
             pre_tjitter_enable=False,
             pre_tjitter_p=0.0,
             pre_tjitter_max=1,
             pre_tjitter_per_sample=False,
             pre_tjitter_layout="TBCHW",
             pre_tjitter_apply_in_eval=False,
             # NEW: Pre-encoding TimeMix args
             pre_tmix_enable=False,
             pre_tmix_p=0.0,
             pre_tmix_alpha=0.3,
             pre_tmix_groups=32,
             pre_tmix_random_split=True,
             pre_tmix_apply_in_eval=False,
             # NEW: Pre-encoding FullDimMix args
             pre_fdmix_enable=False,
             pre_fdmix_p=0.0,
             pre_fdmix_alpha=0.5,
             pre_fdmix_layout="TBCHW",
             pre_fdmix_apply_in_eval=False,
             # NEW: Pre-encoding Frequency Encoding args
             pre_fe_enable=False,
             pre_fe_p=1.0,
             pre_fe_jitter=0,
             pre_fe_radii=None,
             pre_fe_apply_in_eval=False):
    return ResNet_origin_18(BasicBlock_18, [3, 4, 6, 3],
                            num_classes=num_classes, cifar_stem=cifar_stem,
                            patchdrop_keep=patchdrop_keep, patchdrop_size=patchdrop_size,
                            tshift_p=tshift_p, tshift_max=tshift_max,
                            tshift_fold_k=tshift_fold_k, tshift_alpha=tshift_alpha,
                            tmask_p=tmask_p, tmask_num=tmask_num, tmask_max_frac=tmask_max_frac, tmask_min_len=tmask_min_len,
                            tmask_mode=tmask_mode, tmask_noise_std=tmask_noise_std, tmask_layout=tmask_layout,
                            tmask_same_on_batch=tmask_same_on_batch, tmask_per_channel=tmask_per_channel, tmask_channel_groups=tmask_channel_groups,
                            pshuf_p=pshuf_p, pshuf_size=pshuf_size, pshuf_layout=pshuf_layout,
                            pshuf_per_time=pshuf_per_time, pshuf_same_on_batch=pshuf_same_on_batch,
                            pre_tmask_enable=pre_tmask_enable,
                            pre_tmask_p=pre_tmask_p,
                            pre_tmask_num=pre_tmask_num,
                            pre_tmask_max_frac=pre_tmask_max_frac,
                            pre_tmask_min_len=pre_tmask_min_len,
                            pre_tmask_mode=pre_tmask_mode,
                            pre_tmask_noise_std=pre_tmask_noise_std,
                            pre_tmask_layout=pre_tmask_layout,
                            pre_tmask_same_on_batch=pre_tmask_same_on_batch,
                            pre_tmask_per_channel=pre_tmask_per_channel,
                            pre_tmask_channel_groups=pre_tmask_channel_groups,
                            pre_tmask_apply_in_eval=pre_tmask_apply_in_eval,
                            pre_tshift_enable=pre_tshift_enable,
                            pre_tshift_p=pre_tshift_p,
                            pre_tshift_max=pre_tshift_max,
                            pre_tshift_foldk=pre_tshift_foldk,
                            pre_tshift_alpha=pre_tshift_alpha,
                            pre_tshift_apply_in_eval=pre_tshift_apply_in_eval,
                            pre_tjitter_enable=pre_tjitter_enable,
                            pre_tjitter_p=pre_tjitter_p,
                            pre_tjitter_max=pre_tjitter_max,
                            pre_tjitter_per_sample=pre_tjitter_per_sample,
                            pre_tjitter_layout=pre_tjitter_layout,
                            pre_tjitter_apply_in_eval=pre_tjitter_apply_in_eval,
                            pre_tmix_enable=pre_tmix_enable,
                            pre_tmix_p=pre_tmix_p,
                            pre_tmix_alpha=pre_tmix_alpha,
                            pre_tmix_groups=pre_tmix_groups,
                            pre_tmix_random_split=pre_tmix_random_split,
                            pre_tmix_apply_in_eval=pre_tmix_apply_in_eval,
                            pre_fdmix_enable=pre_fdmix_enable,
                            pre_fdmix_p=pre_fdmix_p,
                            pre_fdmix_alpha=pre_fdmix_alpha,
                            pre_fdmix_layout=pre_fdmix_layout,
                            pre_fdmix_apply_in_eval=pre_fdmix_apply_in_eval,
                            pre_fe_enable=pre_fe_enable,
                            pre_fe_p=pre_fe_p,
                            pre_fe_jitter=pre_fe_jitter,
                            pre_fe_radii=pre_fe_radii,
                            pre_fe_apply_in_eval=pre_fe_apply_in_eval)


# ============================================================
# CIFAR MS-ResNet (SR-A) for depth 6n+2 (32/44/56/110)
# ============================================================
class BasicBlock_CIFAR(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,
                 patchdrop_keep=1.0, patchdrop_size=4,
                 tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
                 tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
                 tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
                 tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
                 pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
                 pshuf_per_time=False, pshuf_same_on_batch=False,
                 # NEW
                 parent: Optional[nn.Module] = None):
        super().__init__()

        self.postenc_fallback = PostEncAug2D(
            patchdrop_keep=patchdrop_keep,
            patchdrop_size=patchdrop_size,
            tshift_p=tshift_p,
            tshift_max=tshift_max,
            tshift_fold_k=tshift_fold_k,
            tshift_alpha=tshift_alpha,
            tmask_p=tmask_p,
            tmask_num=tmask_num,
            tmask_max_frac=tmask_max_frac,
            tmask_min_len=tmask_min_len,
            tmask_mode=tmask_mode,
            tmask_noise_std=tmask_noise_std,
            tmask_layout=tmask_layout,
            tmask_same_on_batch=tmask_same_on_batch,
            tmask_per_channel=tmask_per_channel,
            tmask_channel_groups=tmask_channel_groups,
            pshuf_p=pshuf_p,
            pshuf_size=pshuf_size,
            pshuf_layout=pshuf_layout,
            pshuf_per_time=pshuf_per_time,
            pshuf_same_on_batch=pshuf_same_on_batch,
        )

        self.postenc = PostEncApply(parent, self.postenc_fallback)

        self.residual_function = nn.Sequential(
            mem_update(),
            self.postenc,  # <-- POST-ENCODING HERE (after spikes)
            Snn_Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            batch_norm_2d(out_channels),
            mem_update(),
            Snn_Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            batch_norm_2d1(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class MSResNet_CIFAR(nn.Module):
    def __init__(self, block, n, num_classes=100,
                 patchdrop_keep=1.0, patchdrop_size=4,
                 tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
                 tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
                 tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
                 tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
                 pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
                 pshuf_per_time=False, pshuf_same_on_batch=False,
                 # NEW: Pre-encoding TimeMask args
                 pre_tmask_enable=False,
                 pre_tmask_p=0.0,
                 pre_tmask_num=1,
                 pre_tmask_max_frac=0.25,
                 pre_tmask_min_len=1,
                 pre_tmask_mode="zero",
                 pre_tmask_noise_std=0.05,
                 pre_tmask_layout="TBCHW",
                 pre_tmask_same_on_batch=False,
                 pre_tmask_per_channel=False,
                 pre_tmask_channel_groups=1,
                 pre_tmask_apply_in_eval=False,
                 # NEW: Pre-encoding TimeShuffle args
                 pre_tshift_enable=False,
                 pre_tshift_p=0.0,
                 pre_tshift_max=1,
                 pre_tshift_foldk=32,
                 pre_tshift_alpha=0.3,
                 pre_tshift_apply_in_eval=False,
                 # NEW: Pre-encoding Temporal Jitter args
                 pre_tjitter_enable=False,
                 pre_tjitter_p=0.0,
                 pre_tjitter_max=1,
                 pre_tjitter_per_sample=False,
                 pre_tjitter_layout="TBCHW",
                 pre_tjitter_apply_in_eval=False,
                 # NEW: Pre-encoding TimeMix args
                 pre_tmix_enable=False,
                 pre_tmix_p=0.0,
                 pre_tmix_alpha=0.3,
                 pre_tmix_groups=32,
                 pre_tmix_random_split=True,
                 pre_tmix_apply_in_eval=False,
                 # NEW: Pre-encoding FullDimMix args
                 pre_fdmix_enable=False,
                 pre_fdmix_p=0.0,
                 pre_fdmix_alpha=0.5,
                 pre_fdmix_layout="TBCHW",
                 pre_fdmix_apply_in_eval=False,
                 # NEW: Pre-encoding Frequency Encoding args
                 pre_fe_enable=False,
                 pre_fe_p=1.0,
                 pre_fe_jitter=0,
                 pre_fe_radii=None,
                 pre_fe_apply_in_eval=False):
        super().__init__()

        self.time_window = time_window
        self.T = time_window

        # NEW: train_amp.py will set this (PatchMix / TimeMix / HoleFill /
        # LocalTimeShuffle / etc.)
        self.postenc_aug = None

        # NEW: generic pre-encoding hook set from train_amp.py
        self.preenc_aug = None

        # NEW: pre-encoding Frequency Encoding
        self.preenc_fe = None
        if pre_fe_enable and float(pre_fe_p) > 0.0:
            self.preenc_fe = PreEncodingFrequencyEncoding(
                T=time_window,
                radii=pre_fe_radii,
                p=pre_fe_p,
                jitter=pre_fe_jitter,
                apply_in_eval=pre_fe_apply_in_eval,
            )

        # NEW: pre-encoding TimeMask
        self.preenc_tmask = None
        if pre_tmask_enable and float(pre_tmask_p) > 0.0:
            self.preenc_tmask = PreEncodingTimeMask(
                p=pre_tmask_p,
                num_masks=pre_tmask_num,
                max_mask_frac=pre_tmask_max_frac,
                min_mask_len=pre_tmask_min_len,
                mode=pre_tmask_mode,
                noise_std=pre_tmask_noise_std,
                layout=pre_tmask_layout,
                same_on_batch=pre_tmask_same_on_batch,
                per_channel=pre_tmask_per_channel,
                channel_groups=pre_tmask_channel_groups,
                apply_in_eval=pre_tmask_apply_in_eval,
            )

        # NEW: pre-encoding TimeShuffle
        self.preenc_timeshuffle = None
        if pre_tshift_enable and float(pre_tshift_p) > 0.0:
            self.preenc_timeshuffle = PreTimeShuffle(
                p=pre_tshift_p,
                max_shift=pre_tshift_max,
                foldk=pre_tshift_foldk,
                alpha=pre_tshift_alpha,
                apply_in_eval=pre_tshift_apply_in_eval,
            )

        # NEW: pre-encoding Temporal Jitter
        self.preenc_tjitter = None
        if pre_tjitter_enable and float(pre_tjitter_p) > 0.0:
            try:
                self.preenc_tjitter = PreTemporalJitter(
                    p=pre_tjitter_p,
                    max_shift=pre_tjitter_max,
                    per_sample=pre_tjitter_per_sample,
                    layout=pre_tjitter_layout,
                    apply_in_eval=pre_tjitter_apply_in_eval,
                )
            except TypeError:
                self.preenc_tjitter = PreTemporalJitter(
                    p=pre_tjitter_p,
                    max_shift=pre_tjitter_max,
                    per_sample=pre_tjitter_per_sample,
                    layout=pre_tjitter_layout,
                )

        # NEW: pre-encoding TimeMix
        self.preenc_tmix = None
        if pre_tmix_enable and float(pre_tmix_p) > 0.0:
            self.preenc_tmix = PreEncodingTimeMix(
                p=pre_tmix_p,
                alpha=pre_tmix_alpha,
                groups=pre_tmix_groups,
                random_split=pre_tmix_random_split,
                apply_in_eval=pre_tmix_apply_in_eval,
            )

        # NEW: pre-encoding FullDimMix
        self.preenc_fulldimmix = None
        if pre_fdmix_enable and float(pre_fdmix_p) > 0.0:
            self.preenc_fulldimmix = PreEncFullDimMix(
                p=pre_fdmix_p,
                alpha=pre_fdmix_alpha,
                layout=pre_fdmix_layout,
                apply_in_eval=pre_fdmix_apply_in_eval,
            )

        self.in_channels = 16

        self.patchdrop_keep = patchdrop_keep
        self.patchdrop_size = patchdrop_size
        self.tshift_p = tshift_p
        self.tshift_max = tshift_max
        self.tshift_fold_k = tshift_fold_k
        self.tshift_alpha = tshift_alpha

        self.tmask_p = tmask_p
        self.tmask_num = tmask_num
        self.tmask_max_frac = tmask_max_frac
        self.tmask_min_len = tmask_min_len
        self.tmask_mode = tmask_mode
        self.tmask_noise_std = tmask_noise_std
        self.tmask_layout = tmask_layout
        self.tmask_same_on_batch = tmask_same_on_batch
        self.tmask_per_channel = tmask_per_channel
        self.tmask_channel_groups = tmask_channel_groups

        # PatchShuffle params
        self.pshuf_p = pshuf_p
        self.pshuf_size = pshuf_size
        self.pshuf_layout = pshuf_layout
        self.pshuf_per_time = pshuf_per_time
        self.pshuf_same_on_batch = pshuf_same_on_batch

        self.conv1 = nn.Sequential(
            Snn_Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            batch_norm_2d(16),
        )

        self.layer1 = self._make_layer(block, out_channels=16, num_blocks=n, stride=1)
        self.layer2 = self._make_layer(block, out_channels=32, num_blocks=n, stride=2)
        self.layer3 = self._make_layer(block, out_channels=64, num_blocks=n, stride=2)

        self.mem_update = mem_update()
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def set_input_encoder(self, encoder):
        self.preenc_fe = encoder

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s,
                                self.patchdrop_keep, self.patchdrop_size,
                                self.tshift_p, self.tshift_max, self.tshift_fold_k, self.tshift_alpha,
                                self.tmask_p, self.tmask_num, self.tmask_max_frac, self.tmask_min_len,
                                self.tmask_mode, self.tmask_noise_std, self.tmask_layout,
                                self.tmask_same_on_batch, self.tmask_per_channel, self.tmask_channel_groups,
                                self.pshuf_p, self.pshuf_size, self.pshuf_layout,
                                self.pshuf_per_time, self.pshuf_same_on_batch,
                                parent=self))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # NEW: pre-encoding Frequency Encoding before direct repeat
        if self.preenc_fe is not None:
            input = self.preenc_fe(x)
        else:
            input = torch.zeros(time_window, x.size(0), 3, x.size(2), x.size(3), device=x.device)
            input[:] = x

        # NEW: pre-encoding FullDimMix right after 5D expansion
        input = _maybe_apply_preenc_fulldimmix(self, input)

        # NEW: generic pre-encoding augmentation hook from train_amp.py
        input = _maybe_apply_preenc_aug(self, input)

        # NEW: pre-encoding TimeMask before first conv
        if self.preenc_tmask is not None:
            input = self.preenc_tmask(input)

        # NEW: pre-encoding TimeShuffle before first conv
        input = _maybe_apply_preenc_timeshuffle(self, input)

        # NEW: pre-encoding Temporal Jitter before first conv
        # Skip native tjitter if it is already present inside self.preenc_aug
        if not _module_tree_contains_type(getattr(self, "preenc_aug", None), {"PreTimeShuffle", "PreTemporalJitter"}):
            input = _maybe_apply_preenc_temporaljitter(self, input)

        # NEW: pre-encoding TimeMix before first conv
        if self.preenc_tmix is not None:
            input = self.preenc_tmix(input)

        out = self.conv1(input)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.mem_update(out)
        out = F.adaptive_avg_pool3d(out, (None, 1, 1))
        out = out.view(out.size(0), out.size(1), -1)   # [T,B,C]
        out = out.sum(dim=0) / out.size(0)             # [B,C]
        out = self.fc(out)
        return out


def resnet110(num_classes=100,
              patchdrop_keep=1.0, patchdrop_size=4,
              tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
              tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
              tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
              tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
              pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
              pshuf_per_time=False, pshuf_same_on_batch=False,
              # NEW: Pre-encoding TimeMask args
              pre_tmask_enable=False,
              pre_tmask_p=0.0,
              pre_tmask_num=1,
              pre_tmask_max_frac=0.25,
              pre_tmask_min_len=1,
              pre_tmask_mode="zero",
              pre_tmask_noise_std=0.05,
              pre_tmask_layout="TBCHW",
              pre_tmask_same_on_batch=False,
              pre_tmask_per_channel=False,
              pre_tmask_channel_groups=1,
              pre_tmask_apply_in_eval=False,
              # NEW: Pre-encoding TimeShuffle args
              pre_tshift_enable=False,
              pre_tshift_p=0.0,
              pre_tshift_max=1,
              pre_tshift_foldk=32,
              pre_tshift_alpha=0.3,
              pre_tshift_apply_in_eval=False,
              # NEW: Pre-encoding Temporal Jitter args
              pre_tjitter_enable=False,
              pre_tjitter_p=0.0,
              pre_tjitter_max=1,
              pre_tjitter_per_sample=False,
              pre_tjitter_layout="TBCHW",
              pre_tjitter_apply_in_eval=False,
              # NEW: Pre-encoding TimeMix args
              pre_tmix_enable=False,
              pre_tmix_p=0.0,
              pre_tmix_alpha=0.3,
              pre_tmix_groups=32,
              pre_tmix_random_split=True,
              pre_tmix_apply_in_eval=False,
              # NEW: Pre-encoding FullDimMix args
              pre_fdmix_enable=False,
              pre_fdmix_p=0.0,
              pre_fdmix_alpha=0.5,
              pre_fdmix_layout="TBCHW",
              pre_fdmix_apply_in_eval=False,
              # NEW: Pre-encoding Frequency Encoding args
              pre_fe_enable=False,
              pre_fe_p=1.0,
              pre_fe_jitter=0,
              pre_fe_radii=None,
              pre_fe_apply_in_eval=False):
    return MSResNet_CIFAR(BasicBlock_CIFAR, n=18, num_classes=num_classes,
                          patchdrop_keep=patchdrop_keep, patchdrop_size=patchdrop_size,
                          tshift_p=tshift_p, tshift_max=tshift_max,
                          tshift_fold_k=tshift_fold_k, tshift_alpha=tshift_alpha,
                          tmask_p=tmask_p, tmask_num=tmask_num, tmask_max_frac=tmask_max_frac, tmask_min_len=tmask_min_len,
                          tmask_mode=tmask_mode, tmask_noise_std=tmask_noise_std, tmask_layout=tmask_layout,
                          tmask_same_on_batch=tmask_same_on_batch, tmask_per_channel=tmask_per_channel, tmask_channel_groups=tmask_channel_groups,
                          pshuf_p=pshuf_p, pshuf_size=pshuf_size, pshuf_layout=pshuf_layout,
                          pshuf_per_time=pshuf_per_time, pshuf_same_on_batch=pshuf_same_on_batch,
                          pre_tmask_enable=pre_tmask_enable,
                          pre_tmask_p=pre_tmask_p,
                          pre_tmask_num=pre_tmask_num,
                          pre_tmask_max_frac=pre_tmask_max_frac,
                          pre_tmask_min_len=pre_tmask_min_len,
                          pre_tmask_mode=pre_tmask_mode,
                          pre_tmask_noise_std=pre_tmask_noise_std,
                          pre_tmask_layout=pre_tmask_layout,
                          pre_tmask_same_on_batch=pre_tmask_same_on_batch,
                          pre_tmask_per_channel=pre_tmask_per_channel,
                          pre_tmask_channel_groups=pre_tmask_channel_groups,
                          pre_tmask_apply_in_eval=pre_tmask_apply_in_eval,
                          pre_tshift_enable=pre_tshift_enable,
                          pre_tshift_p=pre_tshift_p,
                          pre_tshift_max=pre_tshift_max,
                          pre_tshift_foldk=pre_tshift_foldk,
                          pre_tshift_alpha=pre_tshift_alpha,
                          pre_tshift_apply_in_eval=pre_tshift_apply_in_eval,
                          pre_tjitter_enable=pre_tjitter_enable,
                          pre_tjitter_p=pre_tjitter_p,
                          pre_tjitter_max=pre_tjitter_max,
                          pre_tjitter_per_sample=pre_tjitter_per_sample,
                          pre_tjitter_layout=pre_tjitter_layout,
                          pre_tjitter_apply_in_eval=pre_tjitter_apply_in_eval,
                          pre_tmix_enable=pre_tmix_enable,
                          pre_tmix_p=pre_tmix_p,
                          pre_tmix_alpha=pre_tmix_alpha,
                          pre_tmix_groups=pre_tmix_groups,
                          pre_tmix_random_split=pre_tmix_random_split,
                          pre_tmix_apply_in_eval=pre_tmix_apply_in_eval,
                          pre_fdmix_enable=pre_fdmix_enable,
                          pre_fdmix_p=pre_fdmix_p,
                          pre_fdmix_alpha=pre_fdmix_alpha,
                          pre_fdmix_layout=pre_fdmix_layout,
                          pre_fdmix_apply_in_eval=pre_fdmix_apply_in_eval,
                          pre_fe_enable=pre_fe_enable,
                          pre_fe_p=pre_fe_p,
                          pre_fe_jitter=pre_fe_jitter,
                          pre_fe_radii=pre_fe_radii,
                          pre_fe_apply_in_eval=pre_fe_apply_in_eval)


def resnet56(num_classes=100,
             patchdrop_keep=1.0, patchdrop_size=4,
             tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
             tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
             tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
             tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
             pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
             pshuf_per_time=False, pshuf_same_on_batch=False,
             # NEW: Pre-encoding TimeMask args
             pre_tmask_enable=False,
             pre_tmask_p=0.0,
             pre_tmask_num=1,
             pre_tmask_max_frac=0.25,
             pre_tmask_min_len=1,
             pre_tmask_mode="zero",
             pre_tmask_noise_std=0.05,
             pre_tmask_layout="TBCHW",
             pre_tmask_same_on_batch=False,
             pre_tmask_per_channel=False,
             pre_tmask_channel_groups=1,
             pre_tmask_apply_in_eval=False,
             # NEW: Pre-encoding TimeShuffle args
             pre_tshift_enable=False,
             pre_tshift_p=0.0,
             pre_tshift_max=1,
             pre_tshift_foldk=32,
             pre_tshift_alpha=0.3,
             pre_tshift_apply_in_eval=False,
             # NEW: Pre-encoding Temporal Jitter args
             pre_tjitter_enable=False,
             pre_tjitter_p=0.0,
             pre_tjitter_max=1,
             pre_tjitter_per_sample=False,
             pre_tjitter_layout="TBCHW",
             pre_tjitter_apply_in_eval=False,
             # NEW: Pre-encoding TimeMix args
             pre_tmix_enable=False,
             pre_tmix_p=0.0,
             pre_tmix_alpha=0.3,
             pre_tmix_groups=32,
             pre_tmix_random_split=True,
             pre_tmix_apply_in_eval=False,
             # NEW: Pre-encoding FullDimMix args
             pre_fdmix_enable=False,
             pre_fdmix_p=0.0,
             pre_fdmix_alpha=0.5,
             pre_fdmix_layout="TBCHW",
             pre_fdmix_apply_in_eval=False,
             # NEW: Pre-encoding Frequency Encoding args
             pre_fe_enable=False,
             pre_fe_p=1.0,
             pre_fe_jitter=0,
             pre_fe_radii=None,
             pre_fe_apply_in_eval=False):
    return MSResNet_CIFAR(BasicBlock_CIFAR, n=9, num_classes=num_classes,
                          patchdrop_keep=patchdrop_keep, patchdrop_size=patchdrop_size,
                          tshift_p=tshift_p, tshift_max=tshift_max,
                          tshift_fold_k=tshift_fold_k, tshift_alpha=tshift_alpha,
                          tmask_p=tmask_p, tmask_num=tmask_num, tmask_max_frac=tmask_max_frac, tmask_min_len=tmask_min_len,
                          tmask_mode=tmask_mode, tmask_noise_std=tmask_noise_std, tmask_layout=tmask_layout,
                          tmask_same_on_batch=tmask_same_on_batch, tmask_per_channel=tmask_per_channel, tmask_channel_groups=tmask_channel_groups,
                          pshuf_p=pshuf_p, pshuf_size=pshuf_size, pshuf_layout=pshuf_layout,
                          pshuf_per_time=pshuf_per_time, pshuf_same_on_batch=pshuf_same_on_batch,
                          pre_tmask_enable=pre_tmask_enable,
                          pre_tmask_p=pre_tmask_p,
                          pre_tmask_num=pre_tmask_num,
                          pre_tmask_max_frac=pre_tmask_max_frac,
                          pre_tmask_min_len=pre_tmask_min_len,
                          pre_tmask_mode=pre_tmask_mode,
                          pre_tmask_noise_std=pre_tmask_noise_std,
                          pre_tmask_layout=pre_tmask_layout,
                          pre_tmask_same_on_batch=pre_tmask_same_on_batch,
                          pre_tmask_per_channel=pre_tmask_per_channel,
                          pre_tmask_channel_groups=pre_tmask_channel_groups,
                          pre_tmask_apply_in_eval=pre_tmask_apply_in_eval,
                          pre_tshift_enable=pre_tshift_enable,
                          pre_tshift_p=pre_tshift_p,
                          pre_tshift_max=pre_tshift_max,
                          pre_tshift_foldk=pre_tshift_foldk,
                          pre_tshift_alpha=pre_tshift_alpha,
                          pre_tshift_apply_in_eval=pre_tshift_apply_in_eval,
                          pre_tjitter_enable=pre_tjitter_enable,
                          pre_tjitter_p=pre_tjitter_p,
                          pre_tjitter_max=pre_tjitter_max,
                          pre_tjitter_per_sample=pre_tjitter_per_sample,
                          pre_tjitter_layout=pre_tjitter_layout,
                          pre_tjitter_apply_in_eval=pre_tjitter_apply_in_eval,
                          pre_tmix_enable=pre_tmix_enable,
                          pre_tmix_p=pre_tmix_p,
                          pre_tmix_alpha=pre_tmix_alpha,
                          pre_tmix_groups=pre_tmix_groups,
                          pre_tmix_random_split=pre_tmix_random_split,
                          pre_tmix_apply_in_eval=pre_tmix_apply_in_eval,
                          pre_fdmix_enable=pre_fdmix_enable,
                          pre_fdmix_p=pre_fdmix_p,
                          pre_fdmix_alpha=pre_fdmix_alpha,
                          pre_fdmix_layout=pre_fdmix_layout,
                          pre_fdmix_apply_in_eval=pre_fdmix_apply_in_eval,
                          pre_fe_enable=pre_fe_enable,
                          pre_fe_p=pre_fe_p,
                          pre_fe_jitter=pre_fe_jitter,
                          pre_fe_radii=pre_fe_radii,
                          pre_fe_apply_in_eval=pre_fe_apply_in_eval)


def resnet44(num_classes=100,
             patchdrop_keep=1.0, patchdrop_size=4,
             tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
             tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
             tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
             tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
             pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
             pshuf_per_time=False, pshuf_same_on_batch=False,
             # NEW: Pre-encoding TimeMask args
             pre_tmask_enable=False,
             pre_tmask_p=0.0,
             pre_tmask_num=1,
             pre_tmask_max_frac=0.25,
             pre_tmask_min_len=1,
             pre_tmask_mode="zero",
             pre_tmask_noise_std=0.05,
             pre_tmask_layout="TBCHW",
             pre_tmask_same_on_batch=False,
             pre_tmask_per_channel=False,
             pre_tmask_channel_groups=1,
             pre_tmask_apply_in_eval=False,
             # NEW: Pre-encoding TimeShuffle args
             pre_tshift_enable=False,
             pre_tshift_p=0.0,
             pre_tshift_max=1,
             pre_tshift_foldk=32,
             pre_tshift_alpha=0.3,
             pre_tshift_apply_in_eval=False,
             # NEW: Pre-encoding Temporal Jitter args
             pre_tjitter_enable=False,
             pre_tjitter_p=0.0,
             pre_tjitter_max=1,
             pre_tjitter_per_sample=False,
             pre_tjitter_layout="TBCHW",
             pre_tjitter_apply_in_eval=False,
             # NEW: Pre-encoding TimeMix args
             pre_tmix_enable=False,
             pre_tmix_p=0.0,
             pre_tmix_alpha=0.3,
             pre_tmix_groups=32,
             pre_tmix_random_split=True,
             pre_tmix_apply_in_eval=False,
             # NEW: Pre-encoding FullDimMix args
             pre_fdmix_enable=False,
             pre_fdmix_p=0.0,
             pre_fdmix_alpha=0.5,
             pre_fdmix_layout="TBCHW",
             pre_fdmix_apply_in_eval=False,
             # NEW: Pre-encoding Frequency Encoding args
             pre_fe_enable=False,
             pre_fe_p=1.0,
             pre_fe_jitter=0,
             pre_fe_radii=None,
             pre_fe_apply_in_eval=False):
    return MSResNet_CIFAR(BasicBlock_CIFAR, n=7, num_classes=num_classes,
                          patchdrop_keep=patchdrop_keep, patchdrop_size=patchdrop_size,
                          tshift_p=tshift_p, tshift_max=tshift_max,
                          tshift_fold_k=tshift_fold_k, tshift_alpha=tshift_alpha,
                          tmask_p=tmask_p, tmask_num=tmask_num, tmask_max_frac=tmask_max_frac, tmask_min_len=tmask_min_len,
                          tmask_mode=tmask_mode, tmask_noise_std=tmask_noise_std, tmask_layout=tmask_layout,
                          tmask_same_on_batch=tmask_same_on_batch, tmask_per_channel=tmask_per_channel, tmask_channel_groups=tmask_channel_groups,
                          pshuf_p=pshuf_p, pshuf_size=pshuf_size, pshuf_layout=pshuf_layout,
                          pshuf_per_time=pshuf_per_time, pshuf_same_on_batch=pshuf_same_on_batch,
                          pre_tmask_enable=pre_tmask_enable,
                          pre_tmask_p=pre_tmask_p,
                          pre_tmask_num=pre_tmask_num,
                          pre_tmask_max_frac=pre_tmask_max_frac,
                          pre_tmask_min_len=pre_tmask_min_len,
                          pre_tmask_mode=pre_tmask_mode,
                          pre_tmask_noise_std=pre_tmask_noise_std,
                          pre_tmask_layout=pre_tmask_layout,
                          pre_tmask_same_on_batch=pre_tmask_same_on_batch,
                          pre_tmask_per_channel=pre_tmask_per_channel,
                          pre_tmask_channel_groups=pre_tmask_channel_groups,
                          pre_tmask_apply_in_eval=pre_tmask_apply_in_eval,
                          pre_tshift_enable=pre_tshift_enable,
                          pre_tshift_p=pre_tshift_p,
                          pre_tshift_max=pre_tshift_max,
                          pre_tshift_foldk=pre_tshift_foldk,
                          pre_tshift_alpha=pre_tshift_alpha,
                          pre_tshift_apply_in_eval=pre_tshift_apply_in_eval,
                          pre_tjitter_enable=pre_tjitter_enable,
                          pre_tjitter_p=pre_tjitter_p,
                          pre_tjitter_max=pre_tjitter_max,
                          pre_tjitter_per_sample=pre_tjitter_per_sample,
                          pre_tjitter_layout=pre_tjitter_layout,
                          pre_tjitter_apply_in_eval=pre_tjitter_apply_in_eval,
                          pre_tmix_enable=pre_tmix_enable,
                          pre_tmix_p=pre_tmix_p,
                          pre_tmix_alpha=pre_tmix_alpha,
                          pre_tmix_groups=pre_tmix_groups,
                          pre_tmix_random_split=pre_tmix_random_split,
                          pre_tmix_apply_in_eval=pre_tmix_apply_in_eval,
                          pre_fdmix_enable=pre_fdmix_enable,
                          pre_fdmix_p=pre_fdmix_p,
                          pre_fdmix_alpha=pre_fdmix_alpha,
                          pre_fdmix_layout=pre_fdmix_layout,
                          pre_fdmix_apply_in_eval=pre_fdmix_apply_in_eval,
                          pre_fe_enable=pre_fe_enable,
                          pre_fe_p=pre_fe_p,
                          pre_fe_jitter=pre_fe_jitter,
                          pre_fe_radii=pre_fe_radii,
                          pre_fe_apply_in_eval=pre_fe_apply_in_eval)


def resnet32(num_classes=100,
             patchdrop_keep=1.0, patchdrop_size=4,
             tshift_p=0.0, tshift_max=1, tshift_fold_k=32, tshift_alpha=0.3,
             tmask_p=0.0, tmask_num=1, tmask_max_frac=0.25, tmask_min_len=1,
             tmask_mode="zero", tmask_noise_std=0.05, tmask_layout="TB",
             tmask_same_on_batch=False, tmask_per_channel=False, tmask_channel_groups=1,
             pshuf_p=0.0, pshuf_size=4, pshuf_layout="TB",
             pshuf_per_time=False, pshuf_same_on_batch=False,
             # NEW: Pre-encoding TimeMask args
             pre_tmask_enable=False,
             pre_tmask_p=0.0,
             pre_tmask_num=1,
             pre_tmask_max_frac=0.25,
             pre_tmask_min_len=1,
             pre_tmask_mode="zero",
             pre_tmask_noise_std=0.05,
             pre_tmask_layout="TBCHW",
             pre_tmask_same_on_batch=False,
             pre_tmask_per_channel=False,
             pre_tmask_channel_groups=1,
             pre_tmask_apply_in_eval=False,
             # NEW: Pre-encoding TimeShuffle args
             pre_tshift_enable=False,
             pre_tshift_p=0.0,
             pre_tshift_max=1,
             pre_tshift_foldk=32,
             pre_tshift_alpha=0.3,
             pre_tshift_apply_in_eval=False,
             # NEW: Pre-encoding Temporal Jitter args
             pre_tjitter_enable=False,
             pre_tjitter_p=0.0,
             pre_tjitter_max=1,
             pre_tjitter_per_sample=False,
             pre_tjitter_layout="TBCHW",
             pre_tjitter_apply_in_eval=False,
             # NEW: Pre-encoding TimeMix args
             pre_tmix_enable=False,
             pre_tmix_p=0.0,
             pre_tmix_alpha=0.3,
             pre_tmix_groups=32,
             pre_tmix_random_split=True,
             pre_tmix_apply_in_eval=False,
             # NEW: Pre-encoding FullDimMix args
             pre_fdmix_enable=False,
             pre_fdmix_p=0.0,
             pre_fdmix_alpha=0.5,
             pre_fdmix_layout="TBCHW",
             pre_fdmix_apply_in_eval=False,
             # NEW: Pre-encoding Frequency Encoding args
             pre_fe_enable=False,
             pre_fe_p=1.0,
             pre_fe_jitter=0,
             pre_fe_radii=None,
             pre_fe_apply_in_eval=False):
    return MSResNet_CIFAR(BasicBlock_CIFAR, n=5, num_classes=num_classes,
                          patchdrop_keep=patchdrop_keep, patchdrop_size=patchdrop_size,
                          tshift_p=tshift_p, tshift_max=tshift_max,
                          tshift_fold_k=tshift_fold_k, tshift_alpha=tshift_alpha,
                          tmask_p=tmask_p, tmask_num=tmask_num, tmask_max_frac=tmask_max_frac, tmask_min_len=tmask_min_len,
                          tmask_mode=tmask_mode, tmask_noise_std=tmask_noise_std, tmask_layout=tmask_layout,
                          tmask_same_on_batch=tmask_same_on_batch, tmask_per_channel=tmask_per_channel, tmask_channel_groups=tmask_channel_groups,
                          pshuf_p=pshuf_p, pshuf_size=pshuf_size, pshuf_layout=pshuf_layout,
                          pshuf_per_time=pshuf_per_time, pshuf_same_on_batch=pshuf_same_on_batch,
                          pre_tmask_enable=pre_tmask_enable,
                          pre_tmask_p=pre_tmask_p,
                          pre_tmask_num=pre_tmask_num,
                          pre_tmask_max_frac=pre_tmask_max_frac,
                          pre_tmask_min_len=pre_tmask_min_len,
                          pre_tmask_mode=pre_tmask_mode,
                          pre_tmask_noise_std=pre_tmask_noise_std,
                          pre_tmask_layout=pre_tmask_layout,
                          pre_tmask_same_on_batch=pre_tmask_same_on_batch,
                          pre_tmask_per_channel=pre_tmask_per_channel,
                          pre_tmask_channel_groups=pre_tmask_channel_groups,
                          pre_tmask_apply_in_eval=pre_tmask_apply_in_eval,
                          pre_tshift_enable=pre_tshift_enable,
                          pre_tshift_p=pre_tshift_p,
                          pre_tshift_max=pre_tshift_max,
                          pre_tshift_foldk=pre_tshift_foldk,
                          pre_tshift_alpha=pre_tshift_alpha,
                          pre_tshift_apply_in_eval=pre_tshift_apply_in_eval,
                          pre_tjitter_enable=pre_tjitter_enable,
                          pre_tjitter_p=pre_tjitter_p,
                          pre_tjitter_max=pre_tjitter_max,
                          pre_tjitter_per_sample=pre_tjitter_per_sample,
                          pre_tjitter_layout=pre_tjitter_layout,
                          pre_tjitter_apply_in_eval=pre_tjitter_apply_in_eval,
                          pre_tmix_enable=pre_tmix_enable,
                          pre_tmix_p=pre_tmix_p,
                          pre_tmix_alpha=pre_tmix_alpha,
                          pre_tmix_groups=pre_tmix_groups,
                          pre_tmix_random_split=pre_tmix_random_split,
                          pre_tmix_apply_in_eval=pre_tmix_apply_in_eval,
                          pre_fdmix_enable=pre_fdmix_enable,
                          pre_fdmix_p=pre_fdmix_p,
                          pre_fdmix_alpha=pre_fdmix_alpha,
                          pre_fdmix_layout=pre_fdmix_layout,
                          pre_fdmix_apply_in_eval=pre_fdmix_apply_in_eval,
                          pre_fe_enable=pre_fe_enable,
                          pre_fe_p=pre_fe_p,
                          pre_fe_jitter=pre_fe_jitter,
                          pre_fe_radii=pre_fe_radii,
                          pre_fe_apply_in_eval=pre_fe_apply_in_eval)