# =========================
# File: train_amp.py  (UPDATED - keeps all existing code + adds PRE-encoding PatchMix + PRE-encoding TimeMask + PRE-encoding IPMix + PRE-encoding TimeShuffle + PRE-encoding Temporal Jitter + PRE-encoding FullDimMix + POST-encoding LocalTimeShuffle + Frequency Encoding)
# =========================
# [NEW] Also includes PRE-encoding Center Patch MinLift
import os
import argparse
import time
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import distutils
import distutils.version  # <-- keep
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torch.cuda.amp
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader

from typing import Optional

from PIL import Image
from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms

from models.preenc_timemix import PreEncodingTimeMix

from models.preenc_patchshuffle2d import PreEncodingPatchShuffle2D
from models.preenc_patchdropout2d import PreEncPatchDropout2D
from models.preenc_patchmix2d import PreEncodingPatchMix2D
from models.holefill import HoleFillPostEncoding

# NEW: PRE-encoding Temporal Jitter
from models.pre_temporaljitter import PreTemporalJitter

# NEW: PRE-encoding Center Patch MinLift
from models.center_patch_minlift import CenterPatchMinLift

# NEW: Frequency Encoding
from models.frequency_encoding_aug import FrequencyEncodingAug, make_default_fe_radii

# Post-encoding augs
from models.timemask import TimeMaskPostEncoding
from models.patchshuffle2d import PatchShufflePostEncoding2D
from models.postenc_patchmix import PostEncPatchMix
from models.postenc_timemix import PostEncTimeMix, TimeMixConfig
from models.classbatchmix import ClassBatchMixPostEncoding
from models.postenc_pipeline import PostEncAugPipeline

# [NEW] HoleFill
from models.holefill import HoleFillPostEncoding

# [NEW] IPMix utils (copy your SDT IPMix_utils.py into models/)
import models.IPMix_utils as ipmix_utils


# =========================
# [NEW] LocalTimeShuffle post-encoding augmentation
# =========================
class LocalTimeShufflePostEncoding(nn.Module):
    """
    Local Time Shuffle for temporal spike tensors.

    Supported layouts:
        - TBCHW : [T, B, C, H, W]
        - BTCHW : [B, T, C, H, W]

    It splits the time axis into local windows of size `window_size`
    and shuffles only within each window.

    Notes:
        - train-only by default
        - no learnable parameters
        - safe to attach through core.postenc_aug
    """
    def __init__(
        self,
        window_size: int = 2,
        p: float = 1.0,
        layout: str = "TBCHW",
        per_sample: bool = False,
        apply_in_eval: bool = False,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.p = float(p)
        self.layout = str(layout).upper()
        self.per_sample = bool(per_sample)
        self.apply_in_eval = bool(apply_in_eval)

        if self.layout not in ("TBCHW", "BTCHW"):
            raise ValueError("LocalTimeShufflePostEncoding layout must be 'TBCHW' or 'BTCHW'")

    def extra_repr(self):
        return (
            f"window_size={self.window_size}, p={self.p}, layout='{self.layout}', "
            f"per_sample={self.per_sample}, apply_in_eval={self.apply_in_eval}"
        )

    def _make_index(self, T: int, device):
        ws = max(1, min(self.window_size, T))
        if ws <= 1:
            return None

        parts = []
        start = 0
        base = torch.arange(T, device=device)

        while start < T:
            end = min(start + ws, T)
            perm = torch.randperm(end - start, device=device)
            parts.append(base[start:end][perm])
            start = end

        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if (not self.training) and (not self.apply_in_eval):
            return x

        if x is None or x.dim() != 5:
            return x

        if self.window_size <= 1 or self.p <= 0.0:
            return x

        if self.layout == "TBCHW":
            time_dim = 0
            batch_dim = 1
        else:  # BTCHW
            time_dim = 1
            batch_dim = 0

        T = x.size(time_dim)
        if T <= 1:
            return x

        if self.per_sample:
            B = x.size(batch_dim)
            out = []

            if self.layout == "TBCHW":
                # x[:, b] -> [T, C, H, W]
                for b in range(B):
                    xb = x[:, b]
                    if torch.rand((), device=x.device) < self.p:
                        idx = self._make_index(T, x.device)
                        if idx is not None:
                            xb = xb.index_select(0, idx)
                    out.append(xb)
                return torch.stack(out, dim=1)  # [T, B, C, H, W]

            else:  # BTCHW
                # x[b] -> [T, C, H, W]
                for b in range(B):
                    xb = x[b]
                    if torch.rand((), device=x.device) < self.p:
                        idx = self._make_index(T, x.device)
                        if idx is not None:
                            xb = xb.index_select(0, idx)
                    out.append(xb)
                return torch.stack(out, dim=0)  # [B, T, C, H, W]

        else:
            if torch.rand((), device=x.device) >= self.p:
                return x

            idx = self._make_index(T, x.device)
            if idx is None:
                return x

            return x.index_select(time_dim, idx)


# =========================
# Helpers (folder + logging)
# =========================

def _unwrap_model(net):
    return net.module if hasattr(net, "module") else net


def _set_input_encoder(net, encoder):
    """
    Frequency Encoding attachment helper.

    Supports both:
      1) models exposing set_input_encoder(...)
      2) your updated MS_ResNet.py style with attribute preenc_fe
    """
    m = _unwrap_model(net)

    if hasattr(m, "set_input_encoder"):
        m.set_input_encoder(encoder)
        return

    if hasattr(m, "preenc_fe"):
        m.preenc_fe = encoder
        return

    raise AttributeError(
        "Model does not implement set_input_encoder() and has no 'preenc_fe' attribute. "
        "Please add the MS_ResNet.py Frequency Encoding changes first."
    )


def _attach_preenc_aug(net, aug):
    """
    Generic pre-encoding attachment.
    If a pre-encoding aug already exists, chain them with nn.Sequential
    instead of overwriting it.
    """
    if aug is None:
        return net

    base = _unwrap_model(net)
    current = getattr(base, "preenc_aug", None)

    if current is None:
        base.preenc_aug = aug
    else:
        if isinstance(current, nn.Sequential):
            current.add_module("extra_preenc_aug_{}".format(len(current)), aug)
            base.preenc_aug = current
        else:
            base.preenc_aug = nn.Sequential(current, aug)

    return net


def _module_tree_contains_type(module: Optional[nn.Module], type_name: str) -> bool:
    if module is None:
        return False
    if module.__class__.__name__ == type_name:
        return True
    for child in module.children():
        if _module_tree_contains_type(child, type_name):
            return True
    return False


def _slugify(s: str) -> str:
    # safe folder/file names
    s = s.strip().replace(" ", "_")
    s = s.replace("/", "_")
    s = s.replace(":", "_")
    return s


def _build_aug_tag(args) -> str:
    """
    Priority:
      1) user-provided --aug-name
      2) otherwise combine active aug tags safely

    We allow:
      - at most ONE post-encoding augmentation (already enforced later)
      - plus optional Frequency Encoding
      - plus optional pre-encoding Center Patch MinLift
      - plus optional pre-encoding PatchShuffle
      - plus optional pre-encoding PatchMix
      - plus optional pre-encoding PatchDropout
      - plus optional pre-encoding TimeMask
      - plus optional pre-encoding TimeShuffle
      - plus optional pre-encoding Temporal Jitter
      - plus optional pre-encoding TimeMix
      - plus optional pre-encoding FullDimMix
      - plus optional pre-encoding IPMix
    """
    if getattr(args, "aug_name", ""):
        return _slugify(args.aug_name)

    tags = []

    # ---- Frequency Encoding ----
    if getattr(args, "fe", False):
        p_str = str(getattr(args, "fe_prob", 1.0)).replace(".", "p")
        tag = f"fe_p{p_str}_j{int(getattr(args, 'fe_jitter', 0))}"
        if getattr(args, "fe_radii", None) is not None:
            radii_tag = "-".join(str(r) for r in getattr(args, "fe_radii"))
            tag += f"_r{radii_tag}"
        else:
            tag += "_rauto"
        if getattr(args, "fe_eval", False):
            tag += "_eval"
        tags.append(tag)

    # ---- PRE-encoding Center Patch MinLift ----
    if getattr(args, "center_patch_minlift", False) and float(getattr(args, "center_patch_minlift_p", 0.0)) > 0.0:
        pf_str = str(getattr(args, "center_patch_minlift_patch_frac", 0.5)).replace(".", "p")
        amin_str = str(getattr(args, "center_patch_minlift_alpha_min", 0.3)).replace(".", "p")
        amax_str = str(getattr(args, "center_patch_minlift_alpha_max", 0.7)).replace(".", "p")
        p_str = str(getattr(args, "center_patch_minlift_p", 0.5)).replace(".", "p")
        tags.append(f"pre_centerpatchminlift_pf{pf_str}_amin{amin_str}_amax{amax_str}_p{p_str}")

    # ---- PRE-encoding PatchShuffle ----
    if getattr(args, "pre_pshuf_enable", False) and float(getattr(args, "pre_pshuf_p", 0.0)) > 0.0:
        p_str = str(getattr(args, "pre_pshuf_p", 0.0)).replace(".", "p")
        ps = int(getattr(args, "pre_pshuf_size", 2))
        tags.append(f"pre_patchshuffle_p{p_str}_ps{ps}")

    # ---- PRE-encoding PatchMix ----
    if getattr(args, "pre_pmix_enable", False) and float(getattr(args, "pre_pmix_p", 0.0)) > 0.0:
        p_str = str(getattr(args, "pre_pmix_p", 0.0)).replace(".", "p")
        ps = int(getattr(args, "pre_pmix_size", 16))
        a_str = str(getattr(args, "pre_pmix_alpha", 0.2)).replace(".", "p")
        b_str = str(getattr(args, "pre_pmix_beta", 0.2)).replace(".", "p")
        tag = f"pre_patchmix_p{p_str}_ps{ps}_a{a_str}_b{b_str}"
        if getattr(args, "pre_pmix_no_pad", False):
            tag += "_nopad"
        tags.append(tag)

    # ---- PRE-encoding PatchDropout ----
    if getattr(args, "preenc_patchdropout", False):
        pd_keep = float(getattr(args, "pd_keep", 1.0))
        pd_min_keep = getattr(args, "pd_min_keep", None)
        pd_h = int(getattr(args, "pd_patch_h", 4))
        pd_w = getattr(args, "pd_patch_w", None)
        pd_w = pd_h if pd_w is None else int(pd_w)

        # effective only if keep<1 or min_keep is set
        if (pd_keep < 1.0) or (pd_min_keep is not None):
            if pd_min_keep is not None:
                mk_str = str(pd_min_keep).replace(".", "p")
                tags.append(f"pre_patchdropout_mink{mk_str}_ps{pd_h}x{pd_w}")
            else:
                k_str = str(pd_keep).replace(".", "p")
                tags.append(f"pre_patchdropout_k{k_str}_ps{pd_h}x{pd_w}")

    # ---- PRE-encoding TimeMask ----
    if getattr(args, "pre_tmask_enable", False) and float(getattr(args, "pre_tmask_p", 0.0)) > 0.0:
        p_str = str(getattr(args, "pre_tmask_p", 0.0)).replace(".", "p")
        mf_str = str(getattr(args, "pre_tmask_max_frac", 0.25)).replace(".", "p")
        tag = (
            f"pre_timemask_p{p_str}_n{int(getattr(args, 'pre_tmask_num', 1))}"
            f"_mf{mf_str}_ml{int(getattr(args, 'pre_tmask_min_len', 1))}"
            f"_{str(getattr(args, 'pre_tmask_mode', 'zero'))}"
        )
        tags.append(tag)

    # ---- PRE-encoding TimeShuffle ----
    if getattr(args, "pre_tshift_enable", False) and float(getattr(args, "pre_tshift_p", 0.0)) > 0.0:
        p_str = str(getattr(args, "pre_tshift_p", 0.0)).replace(".", "p")
        a_str = str(getattr(args, "pre_tshift_alpha", 0.3)).replace(".", "p")
        tags.append(
            f"pre_timeshuffle_p{p_str}_max{int(getattr(args, 'pre_tshift_max', 1))}"
            f"_ck{int(getattr(args, 'pre_tshift_foldk', 32))}_a{a_str}"
        )

    # ---- PRE-encoding Temporal Jitter ----
    if getattr(args, "pre_tjitter_enable", False) and float(getattr(args, "pre_tjitter_p", 0.0)) > 0.0:
        p_str = str(getattr(args, "pre_tjitter_p", 0.0)).replace(".", "p")
        tag = (
            f"pre_temporaljitter_p{p_str}_max{int(getattr(args, 'pre_tjitter_max', 1))}"
            f"_{str(getattr(args, 'pre_tjitter_layout', 'TBCHW'))}"
        )
        if getattr(args, "pre_tjitter_per_sample", False):
            tag += "_persample"
        if getattr(args, "pre_tjitter_apply_in_eval", False):
            tag += "_eval"
        tags.append(tag)

    # ---- PRE-encoding TimeMix ----
    if getattr(args, "pre_tmix_enable", False) and float(getattr(args, "pre_tmix_p", 0.0)) > 0.0:
        p_str = str(getattr(args, "pre_tmix_p", 0.0)).replace(".", "p")
        a_str = str(getattr(args, "pre_tmix_alpha", 0.3)).replace(".", "p")
        g_str = str(getattr(args, "pre_tmix_groups", 32))
        split_tag = "rand" if getattr(args, "pre_tmix_random_split", True) else "fixed"
        tags.append(f"pre_timemix_p{p_str}_g{g_str}_a{a_str}_{split_tag}")

    # ---- PRE-encoding FullDimMix ----
    if getattr(args, "pre_fdmix_enable", False) and float(getattr(args, "pre_fdmix_p", 0.0)) > 0.0:
        p_str = str(getattr(args, "pre_fdmix_p", 0.0)).replace(".", "p")
        a_str = str(getattr(args, "pre_fdmix_alpha", 0.5)).replace(".", "p")
        tag = f"pre_fulldimmix_p{p_str}_a{a_str}_{str(getattr(args, 'pre_fdmix_layout', 'TBCHW'))}"
        if getattr(args, "pre_fdmix_apply_in_eval", False):
            tag += "_eval"
        tags.append(tag)

    # ---- PRE-encoding IPMix ----
    if getattr(args, "pre_ipmix_enable", False):
        tag = (
            f"pre_ipmix_k{int(getattr(args, 'pre_ipmix_k', 3))}"
            f"_t{int(getattr(args, 'pre_ipmix_t', 3))}"
            f"_sev{int(getattr(args, 'pre_ipmix_aug_severity', 3))}"
        )
        if getattr(args, "pre_ipmix_no_jsd", False):
            tag += "_nojsd"
        tags.append(tag)

    # ---- post-encoding patchdropout (kept AS-IS for compatibility) ----
    if getattr(args, "patchdrop_keep", 1.0) < 1.0:
        keep_str = str(args.patchdrop_keep).replace(".", "p")
        tags.append(f"postenc_patchdrop_k{keep_str}_ps{args.patchdrop_size}")

    # ---- post-encoding timeshuffle ----
    elif getattr(args, "tshift_p", 0.0) > 0.0:
        p_str = str(getattr(args, "tshift_p", 0.0)).replace(".", "p")
        a_str = str(getattr(args, "tshift_alpha", 0.3)).replace(".", "p")
        tags.append(f"timeshuffle_p{p_str}_max{args.tshift_max}_ck{args.tshift_fold_k}_a{a_str}")

    # ---- post-encoding timemix ----
    elif getattr(args, "timemix_p", 0.0) > 0.0:
        p_str = str(getattr(args, "timemix_p", 0.0)).replace(".", "p")
        a_str = str(getattr(args, "timemix_alpha", 0.5)).replace(".", "p")
        ck_str = str(getattr(args, "timemix_ck", 32))
        split = str(getattr(args, "timemix_split", "random"))
        tags.append(f"timemix_p{p_str}_ck{ck_str}_a{a_str}_{split}")

    # ---- post-encoding timemask ----
    elif getattr(args, "tmask_p", 0.0) > 0.0:
        p_str = str(getattr(args, "tmask_p", 0.0)).replace(".", "p")
        mf_str = str(getattr(args, "tmask_max_frac", 0.25)).replace(".", "p")
        tags.append(f"timemask_p{p_str}_n{args.tmask_num}_mf{mf_str}_ml{args.tmask_min_len}_{args.tmask_mode}")

    # ---- post-encoding localtimeshuffle ----
    elif getattr(args, "postenc_localtimeshuffle", False):
        p_str = str(getattr(args, "lts_p", 1.0)).replace(".", "p")
        tag = f"postenc_localtimeshuffle_p{p_str}_w{int(getattr(args, 'lts_window', 2))}_{str(getattr(args, 'lts_layout', 'TBCHW'))}"
        if getattr(args, "lts_per_sample", False):
            tag += "_persample"
        tags.append(tag)

    # ---- post-encoding holefill ----
    elif getattr(args, "postenc_holefill", False):
        p_str = str(getattr(args, "holefill_p", 1.0)).replace(".", "p")
        layout = str(getattr(args, "holefill_layout", "TBCHW"))
        mode = str(getattr(args, "holefill_mode", "spatiotemporal")).lower()
        tags.append(f"holefill_{mode}_p{p_str}_{layout}")

    # ---- post-encoding classbatchmix ----
    elif getattr(args, "postenc_cbmix", False):
        p_str = str(getattr(args, "cbmix_p", 0.5)).replace(".", "p")
        a_str = str(getattr(args, "cbmix_alpha", 0.4)).replace(".", "p")
        intra = str(getattr(args, "cbmix_intra", "time"))
        ps = int(getattr(args, "cbmix_patch_size", 4))
        tags.append(f"postenc_cbmix_p{p_str}_a{a_str}_{intra}_ps{ps}")

    # ---- post-encoding patchmix ----
    elif getattr(args, "postenc_patchmix", False):
        p_str = str(getattr(args, "pm_p", 0.5)).replace(".", "p")
        a_str = str(getattr(args, "pm_alpha", 0.2)).replace(".", "p")
        b_str = str(getattr(args, "pm_beta", 0.2)).replace(".", "p")
        tags.append(f"postenc_patchmix_p{p_str}_ps{args.pm_ps}_a{a_str}_b{b_str}")

    # ---- post-encoding patchshuffle ----
    elif getattr(args, "pshuf_p", 0.0) > 0.0:
        p_str = str(args.pshuf_p).replace(".", "p")
        tags.append(f"patchshuffle_p{p_str}_ps{args.pshuf_size}")

    if not tags:
        return "noaug"

    return "__".join(tags)


def _setup_run_dir(args) -> str:
    output_root = args.output if args.output else settings.CHECKPOINT_PATH
    os.makedirs(output_root, exist_ok=True)

    if args.experiment:
        run_name = _slugify(args.experiment)
    else:
        aug_tag = _build_aug_tag(args)
        run_name = _slugify(
            f"{aug_tag}_msresnet_{args.dataset}_{args.net}_seed{args.seed}_b{args.b}_lr{args.lr}"
        )

    run_dir = os.path.join(output_root, run_name)
    return run_dir


def _setup_logger(run_dir: str, is_rank0: bool) -> logging.Logger:
    logger = logging.getLogger("train_amp")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if len(logger.handlers) == 0:
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if is_rank0:
            os.makedirs(run_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(run_dir, "train.log"), mode="w")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def _log(msg: str, args):
    if args.local_rank == 0:
        if LOGGER is not None:
            LOGGER.info(msg)
        else:
            print(msg)


def _save_state_dict(path: str, model: torch.nn.Module):
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, path)


def _unwrap_output(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


# =========================
# IPMix helpers
# =========================

def _ipmix_dataset_mean_std(dataset_name: str):
    ds = str(dataset_name).lower()
    if ds == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif ds == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif ds == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unsupported dataset for IPMix mean/std: {dataset_name}")
    return mean, std


def _compute_ipmix_loss(outputs_all, labels, base_loss_fn, jsd_weight: float):
    bsz = labels.size(0)
    expected = 3 * bsz
    if outputs_all.size(0) != expected:
        raise ValueError(
            f"IPMix JSD path expects 3*B logits, got {outputs_all.size(0)} logits for batch size {bsz}."
        )

    logits_clean, logits_aug1, logits_aug2 = torch.split(outputs_all, bsz, dim=0)

    # keep existing train loss style (label smoothing)
    loss = base_loss_fn(logits_clean, labels)

    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)

    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1.0).log()

    jsd = (
        F.kl_div(p_mixture, p_clean, reduction="batchmean") +
        F.kl_div(p_mixture, p_aug1, reduction="batchmean") +
        F.kl_div(p_mixture, p_aug2, reduction="batchmean")
    ) / 3.0

    loss = loss + float(jsd_weight) * jsd
    return loss, logits_clean


class PreEncodingIPMix(nn.Module):
    """
    Pre-encoding IPMix for MS-ResNet training loop.

    Input:
        normalized tensor batch [B, C, H, W]

    Output:
        if no_jsd = False:
            (clean, aug1, aug2)
        else:
            mixed

    Notes:
        - training-only by default
        - uses PIL-based ops from IPMix_utils
        - keeps other augmentations intact
    """
    def __init__(
        self,
        dataset: str,
        mixing_set_root: str,
        k: int = 3,
        t: int = 3,
        aug_severity: int = 3,
        no_jsd: bool = False,
        jsd_weight: float = 12.0,
        img_size: int = 32,
        mixing_resize: int = 36,
        apply_in_eval: bool = False,
    ):
        super().__init__()

        if mixing_set_root is None or len(str(mixing_set_root)) == 0:
            raise ValueError("IPMix requires --pre-ipmix-mixing-set to be a valid ImageFolder path.")

        if not os.path.isdir(mixing_set_root):
            raise ValueError(f"IPMix mixing set path does not exist: {mixing_set_root}")

        self.dataset = str(dataset).lower()
        self.k = int(k)
        self.t = int(t)
        self.aug_severity = int(aug_severity)
        self.no_jsd = bool(no_jsd)
        self.jsd_weight = float(jsd_weight)
        self.img_size = int(img_size)
        self.mixing_resize = int(mixing_resize)
        self.apply_in_eval = bool(apply_in_eval)

        mean, std = _ipmix_dataset_mean_std(self.dataset)
        self.mean_list = mean
        self.std_list = std

        self.preprocess = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=self.mean_list, std=self.std_list),
        ])

        self.to_pil = tv_transforms.ToPILImage()

        mixing_transform = tv_transforms.Compose([
            tv_transforms.Resize(self.mixing_resize),
            tv_transforms.RandomCrop(self.img_size),
        ])
        self.mixing_set = tv_datasets.ImageFolder(
            root=mixing_set_root,
            transform=mixing_transform,
        )

        mean_t = torch.tensor(self.mean_list, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(self.std_list, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean_t)
        self.register_buffer("_std", std_t)

    def _ensure_pil_rgb(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("IPMix expects PIL images internally.")
        return image.convert("RGB")

    def _tensor_to_pil(self, x: torch.Tensor) -> Image.Image:
        # x is normalized tensor [C,H,W]
        with torch.no_grad():
            x_cpu = x.detach().float().cpu()
            mean = self._mean[0].detach().cpu()
            std = self._std[0].detach().cpu()
            x_cpu = x_cpu * std + mean
            x_cpu = torch.clamp(x_cpu, 0.0, 1.0)
            img = self.to_pil(x_cpu)
        return img.convert("RGB")

    def _augment_input(self, image: Image.Image):
        aug_list = ipmix_utils.augmentations_all
        op = random.choice(aug_list)
        return op(image.copy(), self.aug_severity)

    def _ipmix_one(self, image: Image.Image, mixing_pic: Image.Image) -> torch.Tensor:
        image = self._ensure_pil_rgb(image)
        mixing_pic = self._ensure_pil_rgb(mixing_pic)

        mixings = ipmix_utils.mixings
        patch_mixing = ipmix_utils.patch_mixing

        patch_sizes = [4, 8, 16, 32]
        patch_sizes = [p for p in patch_sizes if p <= self.img_size]
        if len(patch_sizes) == 0:
            patch_sizes = [self.img_size]

        mixing_op = ["IMG", "P"]

        ws = np.float32(np.random.dirichlet([1.0] * self.k))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(self.preprocess(image))

        for i in range(self.k):
            mixed = image.copy()
            mixing_way = random.choice(mixing_op)

            if mixing_way == "P":
                for _ in range(np.random.randint(self.t + 1)):
                    patch_size = int(random.choice(patch_sizes))
                    mixed_op = random.choice(mixings)
                    mixed = patch_mixing(mixed, mixing_pic, patch_size, mixed_op, beta=1)
            else:
                for _ in range(np.random.randint(self.t + 1)):
                    mixed = self._augment_input(mixed)

            mix = mix + ws[i] * self.preprocess(mixed)

        clean = self.preprocess(image)
        mix_result = (1.0 - m) * clean + m * mix
        return mix_result

    def forward(self, x: torch.Tensor):
        if (not self.training) and (not self.apply_in_eval):
            return x

        if x.dim() != 4:
            raise ValueError(f"PreEncodingIPMix expects BCHW input, got shape {tuple(x.shape)}")

        device = x.device
        dtype = x.dtype
        batch_size = x.size(0)

        if self.no_jsd:
            mixed_list = []
            for i in range(batch_size):
                base_img = self._tensor_to_pil(x[i])
                rnd_idx = random.randrange(len(self.mixing_set))
                mixing_pic, _ = self.mixing_set[rnd_idx]
                mixing_pic = self._ensure_pil_rgb(mixing_pic)
                mixed = self._ipmix_one(base_img, mixing_pic)
                mixed_list.append(mixed)
            mixed_batch = torch.stack(mixed_list, dim=0).to(device=device, dtype=dtype)
            return mixed_batch

        aug1_list = []
        aug2_list = []
        for i in range(batch_size):
            base_img = self._tensor_to_pil(x[i])
            rnd_idx = random.randrange(len(self.mixing_set))
            mixing_pic, _ = self.mixing_set[rnd_idx]
            mixing_pic = self._ensure_pil_rgb(mixing_pic)

            aug1 = self._ipmix_one(base_img, mixing_pic)
            aug2 = self._ipmix_one(base_img, mixing_pic)

            aug1_list.append(aug1)
            aug2_list.append(aug2)

        aug1_batch = torch.stack(aug1_list, dim=0).to(device=device, dtype=dtype)
        aug2_batch = torch.stack(aug2_list, dim=0).to(device=device, dtype=dtype)

        return (x, aug1_batch, aug2_batch)


# =========================
# Post-encoding aug helpers
# =========================

def _unwrap_ddp(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


class _PostEncLabelProxy(nn.Module):
    """
    MS-ResNet calls core.postenc_aug(x) (x-only).
    ClassBatchMix needs labels, so we store labels on the core each iteration:
        core._postenc_targets = labels

    IMPORTANT: Do NOT store `core` (nn.Module) as an attribute here,
    otherwise you create a circular module reference and SyncBatchNorm conversion recurses forever.
    """
    def __init__(self, aug: nn.Module, targets_attr: str = "_postenc_targets"):
        super().__init__()
        self.aug = aug
        self.targets_attr = targets_attr
        self.targets_getter = None  # callable, NOT a Module

    def forward(self, x):
        y = None
        if self.targets_getter is not None:
            y = self.targets_getter()

        try:
            return self.aug(x, y)
        except TypeError:
            return self.aug(x)


def _attach_postenc_aug(net: torch.nn.Module, aug: Optional[torch.nn.Module]):
    core = _unwrap_ddp(net)
    core.postenc_aug = aug

    if aug is not None and hasattr(aug, "targets_getter"):
        aug.targets_getter = (lambda c=core: getattr(c, "_postenc_targets", None))


def _set_postenc_aug_train_eval(net: torch.nn.Module, train_mode: bool):
    core = _unwrap_ddp(net)
    if hasattr(core, "postenc_aug") and core.postenc_aug is not None:
        core.postenc_aug.train(train_mode)


# ==============
# Train / Eval
# ==============

def train(epoch, args):
    running_loss = 0.0
    start = time.time()
    net.train()
    _set_postenc_aug_train_eval(net, True)

    if preenc_center_patch_minlift is not None and hasattr(preenc_center_patch_minlift, "train"):
        preenc_center_patch_minlift.train()

    if preenc_patchshuffle is not None and hasattr(preenc_patchshuffle, "train"):
        preenc_patchshuffle.train()

    if preenc_patchmix is not None and hasattr(preenc_patchmix, "train"):
        preenc_patchmix.train()

    if preenc_patchdropout is not None and hasattr(preenc_patchdropout, "train"):
        preenc_patchdropout.train()

    if preenc_ipmix is not None and hasattr(preenc_ipmix, "train"):
        preenc_ipmix.train()

    if fe_train is not None and hasattr(fe_train, "train"):
        fe_train.train()

    correct = 0.0
    num_sample = 0

    for batch_index, (images, labels) in enumerate(ImageNet_training_loader):
        if args.gpu:
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)

        # PRE-encoding Center Patch MinLift
        if preenc_center_patch_minlift is not None:
            images = preenc_center_patch_minlift(images)

        # PRE-encoding PatchShuffle
        if preenc_patchshuffle is not None:
            images = preenc_patchshuffle(images)

        # PRE-encoding PatchMix
        if preenc_patchmix is not None:
            try:
                images = preenc_patchmix(images)
            except ValueError as e:
                if batch_index == 0 or batch_index % 100 == 0:
                    _log(f"[AUG] PreEncPatchMix2D skipped on this batch: {e}", args)

        # PRE-encoding PatchDropout
        if preenc_patchdropout is not None:
            images = preenc_patchdropout(images)

        # PRE-encoding IPMix
        if preenc_ipmix is not None:
            images = preenc_ipmix(images)

        core = _unwrap_ddp(net)
        core._postenc_targets = labels

        batch_size_curr = images[0].size(0) if isinstance(images, (tuple, list)) else images.size(0)
        num_sample += batch_size_curr
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            if isinstance(images, (tuple, list)):
                merged_images = torch.cat(images, dim=0)
                outputs_all = _unwrap_output(net(merged_images))
                loss, outputs_for_acc = _compute_ipmix_loss(
                    outputs_all,
                    labels,
                    loss_function,
                    args.pre_ipmix_jsd_weight,
                )
                outputs = outputs_for_acc
            else:
                outputs = _unwrap_output(net(images))
                outputs_for_acc = outputs
                loss = loss_function(outputs, labels)

            _, preds = outputs_for_acc.max(1)
            correct += preds.eq(labels).sum()
            running_loss += float(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n_iter = (epoch - 1) * len(ImageNet_training_loader) + batch_index + 1
        if batch_index % 10 == 9:
            msg = (
                "Training Epoch: {epoch} [{trained_samples}/{total_samples}]  "
                "Loss: {loss:0.4f}  LR: {lr:0.6f}"
            ).format(
                epoch=epoch,
                trained_samples=batch_index * args.b + batch_size_curr,
                total_samples=len(ImageNet_training_loader.dataset),
                loss=running_loss / 10.0,
                lr=optimizer.param_groups[0]["lr"],
            )
            _log(msg, args)
            _log("training time consumed: {:.2f}s".format(time.time() - start), args)

            if args.local_rank == 0 and writer is not None:
                writer.add_scalar("Train/avg_loss", running_loss / 10.0, n_iter)
                writer.add_scalar("Train/avg_loss_numpic", running_loss / 10.0, n_iter * args.b)

            running_loss = 0.0

    finish = time.time()
    train_acc = (correct.float() / float(num_sample)) * 100.0

    if args.local_rank == 0 and writer is not None:
        writer.add_scalar("Train/acc", train_acc, epoch)

    _log("Training accuracy: {:.2f} of epoch {}".format(train_acc, epoch), args)
    _log("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start), args)


@torch.no_grad()
def eval_training(epoch, args):
    start = time.time()
    net.eval()
    _set_postenc_aug_train_eval(net, False)

    if preenc_center_patch_minlift is not None and hasattr(preenc_center_patch_minlift, "eval"):
        preenc_center_patch_minlift.eval()

    if preenc_patchshuffle is not None and hasattr(preenc_patchshuffle, "eval"):
        preenc_patchshuffle.eval()

    if preenc_patchmix is not None and hasattr(preenc_patchmix, "eval"):
        preenc_patchmix.eval()

    if preenc_patchdropout is not None and hasattr(preenc_patchdropout, "eval"):
        preenc_patchdropout.eval()

    if preenc_ipmix is not None and hasattr(preenc_ipmix, "eval"):
        preenc_ipmix.eval()

    if fe_eval is not None and hasattr(fe_eval, "eval"):
        fe_eval.eval()

    test_loss = 0.0
    correct = 0.0
    real_batch = 0

    for (images, labels) in ImageNet_test_loader:
        real_batch += images.size(0)
        if args.gpu:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # IMPORTANT:
        # Center Patch MinLift is TRAINING ONLY, so we do NOT apply it in eval.
        # PatchMix is TRAINING ONLY, so we do NOT apply preenc_patchmix in eval.
        # PatchDropout is TRAINING ONLY, so we do NOT apply preenc_patchdropout in eval.
        # PatchShuffle pre-encoding also remains disabled here as before by not applying it.
        # Pre-encoding TimeMask / TimeShuffle / TimeMix / Temporal Jitter / FullDimMix
        # are handled INSIDE the model and are disabled in eval by default unless their
        # apply-in-eval flags are set.
        # IPMix is TRAINING ONLY by default, so we do NOT apply preenc_ipmix in eval
        # unless --pre-ipmix-apply-in-eval is explicitly set inside the module.
        # Post-encoding LocalTimeShuffle is handled via core.postenc_aug and is also
        # disabled in eval by default unless --lts-apply-in-eval is set.
        # Frequency Encoding is handled via _set_input_encoder(...) before eval.
        outputs = _unwrap_output(net(images))
        loss = loss_function(outputs, labels)
        test_loss += float(loss.item())
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    avg_loss = test_loss * args.b / len(ImageNet_test_loader.dataset)
    acc_pct = correct.float() / float(real_batch) * 100.0

    _log("Evaluating Network.....", args)
    _log(
        "Test set: Average loss: {:.4f}, Accuracy: {:.4f}%, Time consumed:{:.2f}s".format(
            avg_loss, acc_pct, finish - start
        ),
        args,
    )

    if args.local_rank == 0 and writer is not None:
        writer.add_scalar("Test/Average loss", avg_loss, epoch)
        writer.add_scalar("Test/Accuracy", acc_pct, epoch)

    return correct.float() / float(len(ImageNet_test_loader.dataset))


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes=1000, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


# =====================
# Globals
# =====================
preenc_center_patch_minlift = None
preenc_patchshuffle = None
preenc_patchmix = None
preenc_patchdropout = None
preenc_ipmix = None
pre_temporaljitter = None
fe_train = None
fe_eval = None
net = None
optimizer = None
loss_function = None
scaler = None
writer = None
LOGGER = None
RUN_DIR = None
ImageNet_training_loader = None
ImageNet_test_loader = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument("-gpu", action="store_true", default=True, help="use gpu or not")
    parser.add_argument("-b", type=int, default=256, help="batch size for dataloader")
    parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "cifar10", "cifar100"])

    # seed + aug name
    parser.add_argument("--seed", type=int, default=445, help="random seed for reproducibility")
    parser.add_argument(
        "--aug-name",
        type=str,
        default="",
        help="name of augmentation technique. If empty, auto from args.",
    )

    # output + experiment naming
    parser.add_argument("--output", type=str, default="", help="Root folder for run folder.")
    parser.add_argument("--experiment", type=str, default="", help="Run folder name (EXP).")

    # PatchDropout args (kept AS-IS for compatibility)
    parser.add_argument("--patchdrop-keep", type=float, default=1.0, help="PatchDropout keep rate (1.0 disables).")
    parser.add_argument("--patchdrop-size", type=int, default=4, help="Patch size on feature map (e.g., 2, 4).")

    # TimeShuffle args
    parser.add_argument("--tshift-p", type=float, default=0.0, help="TimeShuffle probability (0 disables).")
    parser.add_argument("--tshift-max", type=int, default=1, help="Max temporal shift (>=1).")
    parser.add_argument("--tshift-fold-k", type=int, default=32, help="Channel folding factor (TS-SNN C_k).")
    parser.add_argument("--tshift-alpha", type=float, default=0.3, help="Residual penalty alpha (typ. 0.2..0.5).")

    # TimeMix args
    parser.add_argument("--timemix-p", type=float, default=0.0, help="TimeMix probability (0 disables).")
    parser.add_argument("--timemix-ck", type=int, default=32, help="Channel folding factor (groups).")
    parser.add_argument("--timemix-alpha", type=float, default=0.5, help="Residual penalty alpha.")
    parser.add_argument("--timemix-split", type=str, default="random", choices=["random", "fixed"],
                        help="How to choose split points for channel groups.")
    parser.add_argument("--timemix-fixed-g1", type=float, default=0.25, help="Fixed g1 fraction if split=fixed.")
    parser.add_argument("--timemix-fixed-g2", type=float, default=0.50, help="Fixed g2 fraction if split=fixed.")
    parser.add_argument("--timemix-learnable-alpha", action="store_true", help="Make alpha learnable.")
    parser.add_argument("--timemix-apply-in-eval", action="store_true", help="Apply in eval too (default: no).")
    parser.add_argument("--timemix-time-dim", type=int, default=-1,
                        help="-1/0 => TBCHW (MS-ResNet), 1 => BTCHW")

    # TimeMask args
    parser.add_argument("--tmask-p", type=float, default=0.0, help="TimeMask probability (0 disables).")
    parser.add_argument("--tmask-num", type=int, default=1, help="Number of masked intervals per sample.")
    parser.add_argument("--tmask-max-frac", type=float, default=0.25, help="Max fraction of T per mask interval.")
    parser.add_argument("--tmask-min-len", type=int, default=1, help="Min masked length (timesteps).")
    parser.add_argument("--tmask-mode", type=str, default="zero", choices=["zero", "noise"], help="Mask mode.")
    parser.add_argument("--tmask-noise-std", type=float, default=0.05, help="Noise std if mode=noise.")
    parser.add_argument("--tmask-layout", type=str, default="TB", choices=["BT", "TB"], help="Spike layout.")
    parser.add_argument("--tmask-same-on-batch", action="store_true", help="Use same mask for all batch samples.")
    parser.add_argument("--tmask-per-channel", action="store_true", help="Independent masking per channel-group.")
    parser.add_argument("--tmask-channel-groups", type=int, default=1, help="Number of channel groups.")

    # PatchMix args
    parser.add_argument("--postenc_patchmix", action="store_true", help="Enable post-encoding PatchMix")
    parser.add_argument("--pm_p", type=float, default=0.5, help="Patch-mix probability p")
    parser.add_argument("--pm_ps", type=int, default=16, help="Patch size (e.g., 16 for CIFAR 32x32)")
    parser.add_argument("--pm_alpha", type=float, default=0.2, help="Beta(alpha,beta) for lambda")
    parser.add_argument("--pm_beta", type=float, default=0.2, help="Beta(alpha,beta) for lambda")
    parser.add_argument("--pm_layout", type=str, default="TBCHW",
                        choices=["BTCHW", "TBCHW", "BCHW"],
                        help="Spike tensor layout for PatchMix")

    parser.add_argument('--postenc_cbmix', action='store_true',
                        help='Post-encoding ClassBatchMix (same-class across-batch mix; otherwise intra-sample mix).')
    parser.add_argument('--cbmix_p', type=float, default=0.5)
    parser.add_argument('--cbmix_alpha', type=float, default=0.4)
    parser.add_argument('--cbmix_intra', type=str, default='time', choices=['time', 'patch', 'both'])
    parser.add_argument('--cbmix_patch_size', type=int, default=4)
    parser.add_argument('--cbmix_binarize', action='store_true')

    parser.add_argument("--pm_no_mix_across_time", action="store_true",
                        help="If set: PatchMix is applied per timestep independently")
    parser.add_argument("--pm_no_pad", action="store_true",
                        help="If set: do NOT pad H/W to multiple of patch size")

    # PatchShuffle args
    parser.add_argument("--pshuf-p", type=float, default=0.0, help="PatchShuffle probability (0 disables).")
    parser.add_argument("--pshuf-size", type=int, default=4, help="Patch size (must divide H and W).")
    parser.add_argument("--pshuf-layout", type=str, default="TB", choices=["TB", "BT"],
                        help="Spike layout. MS-ResNet uses TB by default.")
    parser.add_argument("--pshuf-per-time", action="store_true",
                        help="If set: use different permutation per timestep (default: same over time).")
    parser.add_argument("--pshuf-same-on-batch", action="store_true",
                        help="If set: use same permutation for all samples in the batch.")

    # -----------------------------
    # Center Patch MinLift (pre-encoding, image-domain)
    # -----------------------------
    parser.add_argument(
        "--center-patch-minlift",
        action="store_true",
        default=False,
        help="Enable Center Patch MinLift pre-encoding augmentation.",
    )
    parser.add_argument(
        "--center-patch-minlift-patch-frac",
        type=float,
        default=0.5,
        help="Fraction of image height/width used for the center patch.",
    )
    parser.add_argument(
        "--center-patch-minlift-alpha-min",
        type=float,
        default=0.3,
        help="Minimum alpha for Center Patch MinLift.",
    )
    parser.add_argument(
        "--center-patch-minlift-alpha-max",
        type=float,
        default=0.7,
        help="Maximum alpha for Center Patch MinLift.",
    )
    parser.add_argument(
        "--center-patch-minlift-p",
        type=float,
        default=0.5,
        help="Probability of applying Center Patch MinLift.",
    )

    # ---- NEW: Frequency Encoding args ----
    parser.add_argument(
        "--fe",
        action="store_true",
        default=False,
        help="Enable Frequency Encoding augmentation for static image inputs."
    )
    parser.add_argument(
        "--fe-prob",
        type=float,
        default=1.0,
        help="Probability to apply FE per batch during training."
    )
    parser.add_argument(
        "--fe-jitter",
        type=int,
        default=0,
        help="Integer jitter added to radii, then sorted descending."
    )
    parser.add_argument(
        "--fe-radii",
        type=int,
        nargs="+",
        default=None,
        help="Explicit FE radii list. Example T=4 on CIFAR: 16 14 12 10"
    )
    parser.add_argument(
        "--fe-eval",
        action="store_true",
        default=False,
        help="Apply FE during evaluation too. Usually keep this False unless you explicitly want FE at test time."
    )

    # ---- [NEW] Post-encoding LocalTimeShuffle args ----
    parser.add_argument(
        "--postenc_localtimeshuffle",
        action="store_true",
        help="Enable post-encoding LocalTimeShuffle (windowed shuffle along time axis)."
    )
    parser.add_argument(
        "--lts-window",
        type=int,
        default=2,
        help="Window size for post-encoding LocalTimeShuffle."
    )
    parser.add_argument(
        "--lts-p",
        type=float,
        default=1.0,
        help="Apply probability for post-encoding LocalTimeShuffle."
    )
    parser.add_argument(
        "--lts-layout",
        type=str,
        default="TBCHW",
        choices=["TBCHW", "BTCHW"],
        help="Spike tensor layout for LocalTimeShuffle."
    )
    parser.add_argument(
        "--lts-per-sample",
        action="store_true",
        help="Use different local shuffle index per sample."
    )
    parser.add_argument(
        "--lts-apply-in-eval",
        action="store_true",
        help="Apply LocalTimeShuffle in eval too (default: no)."
    )

    # ---- PRE-encoding TimeMix args (mapped style you already use) ----
    parser.add_argument(
        "--preenc-timemix-prob",
        type=float,
        default=0.0,
        help="Probability of applying TS-SNN-style TimeMix before MS-ResNet stem."
    )
    parser.add_argument(
        "--preenc-timemix-alpha",
        type=float,
        default=0.3,
        help="Residual mixing strength alpha in out = x + alpha * shifted."
    )
    parser.add_argument(
        "--preenc-timemix-groups",
        type=int,
        default=32,
        help="Channel group count for TS-SNN-style TimeMix."
    )
    parser.add_argument(
        "--preenc-timemix-apply-in-eval",
        action="store_true",
        default=False,
        help="Apply pre-encoding TimeMix during evaluation too."
    )
    # two flags so default can stay True cleanly in Python 3.7 argparse
    parser.add_argument(
        "--preenc-timemix-random-split",
        dest="preenc_timemix_random_split",
        action="store_true",
        help="Use random split points g1, g2 each forward."
    )
    parser.add_argument(
        "--preenc-timemix-fixed-split",
        dest="preenc_timemix_random_split",
        action="store_false",
        help="Use deterministic split points instead of random ones."
    )
    parser.set_defaults(preenc_timemix_random_split=True)

    # ---- PRE-encoding FullDimMix args ----
    parser.add_argument(
        "--preenc-fdmix-p",
        type=float,
        default=0.0,
        help="Probability of applying pre-encoding FullDimMix on 5D input."
    )
    parser.add_argument(
        "--preenc-fdmix-alpha",
        type=float,
        default=0.5,
        help="Blend factor for FullDimMix: y = alpha*x + (1-alpha)*x_perm."
    )
    parser.add_argument(
        "--preenc-fdmix-layout",
        type=str,
        default="TBCHW",
        choices=["TBCHW", "BTCHW"],
        help="5D tensor layout for pre-encoding FullDimMix."
    )
    parser.add_argument(
        "--preenc-fdmix-apply-in-eval",
        action="store_true",
        default=False,
        help="Apply pre-encoding FullDimMix during evaluation too."
    )

    # === Pre-encoding PatchDropout arguments ===
    parser.add_argument(
        "--preenc-patchdropout",
        action="store_true",
        default=False,
        help="Enable PatchDropout as a pre-encoding augmentation during TRAINING only.",
    )
    parser.add_argument(
        "--pd-keep",
        type=float,
        default=1.0,
        help="Fixed keep rate for PatchDropout (0 < keep <= 1).",
    )
    parser.add_argument(
        "--pd-min-keep",
        type=float,
        default=None,
        help="If set, keep rate is sampled uniformly from [pd_min_keep, 1.0] each step.",
    )
    parser.add_argument(
        "--pd-patch-h",
        type=int,
        default=4,
        help="Patch height for pre-encoding PatchDropout.",
    )
    parser.add_argument(
        "--pd-patch-w",
        type=int,
        default=None,
        help="Patch width for pre-encoding PatchDropout. Defaults to pd_patch_h.",
    )
    # === End Pre-encoding PatchDropout arguments ===

    # === Pre-encoding PatchMix arguments ===
    parser.add_argument(
        "--pre-pmix-enable",
        action="store_true",
        help="Enable PRE-encoding PatchMix on input images during TRAINING only."
    )
    parser.add_argument(
        "--pre-pmix-p",
        type=float,
        default=0.0,
        help="Pre-encoding PatchMix probability p."
    )
    parser.add_argument(
        "--pre-pmix-size",
        type=int,
        default=16,
        help="Pre-encoding PatchMix patch size."
    )
    parser.add_argument(
        "--pre-pmix-alpha",
        type=float,
        default=0.2,
        help="Alpha for Beta(alpha, beta) in pre-encoding PatchMix."
    )
    parser.add_argument(
        "--pre-pmix-beta",
        type=float,
        default=0.2,
        help="Beta for Beta(alpha, beta) in pre-encoding PatchMix."
    )
    parser.add_argument(
        "--pre-pmix-no-pad",
        action="store_true",
        help="If set, require H/W divisible by patch size instead of padding to a multiple."
    )
    # === End Pre-encoding PatchMix arguments ===

    # === NEW: Pre-encoding TimeMask arguments ===
    parser.add_argument(
        "--pre-tmask-enable",
        action="store_true",
        help="Enable PRE-encoding TimeMask inside MS-ResNet before the first conv."
    )
    parser.add_argument(
        "--pre-tmask-p",
        type=float,
        default=0.0,
        help="Pre-encoding TimeMask probability."
    )
    parser.add_argument(
        "--pre-tmask-num",
        type=int,
        default=1,
        help="Number of temporal mask intervals per sample."
    )
    parser.add_argument(
        "--pre-tmask-max-frac",
        type=float,
        default=0.25,
        help="Maximum masked fraction of T per interval."
    )
    parser.add_argument(
        "--pre-tmask-min-len",
        type=int,
        default=1,
        help="Minimum masked temporal length."
    )
    parser.add_argument(
        "--pre-tmask-mode",
        type=str,
        default="zero",
        choices=["zero", "noise"],
        help="Mask mode for pre-encoding TimeMask."
    )
    parser.add_argument(
        "--pre-tmask-noise-std",
        type=float,
        default=0.05,
        help="Noise std if pre-tmask-mode=noise."
    )
    parser.add_argument(
        "--pre-tmask-layout",
        type=str,
        default="TBCHW",
        choices=["TBCHW", "BTCHW"],
        help="Temporal tensor layout used inside the model."
    )
    parser.add_argument(
        "--pre-tmask-same-on-batch",
        action="store_true",
        help="Use same temporal mask for all samples in the batch."
    )
    parser.add_argument(
        "--pre-tmask-per-channel",
        action="store_true",
        help="Apply independent masks per channel-group."
    )
    parser.add_argument(
        "--pre-tmask-channel-groups",
        type=int,
        default=1,
        help="Number of channel groups for pre-encoding TimeMask."
    )
    parser.add_argument(
        "--pre-tmask-apply-in-eval",
        action="store_true",
        help="Apply pre-encoding TimeMask in eval too (default: no)."
    )
    # === End PRE-encoding TimeMask arguments ===

    # === NEW: Pre-encoding TimeShuffle arguments ===
    parser.add_argument(
        "--pre-tshift-enable",
        action="store_true",
        help="Enable PRE-encoding TimeShuffle inside MS-ResNet before the first conv."
    )
    parser.add_argument(
        "--pre-tshift-p",
        type=float,
        default=0.0,
        help="Pre-encoding TimeShuffle probability."
    )
    parser.add_argument(
        "--pre-tshift-max",
        type=int,
        default=1,
        help="Maximum temporal shift for pre-encoding TimeShuffle."
    )
    parser.add_argument(
        "--pre-tshift-foldk",
        type=int,
        default=32,
        help="Channel folding factor Ck for pre-encoding TimeShuffle."
    )
    parser.add_argument(
        "--pre-tshift-alpha",
        type=float,
        default=0.3,
        help="Residual alpha for pre-encoding TimeShuffle."
    )
    parser.add_argument(
        "--pre-tshift-apply-in-eval",
        action="store_true",
        help="Apply pre-encoding TimeShuffle in eval too (default: no)."
    )
    # === End PRE-encoding TimeShuffle arguments ===

    # === NEW: Pre-encoding Temporal Jitter arguments ===
    parser.add_argument(
        "--pre-tjitter-enable",
        action="store_true",
        help="Enable PRE-encoding Temporal Jitter (circular temporal shift) inside MS-ResNet before the first conv."
    )
    parser.add_argument(
        "--pre-tjitter-p",
        type=float,
        default=0.0,
        help="Pre-encoding Temporal Jitter probability."
    )
    parser.add_argument(
        "--pre-tjitter-max",
        type=int,
        default=1,
        help="Maximum absolute temporal shift sampled from [-max, +max]."
    )
    parser.add_argument(
        "--pre-tjitter-per-sample",
        action="store_true",
        help="Use a different temporal shift for each sample in the batch."
    )
    parser.add_argument(
        "--pre-tjitter-layout",
        type=str,
        default="TBCHW",
        choices=["TBCHW", "BTCHW"],
        help="Temporal tensor layout for PRE-encoding Temporal Jitter."
    )
    parser.add_argument(
        "--pre-tjitter-apply-in-eval",
        action="store_true",
        help="Apply pre-encoding Temporal Jitter in eval too (default: no)."
    )
    # === End PRE-encoding Temporal Jitter arguments ===

    # === NEW: Pre-encoding IPMix arguments ===
    parser.add_argument(
        "--pre-ipmix-enable",
        action="store_true",
        help="Enable PRE-encoding IPMix on input images during TRAINING only."
    )
    parser.add_argument(
        "--pre-ipmix-mixing-set",
        type=str,
        default="",
        help="Path to IPMix mixing set (ImageFolder)."
    )
    parser.add_argument(
        "--pre-ipmix-k",
        type=int,
        default=3,
        help="Dirichlet components for IPMix."
    )
    parser.add_argument(
        "--pre-ipmix-t",
        type=int,
        default=3,
        help="Maximum number of ops per IMG/P chain in IPMix."
    )
    parser.add_argument(
        "--pre-ipmix-aug-severity",
        type=int,
        default=3,
        help="Severity for image-level ops in IPMix."
    )
    parser.add_argument(
        "--pre-ipmix-no-jsd",
        action="store_true",
        help="Disable JSD term and return a single IPMix view."
    )
    parser.add_argument(
        "--pre-ipmix-jsd-weight",
        type=float,
        default=12.0,
        help="Weight of the JSD term for IPMix."
    )
    parser.add_argument(
        "--pre-ipmix-img-size",
        type=int,
        default=32,
        help="Image size used by IPMix."
    )
    parser.add_argument(
        "--pre-ipmix-mixing-resize",
        type=int,
        default=36,
        help="Resize before RandomCrop for IPMix mixing-set images."
    )
    parser.add_argument(
        "--pre-ipmix-apply-in-eval",
        action="store_true",
        help="Apply pre-encoding IPMix in eval too (default: no)."
    )
    # === End Pre-encoding IPMix arguments ===

    # Pre-encoding PatchShuffle args
    parser.add_argument(
        "--pre-pshuf-enable",
        action="store_true",
        help="Enable PRE-encoding PatchShuffle on input images."
    )
    parser.add_argument(
        "--pre-pshuf-p",
        type=float,
        default=0.0,
        help="Pre-encoding PatchShuffle probability."
    )
    parser.add_argument(
        "--pre-pshuf-size",
        type=int,
        default=2,
        help="Pre-encoding PatchShuffle patch size."
    )
    parser.add_argument(
        "--pre-pshuf-same-on-batch",
        action="store_true",
        help="Use the same patch permutation field for the whole batch."
    )
    parser.add_argument("--traindir", type=str, default="/data/imagenet/train")
    parser.add_argument("--valdir", type=str, default="/data/imagenet/val")
    
        # [NEW] HoleFill args
    parser.add_argument("--postenc_holefill", action="store_true",
                        help="Enable post-encoding HoleFill.")
    parser.add_argument("--holefill_p", type=float, default=1.0,
                        help="HoleFill apply probability (train only unless eval flag is set).")
    parser.add_argument("--holefill_mode", type=str, default="spatiotemporal",
                        choices=["spatial", "spatiotemporal"],
                        help="HoleFill mode: spatial or spatiotemporal.")
    parser.add_argument("--holefill_layout", type=str, default="TBCHW",
                        choices=["TBCHW", "BTCHW", "BCHW"],
                        help="Spike layout for HoleFill. MS-ResNet default is TBCHW.")
    parser.add_argument("--holefill_apply_in_eval", action="store_true",
                        help="Apply HoleFill in eval too (default: no).")
    parser.add_argument(
    "--data-root",
    "--data_root",
    dest="data_root",
    type=str,
    default="/media/homes/bist/data"
    )
    args = parser.parse_args()

    # -------------------------
    # Map CLI preenc-timemix args to the names expected by MS_ResNet.py / get_network(...)
    # -------------------------
    args.pre_tmix_enable = float(args.preenc_timemix_prob) > 0.0
    args.pre_tmix_p = float(args.preenc_timemix_prob)
    args.pre_tmix_alpha = float(args.preenc_timemix_alpha)
    args.pre_tmix_groups = int(args.preenc_timemix_groups)
    args.pre_tmix_random_split = bool(args.preenc_timemix_random_split)
    args.pre_tmix_apply_in_eval = bool(args.preenc_timemix_apply_in_eval)

    # -------------------------
    # Map CLI preenc-fdmix args to the names expected by MS_ResNet.py / get_network(...)
    # -------------------------
    args.pre_fdmix_enable = float(args.preenc_fdmix_p) > 0.0
    args.pre_fdmix_p = float(args.preenc_fdmix_p)
    args.pre_fdmix_alpha = float(args.preenc_fdmix_alpha)
    args.pre_fdmix_layout = str(args.preenc_fdmix_layout)
    args.pre_fdmix_apply_in_eval = bool(args.preenc_fdmix_apply_in_eval)

    # -------------------------
    # Normalize pre-encoding TimeShuffle args so get_network(args)
    # can forward them directly into MS_ResNet.py
    # -------------------------
    args.pre_tshift_enable = bool(args.pre_tshift_enable or (float(args.pre_tshift_p) > 0.0))
    args.pre_tshift_p = float(args.pre_tshift_p)
    args.pre_tshift_max = int(args.pre_tshift_max)
    args.pre_tshift_foldk = int(args.pre_tshift_foldk)
    args.pre_tshift_alpha = float(args.pre_tshift_alpha)
    args.pre_tshift_apply_in_eval = bool(args.pre_tshift_apply_in_eval)

    # -------------------------
    # Normalize pre-encoding Temporal Jitter args
    # -------------------------
    args.pre_tjitter_enable = bool(args.pre_tjitter_enable or (float(args.pre_tjitter_p) > 0.0))
    args.pre_tjitter_p = float(args.pre_tjitter_p)
    args.pre_tjitter_max = int(args.pre_tjitter_max)
    args.pre_tjitter_per_sample = bool(args.pre_tjitter_per_sample)
    args.pre_tjitter_layout = str(args.pre_tjitter_layout)
    args.pre_tjitter_apply_in_eval = bool(args.pre_tjitter_apply_in_eval)

    args.cifar_stem = args.dataset in ["cifar10", "cifar100"]

    if args.dataset == "imagenet":
        args.num_classes = 1000
    elif args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100

    # DDP init
    if args.local_rank is None or args.local_rank < 0:
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)

    # run folder + logger + TB
    RUN_DIR = _setup_run_dir(args)
    is_rank0 = (args.local_rank == 0)
    LOGGER = _setup_logger(RUN_DIR, is_rank0)

    if is_rank0:
        os.makedirs(RUN_DIR, exist_ok=True)
        with open(os.path.join(RUN_DIR, "args.txt"), "w") as f:
            for k, v in sorted(vars(args).items()):
                f.write(f"{k}: {v}\n")

        writer = SummaryWriter(log_dir=os.path.join(RUN_DIR, "tb"))
        _log(f"Run directory: {RUN_DIR}", args)

    # seed (rank-offset)
    base_seed = args.seed
    seed = base_seed + (args.local_rank if args.local_rank is not None and args.local_rank >= 0 else 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model
    net = get_network(args)

    # =========================
    # Optional Frequency Encoding
    # =========================
    fe_train = None
    fe_eval = None

    if args.fe:
        model_for_cfg = _unwrap_model(net)

        # try to infer time steps robustly
        time_steps = int(
            getattr(
                model_for_cfg,
                "T",
                getattr(
                    model_for_cfg,
                    "time_window",
                    6
                ),
            )
        )

        # choose image size robustly
        if hasattr(args, "pre_ipmix_img_size") and args.pre_ipmix_img_size is not None and args.dataset in ["cifar10", "cifar100"]:
            img_size = int(args.pre_ipmix_img_size)
        else:
            img_size = 32 if args.dataset in ["cifar10", "cifar100"] else 224

        radii = args.fe_radii
        if radii is None:
            radii = make_default_fe_radii(img_size, img_size, time_steps)

        if len(radii) != time_steps:
            raise ValueError(
                "Length of --fe-radii ({}) must match model T/time_window ({})."
                .format(len(radii), time_steps)
            )

        fe_train = FrequencyEncodingAug(
            T=time_steps,
            radii=radii,
            p=args.fe_prob,
            jitter=args.fe_jitter,
            return_layout="TBCHW",
        )

        if args.fe_eval:
            fe_eval = FrequencyEncodingAug(
                T=time_steps,
                radii=radii,
                p=1.0,
                jitter=0,
                return_layout="TBCHW",
            )

        if is_rank0:
            _log(
                f"[AUG] FrequencyEncoding enabled: "
                f"{{'T': {time_steps}, 'img_size': {img_size}, 'radii': {radii}, "
                f"'fe_prob': {args.fe_prob}, 'fe_jitter': {args.fe_jitter}, "
                f"'fe_eval': {args.fe_eval}}}",
                args
            )

    if is_rank0 and getattr(args, "center_patch_minlift", False) and float(getattr(args, "center_patch_minlift_p", 0.0)) > 0.0:
        _log(
            f"[AUG] CenterPatchMinLift requested: "
            f"{{'patch_frac': {args.center_patch_minlift_patch_frac}, "
            f"'alpha_min': {args.center_patch_minlift_alpha_min}, "
            f"'alpha_max': {args.center_patch_minlift_alpha_max}, "
            f"'p': {args.center_patch_minlift_p}}}",
            args
        )

    if is_rank0 and getattr(args, "pre_tshift_enable", False) and float(getattr(args, "pre_tshift_p", 0.0)) > 0.0:
        _log(
            f"[AUG] PreEncodingTimeShuffle enabled inside model: "
            f"{{'p': {args.pre_tshift_p}, "
            f"'max_shift': {args.pre_tshift_max}, "
            f"'foldk': {args.pre_tshift_foldk}, "
            f"'alpha': {args.pre_tshift_alpha}, "
            f"'apply_in_eval': {args.pre_tshift_apply_in_eval}}}",
            args
        )

    if is_rank0 and getattr(args, "pre_tjitter_enable", False) and float(getattr(args, "pre_tjitter_p", 0.0)) > 0.0:
        _log(
            f"[AUG] PreTemporalJitter enabled: "
            f"{{'p': {args.pre_tjitter_p}, "
            f"'max_shift': {args.pre_tjitter_max}, "
            f"'per_sample': {args.pre_tjitter_per_sample}, "
            f"'layout': '{args.pre_tjitter_layout}', "
            f"'apply_in_eval': {args.pre_tjitter_apply_in_eval}}}",
            args
        )

    if is_rank0 and getattr(args, "pre_tmix_enable", False):
        _log(
            f"[AUG] PreEncodingTimeMix enabled inside model: "
            f"{{'p': {args.pre_tmix_p}, "
            f"'alpha': {args.pre_tmix_alpha}, "
            f"'groups': {args.pre_tmix_groups}, "
            f"'random_split': {args.pre_tmix_random_split}, "
            f"'apply_in_eval': {args.pre_tmix_apply_in_eval}}}",
            args
        )

    if is_rank0 and getattr(args, "pre_fdmix_enable", False):
        _log(
            f"[AUG] PreEncFullDimMix enabled inside model: "
            f"{{'p': {args.pre_fdmix_p}, "
            f"'alpha': {args.pre_fdmix_alpha}, "
            f"'layout': '{args.pre_fdmix_layout}', "
            f"'apply_in_eval': {args.pre_fdmix_apply_in_eval}}}",
            args
        )

    # =========================
    # Optional PRE-encoding Center Patch MinLift
    # =========================
    preenc_center_patch_minlift = None
    if args.center_patch_minlift and args.center_patch_minlift_p > 0.0:
        preenc_center_patch_minlift = CenterPatchMinLift(
            patch_frac=float(args.center_patch_minlift_patch_frac),
            alpha_min=float(args.center_patch_minlift_alpha_min),
            alpha_max=float(args.center_patch_minlift_alpha_max),
            p=float(args.center_patch_minlift_p),
            inplace=False,
        )

        if args.local_rank == 0:
            _log(
                f"[AUG] CenterPatchMinLift enabled: "
                f"{{'patch_frac': {args.center_patch_minlift_patch_frac}, "
                f"'alpha_min': {args.center_patch_minlift_alpha_min}, "
                f"'alpha_max': {args.center_patch_minlift_alpha_max}, "
                f"'p': {args.center_patch_minlift_p}}}",
                args
            )

    # =========================
    # Optional PRE-encoding PatchDropout
    # =========================
    preenc_patchdropout = None
    if args.preenc_patchdropout:
        preenc_patchdropout = PreEncPatchDropout2D(
            patch_h=args.pd_patch_h,
            patch_w=args.pd_patch_w,
            keep=args.pd_keep,
            min_keep=args.pd_min_keep,
        )

        if args.local_rank == 0:
            if args.pd_min_keep is not None:
                _log(
                    f"[AUG] PreEncPatchDropout2D enabled: "
                    f"{{'keep': 'U[{args.pd_min_keep}, 1.0]', "
                    f"'patch_h': {args.pd_patch_h}, "
                    f"'patch_w': {args.pd_patch_w if args.pd_patch_w is not None else args.pd_patch_h}}}",
                    args
                )
            else:
                _log(
                    f"[AUG] PreEncPatchDropout2D enabled: "
                    f"{{'keep': {args.pd_keep}, "
                    f"'patch_h': {args.pd_patch_h}, "
                    f"'patch_w': {args.pd_patch_w if args.pd_patch_w is not None else args.pd_patch_h}}}",
                    args
                )

    # =========================
    # Optional PRE-encoding IPMix
    # =========================
    preenc_ipmix = None
    if args.pre_ipmix_enable:
        if args.dataset not in ["cifar10", "cifar100"]:
            raise ValueError("PRE-encoding IPMix is currently intended for cifar10/cifar100 in this MS-ResNet setup.")

        preenc_ipmix = PreEncodingIPMix(
            dataset=args.dataset,
            mixing_set_root=args.pre_ipmix_mixing_set,
            k=args.pre_ipmix_k,
            t=args.pre_ipmix_t,
            aug_severity=args.pre_ipmix_aug_severity,
            no_jsd=args.pre_ipmix_no_jsd,
            jsd_weight=args.pre_ipmix_jsd_weight,
            img_size=args.pre_ipmix_img_size,
            mixing_resize=args.pre_ipmix_mixing_resize,
            apply_in_eval=args.pre_ipmix_apply_in_eval,
        )

    # =========================
    # Optional PRE-encoding PatchMix
    # =========================
    preenc_patchmix = None
    if args.pre_pmix_enable and args.pre_pmix_p > 0.0:
        preenc_patchmix = PreEncodingPatchMix2D(
            p=float(args.pre_pmix_p),
            patch_size=int(args.pre_pmix_size),
            alpha=float(args.pre_pmix_alpha),
            beta=float(args.pre_pmix_beta),
            pad_to_multiple=not bool(args.pre_pmix_no_pad),
        )

    # =========================
    # Optional PRE-encoding PatchShuffle
    # =========================
    preenc_patchshuffle = None
    if args.pre_pshuf_enable and args.pre_pshuf_p > 0.0:
        preenc_patchshuffle = PreEncodingPatchShuffle2D(
            p=float(args.pre_pshuf_p),
            patch_size=int(args.pre_pshuf_size),
            same_perm_on_batch=bool(args.pre_pshuf_same_on_batch),
        )

    # =========================
    # Optional PRE-encoding Temporal Jitter
    # =========================
    pre_temporaljitter = None
    if args.pre_tjitter_enable and args.pre_tjitter_p > 0.0:
        try:
            pre_temporaljitter = PreTemporalJitter(
                p=float(args.pre_tjitter_p),
                max_shift=int(args.pre_tjitter_max),
                per_sample=bool(args.pre_tjitter_per_sample),
                layout=str(args.pre_tjitter_layout),
                apply_in_eval=bool(args.pre_tjitter_apply_in_eval),
            )
        except TypeError:
            # fallback in case your class does not expose apply_in_eval in __init__
            pre_temporaljitter = PreTemporalJitter(
                p=float(args.pre_tjitter_p),
                max_shift=int(args.pre_tjitter_max),
                per_sample=bool(args.pre_tjitter_per_sample),
                layout=str(args.pre_tjitter_layout),
            )

    # =========================
    # Build ONE post-encoding augmentation (avoid silent overwrite)
    # =========================
    enabled = []
    if getattr(args, "postenc_patchmix", False):
        enabled.append("patchmix")
    if float(getattr(args, "pshuf_p", 0.0)) > 0.0:
        enabled.append("patchshuffle")
    if float(getattr(args, "tmask_p", 0.0)) > 0.0:
        enabled.append("timemask")
    if float(getattr(args, "timemix_p", 0.0)) > 0.0:
        enabled.append("timemix")
    if getattr(args, "postenc_cbmix", False):
        enabled.append("cbmix")
    if getattr(args, "postenc_localtimeshuffle", False):
        enabled.append("localtimeshuffle")
    if getattr(args, "postenc_holefill", False):
        enabled.append("holefill")

    if len(enabled) > 1:
        raise ValueError(f"Enable only ONE post-encoding augmentation at a time. Got: {enabled}")

    postenc_aug = None
    postenc_cfg = None
    postenc_name = None

    # ---- PatchMix ----
    if "patchmix" in enabled:
        postenc_name = "PostEncPatchMix"
        postenc_cfg = {
            "p": args.pm_p,
            "patch_size": args.pm_ps,
            "alpha": args.pm_alpha,
            "beta": args.pm_beta,
            "layout": args.pm_layout,
            "mix_across_time": (not args.pm_no_mix_across_time),
            "pad_to_multiple": (not args.pm_no_pad),
        }
        postenc_aug = PostEncPatchMix(
            patch_size=int(postenc_cfg["patch_size"]),
            p=float(postenc_cfg["p"]),
            alpha=float(postenc_cfg["alpha"]),
            beta=float(postenc_cfg["beta"]),
            layout=str(postenc_cfg["layout"]),
            mix_across_time=bool(postenc_cfg["mix_across_time"]),
            pad_to_multiple=bool(postenc_cfg["pad_to_multiple"]),
        )

    # ---- PatchShuffle ----
    elif "patchshuffle" in enabled:
        postenc_name = "PatchShufflePostEncoding2D"
        pshuf_cfg = {
            "p": getattr(args, "pshuf_p", 0.0),
            "patch_size": getattr(args, "pshuf_size", 4),
            "layout": getattr(args, "pshuf_layout", "TB"),
            "same_on_time": (not getattr(args, "pshuf_per_time", False)),
            "same_perm_on_batch": getattr(args, "pshuf_same_on_batch", False),
        }
        postenc_cfg = pshuf_cfg
        postenc_aug = PatchShufflePostEncoding2D(
            p=float(pshuf_cfg["p"]),
            patch_size=int(pshuf_cfg["patch_size"]),
            layout=str(pshuf_cfg["layout"]),
            same_on_time=bool(pshuf_cfg["same_on_time"]),
            same_perm_on_batch=bool(pshuf_cfg["same_perm_on_batch"]),
        )

    # ---- TimeMask ----
    elif "timemask" in enabled:
        postenc_name = "TimeMaskPostEncoding"
        tmask_cfg = getattr(net, "tmask_cfg", None)
        if tmask_cfg is None:
            tmask_cfg = {
                "p": getattr(args, "tmask_p", 0.0),
                "num_masks": getattr(args, "tmask_num", 1),
                "max_mask_frac": getattr(args, "tmask_max_frac", 0.25),
                "min_mask_len": getattr(args, "tmask_min_len", 1),
                "mode": getattr(args, "tmask_mode", "zero"),
                "noise_std": getattr(args, "tmask_noise_std", 0.05),
                "layout": getattr(args, "tmask_layout", "TB"),
                "same_on_batch": getattr(args, "tmask_same_on_batch", False),
                "per_channel": getattr(args, "tmask_per_channel", False),
                "channel_groups": getattr(args, "tmask_channel_groups", 1),
            }
        postenc_cfg = tmask_cfg
        postenc_aug = TimeMaskPostEncoding(
            p=float(tmask_cfg["p"]),
            num_masks=int(tmask_cfg["num_masks"]),
            max_mask_frac=float(tmask_cfg["max_mask_frac"]),
            min_mask_len=int(tmask_cfg["min_mask_len"]),
            mode=str(tmask_cfg["mode"]),
            noise_std=float(tmask_cfg["noise_std"]),
            layout=str(tmask_cfg["layout"]),
            same_on_batch=bool(tmask_cfg["same_on_batch"]),
            per_channel=bool(tmask_cfg["per_channel"]),
            channel_groups=int(tmask_cfg["channel_groups"]),
        )

    # ---- TimeMix ----
    elif "timemix" in enabled:
        td = int(getattr(args, "timemix_time_dim", -1))
        timemix_layout = "BTCHW" if td == 1 else "TBCHW"

        postenc_name = "PostEncTimeMix"
        postenc_cfg = {
            "p": float(getattr(args, "timemix_p", 0.0)),
            "ck": int(getattr(args, "timemix_ck", 32)),
            "alpha": float(getattr(args, "timemix_alpha", 0.5)),
            "split": str(getattr(args, "timemix_split", "random")),
            "fixed_g1": float(getattr(args, "timemix_fixed_g1", 0.25)),
            "fixed_g2": float(getattr(args, "timemix_fixed_g2", 0.50)),
            "learnable_alpha": bool(getattr(args, "timemix_learnable_alpha", False)),
            "apply_in_eval": bool(getattr(args, "timemix_apply_in_eval", False)),
            "layout": timemix_layout,
        }
        cfg = TimeMixConfig(
            p=postenc_cfg["p"],
            ck=postenc_cfg["ck"],
            alpha=postenc_cfg["alpha"],
            split=postenc_cfg["split"],
            fixed_g1=postenc_cfg["fixed_g1"],
            fixed_g2=postenc_cfg["fixed_g2"],
            learnable_alpha=postenc_cfg["learnable_alpha"],
            apply_in_eval=postenc_cfg["apply_in_eval"],
            layout=postenc_cfg["layout"],
        )
        postenc_aug = PostEncTimeMix(cfg)

    # ---- ClassBatchMix (label-aware) ----
    elif "cbmix" in enabled:
        postenc_name = "ClassBatchMixPostEncoding"
        cb_cfg = {
            "p": float(getattr(args, "cbmix_p", 0.5)),
            "alpha": float(getattr(args, "cbmix_alpha", 0.4)),
            "intra": str(getattr(args, "cbmix_intra", "time")),
            "patch_size": int(getattr(args, "cbmix_patch_size", 4)),
            "binarize": bool(getattr(args, "cbmix_binarize", False)),
        }
        postenc_cfg = cb_cfg
        cb_aug = ClassBatchMixPostEncoding(
            p=cb_cfg["p"],
            alpha=cb_cfg["alpha"],
            intra=cb_cfg["intra"],
            patch_size=cb_cfg["patch_size"],
            binarize=cb_cfg["binarize"],
        )
        postenc_aug = _PostEncLabelProxy(cb_aug)

    # ---- LocalTimeShuffle ----
    elif "localtimeshuffle" in enabled:
        postenc_name = "LocalTimeShufflePostEncoding"
        lts_cfg = {
            "window_size": int(getattr(args, "lts_window", 2)),
            "p": float(getattr(args, "lts_p", 1.0)),
            "layout": str(getattr(args, "lts_layout", "TBCHW")),
            "per_sample": bool(getattr(args, "lts_per_sample", False)),
            "apply_in_eval": bool(getattr(args, "lts_apply_in_eval", False)),
        }
        postenc_cfg = lts_cfg
        postenc_aug = LocalTimeShufflePostEncoding(
            window_size=lts_cfg["window_size"],
            p=lts_cfg["p"],
            layout=lts_cfg["layout"],
            per_sample=lts_cfg["per_sample"],
            apply_in_eval=lts_cfg["apply_in_eval"],
        )

    # ---- HoleFill ----
    # ---- HoleFill ----
    elif "holefill" in enabled:
        postenc_name = "HoleFillPostEncoding"
        hf_cfg = {
            "p": float(getattr(args, "holefill_p", 1.0)),
            "mode": str(getattr(args, "holefill_mode", "spatiotemporal")).lower(),
            "layout": str(getattr(args, "holefill_layout", "TBCHW")),
            "apply_in_eval": bool(getattr(args, "holefill_apply_in_eval", False)),
        }
        postenc_cfg = hf_cfg
        postenc_aug = HoleFillPostEncoding(
            p=hf_cfg["p"],
            mode=hf_cfg["mode"],
            layout=hf_cfg["layout"],
            apply_in_eval=hf_cfg["apply_in_eval"],
        )

    # Move to GPU
    net.cuda()
    if postenc_aug is not None:
        postenc_aug.cuda()
    if preenc_center_patch_minlift is not None and hasattr(preenc_center_patch_minlift, "cuda"):
        preenc_center_patch_minlift = preenc_center_patch_minlift.cuda()
    if preenc_patchshuffle is not None:
        preenc_patchshuffle.cuda()
    if preenc_patchmix is not None:
        preenc_patchmix = preenc_patchmix.cuda()
    if preenc_patchdropout is not None and hasattr(preenc_patchdropout, "cuda"):
        preenc_patchdropout = preenc_patchdropout.cuda()
    if preenc_ipmix is not None and hasattr(preenc_ipmix, "cuda"):
        preenc_ipmix = preenc_ipmix.cuda()
    if pre_temporaljitter is not None and hasattr(pre_temporaljitter, "cuda"):
        pre_temporaljitter = pre_temporaljitter.cuda()
    if fe_train is not None and hasattr(fe_train, "cuda"):
        fe_train = fe_train.cuda()
    if fe_eval is not None and hasattr(fe_eval, "cuda"):
        fe_eval = fe_eval.cuda()

    # Attach aug BEFORE DDP
    # PRE-encoding Temporal Jitter is attached through core.preenc_aug so it will not
    # overwrite any existing pre-encoding augmentation chain already present in the model.
    if pre_temporaljitter is not None:
        core_pre = _unwrap_model(net)
        existing_pre = getattr(core_pre, "preenc_aug", None)
        if not _module_tree_contains_type(existing_pre, pre_temporaljitter.__class__.__name__):
            _attach_preenc_aug(net, pre_temporaljitter)

    _attach_postenc_aug(net, postenc_aug)

    # DDP wrap
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and is_rank0:
        _log(f"Let's use {num_gpus} GPUs!", args)

    if is_rank0 and preenc_center_patch_minlift is not None:
        _log(
            f"[AUG] CenterPatchMinLift active during TRAIN only: "
            f"{{'patch_frac': {args.center_patch_minlift_patch_frac}, "
            f"'alpha_min': {args.center_patch_minlift_alpha_min}, "
            f"'alpha_max': {args.center_patch_minlift_alpha_max}, "
            f"'p': {args.center_patch_minlift_p}}}",
            args
        )

    if is_rank0 and preenc_patchshuffle is not None:
        _log(
            f"[AUG] PreEncodingPatchShuffle2D enabled: "
            f"{{'p': {args.pre_pshuf_p}, 'patch_size': {args.pre_pshuf_size}, "
            f"'same_perm_on_batch': {args.pre_pshuf_same_on_batch}}}",
            args
        )

    if is_rank0 and preenc_patchmix is not None:
        _log(
            f"[AUG] PreEncodingPatchMix2D enabled: "
            f"{{'p': {args.pre_pmix_p}, 'patch_size': {args.pre_pmix_size}, "
            f"'alpha': {args.pre_pmix_alpha}, 'beta': {args.pre_pmix_beta}, "
            f"'pad_to_multiple': {not args.pre_pmix_no_pad}}}",
            args
        )

    if is_rank0 and preenc_patchdropout is not None:
        _log(
            f"[AUG] PreEncPatchDropout2D active during TRAIN only.",
            args
        )

    if is_rank0 and getattr(args, "pre_tmask_enable", False) and float(getattr(args, "pre_tmask_p", 0.0)) > 0.0:
        _log(
            f"[AUG] PreEncodingTimeMask enabled inside model: "
            f"{{'p': {args.pre_tmask_p}, "
            f"'num_masks': {args.pre_tmask_num}, "
            f"'max_mask_frac': {args.pre_tmask_max_frac}, "
            f"'min_mask_len': {args.pre_tmask_min_len}, "
            f"'mode': '{args.pre_tmask_mode}', "
            f"'noise_std': {args.pre_tmask_noise_std}, "
            f"'layout': '{args.pre_tmask_layout}', "
            f"'same_on_batch': {args.pre_tmask_same_on_batch}, "
            f"'per_channel': {args.pre_tmask_per_channel}, "
            f"'channel_groups': {args.pre_tmask_channel_groups}, "
            f"'apply_in_eval': {args.pre_tmask_apply_in_eval}}}",
            args
        )

    if is_rank0 and preenc_ipmix is not None:
        _log(
            f"[AUG] PreEncodingIPMix enabled: "
            f"{{'mixing_set': '{args.pre_ipmix_mixing_set}', "
            f"'k': {args.pre_ipmix_k}, "
            f"'t': {args.pre_ipmix_t}, "
            f"'aug_severity': {args.pre_ipmix_aug_severity}, "
            f"'no_jsd': {args.pre_ipmix_no_jsd}, "
            f"'jsd_weight': {args.pre_ipmix_jsd_weight}, "
            f"'img_size': {args.pre_ipmix_img_size}, "
            f"'mixing_resize': {args.pre_ipmix_mixing_resize}, "
            f"'apply_in_eval': {args.pre_ipmix_apply_in_eval}, "
            f"'mixing_set_size': {len(preenc_ipmix.mixing_set)}}}",
            args
        )

    if is_rank0 and pre_temporaljitter is not None:
        _log(
            f"[AUG] PreTemporalJitter attached through core.preenc_aug: "
            f"{{'p': {args.pre_tjitter_p}, "
            f"'max_shift': {args.pre_tjitter_max}, "
            f"'per_sample': {args.pre_tjitter_per_sample}, "
            f"'layout': '{args.pre_tjitter_layout}', "
            f"'apply_in_eval': {args.pre_tjitter_apply_in_eval}}}",
            args
        )

    if is_rank0 and postenc_aug is not None:
        _log(f"[AUG] {postenc_name} enabled: {postenc_cfg}", args)

    # ---- data ----
    ImageNet_training_loader, train_sampler = get_training_dataloader(
        dataset=args.dataset,
        data_root=args.data_root,
        traindir=args.traindir,
        num_workers=2,
        batch_size=args.b // max(num_gpus, 1),
        shuffle=False,
        sampler=1,
    )

    ImageNet_test_loader, _ = get_test_dataloader(
        dataset=args.dataset,
        data_root=args.data_root,
        valdir=args.valdir,
        num_workers=2,
        batch_size=args.b // max(num_gpus, 1),
        shuffle=False,
        sampler=1,
    )

    # ---- optim ----
    b_lr = args.lr
    loss_function = CrossEntropyLabelSmooth(num_classes=args.num_classes, epsilon=0.1)

    optimizer = optim.SGD(
        [{"params": net.parameters(), "initial_lr": b_lr}],
        momentum=0.9,
        lr=b_lr,
        weight_decay=1e-5,
    )

    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.EPOCH, eta_min=0, last_epoch=0
    )

    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0

    # ---- checkpoints ----
    ckpt_last = os.path.join(RUN_DIR, "checkpoint_last.pth")
    ckpt_best = os.path.join(RUN_DIR, "model_best.pth")

    for epoch in range(1, settings.EPOCH + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Frequency Encoding switch for TRAIN
        if args.fe:
            _set_input_encoder(net, fe_train)
        else:
            _set_input_encoder(net, None)

        train(epoch, args)
        train_scheduler.step()

        # Frequency Encoding switch for EVAL
        if args.fe and args.fe_eval:
            _set_input_encoder(net, fe_eval)
        else:
            _set_input_encoder(net, None)

        acc = eval_training(epoch, args)

        if is_rank0:
            _save_state_dict(ckpt_last, net)

            if epoch > (settings.EPOCH - 5) and best_acc < acc:
                _save_state_dict(ckpt_best, net)
                best_acc = acc
                _log(f"[CKPT] Updated BEST at epoch={epoch} acc={float(acc):.6f}", args)

            if epoch >= (settings.EPOCH - 5) or (epoch % settings.SAVE_EPOCH == 0):
                ckpt_ep = os.path.join(RUN_DIR, f"checkpoint_epoch{epoch:03d}.pth")
                _save_state_dict(ckpt_ep, net)
                _log(f"[CKPT] Saved {os.path.basename(ckpt_ep)}", args)

    if is_rank0 and writer is not None:
        writer.close()
        _log("Finished training. Logs + checkpoints are in: {}".format(RUN_DIR), args)