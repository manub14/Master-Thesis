import argparse
import time
import yaml
import os
import logging
import numpy as np
import gc
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from typing import Optional

from spikingjelly.clock_driven import functional
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

import torch
import torch.nn as nn
import torchvision.utils
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torchinfo

from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint as timm_resume_checkpoint,  # avoid shadowing
    load_checkpoint,
    convert_splitbn_model,
    model_parameters,
)
from timm.models.helpers import clean_state_dict
from timm.utils import *
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
    BinaryCrossEntropy,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

import model, dvs_utils, criterion

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


# === NEW: PatchDropoutTokens (paper-faithful version) =======================
class PatchDropoutTokens(nn.Module):
    """
    PatchDropout implemented on patch tokens (not raw pixels).

    Supported shapes:
      (B, N, D)        standard ViT
      (B, T, N, D)     time-dimension / spiking variant
    """

    def __init__(
    self,
    keep: float = 1.0,
    min_keep: Optional[float] = None,
    num_prefix_tokens: int = 1,
    mask_output: bool = False,
):
        super().__init__()
        self.keep = float(keep)
        self.min_keep = None if min_keep is None else float(min_keep)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.mask_output = bool(mask_output)
        self._debug_printed = False

    def _sample_keep_rate(self, device):
        if self.min_keep is not None:
            r = torch.empty(1, device=device).uniform_(self.min_keep, 1.0).item()
            return max(min(r, 1.0), 0.0)
        return max(min(self.keep, 1.0), 0.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return tokens

        reshaped = False
                # SDT / DVS feature-map mode, e.g. (T, B, C, H, W) or (B, T, C, H, W)
        # We mask spatial feature positions after encoding without changing shape.
        if tokens.dim() == 5:
            if not self.mask_output:
                raise RuntimeError(
                    "5D SDT feature-map PatchDropout requires --pd-mask-output. "
                    "Real token dropping would change the HxW grid shape."
                )

            orig_shape = tokens.shape
            C, H, W = orig_shape[-3], orig_shape[-2], orig_shape[-1]

            keep_rate = self._sample_keep_rate(tokens.device)
            if keep_rate >= 1.0:
                return tokens

            x = tokens.reshape(-1, C, H, W)
            B_eff = x.shape[0]

            num_patches = H * W
            k = max(1, int(round(keep_rate * num_patches)))

            mask = torch.zeros(B_eff, 1, H, W, device=tokens.device, dtype=tokens.dtype)

            for b in range(B_eff):
                keep_idx = torch.randperm(num_patches, device=tokens.device)[:k]
                mask[b, 0].view(-1)[keep_idx] = 1.0

            x = x * mask

            if not self._debug_printed:
                print(
                    f"[PatchDropoutTokens] 5D MASK mode keep_rate={keep_rate:.3f}, "
                    f"feature_shape={tuple(orig_shape)}, kept_spatial={k}/{num_patches}",
                    flush=True,
                )
                self._debug_printed = True

            return x.reshape(orig_shape)
        if tokens.dim() == 4:
            # (B, T, N, D) -> (B*T, N, D)
            B, T, N, D = tokens.shape
            tokens = tokens.reshape(B * T, N, D)
            reshaped = True

        elif tokens.dim() == 3:
            # (B, N, D)
            B, N, D = tokens.shape
            T = None

        else:
            return tokens

        num_prefix = self.num_prefix_tokens

        if N <= num_prefix:
            return tokens if not reshaped else tokens.reshape(B, T, N, D)

        keep_rate = self._sample_keep_rate(tokens.device)

        if keep_rate >= 1.0:
            return tokens if not reshaped else tokens.reshape(B, T, N, D)

        n_patch_tokens = N - num_prefix
        k = max(1, int(round(keep_rate * n_patch_tokens)))

        B_eff = tokens.shape[0]

        # ============================================================
        # SDT-safe mode: MASK tokens but DO NOT remove them.
        # This keeps sequence length N unchanged, so reshape to H*W works.
        # ============================================================
        if getattr(self, "mask_output", False):
            mask = torch.zeros(B_eff, N, device=tokens.device, dtype=tokens.dtype)

            # always keep prefix tokens, if any
            if num_prefix > 0:
                mask[:, :num_prefix] = 1.0

            for b in range(B_eff):
                rand_patch_idx = torch.randperm(n_patch_tokens, device=tokens.device)[:k]
                rand_patch_idx = rand_patch_idx + num_prefix
                mask[b, rand_patch_idx] = 1.0

            tokens = tokens * mask.unsqueeze(-1)

            if not self._debug_printed:
                print(
                    f"[PatchDropoutTokens] MASK mode keep_rate={keep_rate:.3f}, "
                    f"num_prefix={num_prefix}, seq_len={N}, kept_per_sample={k}",
                    flush=True,
                )
                self._debug_printed = True

            if reshaped:
                tokens = tokens.reshape(B, T, N, D)

            return tokens

    # ============================================================
    # Original token-dropping mode: removes tokens.
    # Do NOT use this for SDT if the model later reshapes to H x W.
    # ============================================================
        all_idx = []
        for b in range(B_eff):
            rand_patch_idx = torch.randperm(n_patch_tokens, device=tokens.device)[:k]
            rand_patch_idx = rand_patch_idx + num_prefix
            prefix_idx = torch.arange(num_prefix, device=tokens.device)
            full_idx = torch.cat([prefix_idx, rand_patch_idx], dim=0)
            all_idx.append(full_idx.unsqueeze(0))

        idx = torch.cat(all_idx, dim=0)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, D)
        tokens = tokens.gather(dim=1, index=gather_idx)

        if not self._debug_printed:
            print(
                f"[PatchDropoutTokens] DROP mode keep_rate={keep_rate:.3f}, "
                f"num_prefix={num_prefix}, kept_len={tokens.shape[1]}",
                flush=True,
            )
            self._debug_printed = True

        if reshaped:
            tokens = tokens.reshape(B, T, tokens.shape[1], D)
        return tokens
    
# ============================================================
# ADD after PatchDropoutTokens class
# ============================================================

def _apply_patchdrop_to_output(pd_layer, output):
    """
    Applies PatchDropoutTokens to module output if output looks like token tensor.
    Supports:
      Tensor: (B,N,D) or (B,T,N,D)
      tuple/list where first element is token tensor
    """
    if pd_layer is None:
        return output

    if torch.is_tensor(output):
        if not hasattr(pd_layer, "_shape_printed"):
            print(f"[PatchDropoutTokens] hook output shape = {tuple(output.shape)}", flush=True)
            pd_layer._shape_printed = True

        if output.dim() in (3, 4, 5):
            return pd_layer(output)
        return output


    if torch.is_tensor(output):
        if output.dim() in (3, 4, 5):
            return pd_layer(output)
        return output

    if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
        first = output[0]
        if first.dim() in (3, 4, 5):
            return (pd_layer(first),) + output[1:]
        return output

    if isinstance(output, list) and len(output) > 0 and torch.is_tensor(output[0]):
        first = output[0]
        if first.dim() in (3, 4, 5):
            new_output = list(output)
            new_output[0] = pd_layer(first)
            return new_output
        return output

    return output


def attach_patchdropout_hook(net, pd_layer, preferred_names=("patch_embed", "patch_embed.proj")):
    if pd_layer is None:
        return None

    handles = []

    def hook_fn(module, inputs, output):
        return _apply_patchdrop_to_output(pd_layer, output)

    for name, module in net.named_modules():
        lname = name.lower()
        if any(pname in lname for pname in preferred_names):
            handle = module.register_forward_hook(hook_fn)
            handles.append(handle)
            print(f"[PatchDropoutTokens] Hook attached to module: {name}", flush=True)
            return handles
    _logger.warning(
        "[PatchDropoutTokens] PatchDropout requested but no hook was attached. "
        "Check model module names with: print(dict(net.named_modules()).keys())"
    )
    print(
        "[PatchDropoutTokens] WARNING: no patch_embed module found. "
        "PatchDropout was NOT attached. You may need to add it inside model forward().",
        flush=True,
    )
    return None
# ===========================================================================


def resume_checkpoint(
    model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True
):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if log_info:
                _logger.info("Restoring model state from checkpoint...")
            state_dict = clean_state_dict(checkpoint["state_dict"])
            model.load_state_dict(state_dict, strict=False)

            if optimizer is not None and "optimizer" in checkpoint:
                if log_info:
                    _logger.info("Restoring optimizer state from checkpoint...")
                optimizer.load_state_dict(checkpoint["optimizer"])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info("Restoring AMP loss scaler state from checkpoint...")
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if "epoch" in checkpoint:
                resume_epoch = checkpoint["epoch"]
                if "version" in checkpoint and checkpoint["version"] > 1:
                    resume_epoch += 1

            if log_info:
                _logger.info(
                    "Loaded checkpoint '{}' (epoch {})".format(
                        checkpoint_path, checkpoint["epoch"]
                    )
                )
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


torch.backends.cudnn.benchmark = True

# The first arg parser parses only --config
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="imagenet.yml",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# Dataset / Model parameters
parser.add_argument(
    "-data-dir",
    metavar="DIR",
    default="",
    help="path to dataset",
)
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="torch/cifar10",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
parser.add_argument(
    "--train-split-path",
    type=str,
    default=None,
    metavar="N",
    help="Optional .npy with subset indices for training split",
)
parser.add_argument(
    "--model",
    default="sdt",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "sdt")',
)
parser.add_argument(
    "--pooling-stat",
    default="1111",
    type=str,
    help="pooling layers in SPS modules",
)
parser.add_argument(
    "--TET",
    default=False,
    type=bool,
    help="Whether to use TET loss",
)
parser.add_argument(
    "--TET-means",
    default=1.0,
    type=float,
    help="TET loss mean scaling",
)
parser.add_argument(
    "--TET-lamb",
    default=0.0,
    type=float,
    help="TET loss lambda",
)
parser.add_argument(
    "--spike-mode",
    default="lif",
    type=str,
    help="Spiking neuron type",
)
parser.add_argument(
    "--layer",
    default=4,
    type=int,
    help="Model depth (#layers)",
)
parser.add_argument(
    "--in-channels",
    default=3,
    type=int,
    help="Input channels",
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained weights if available",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (before training)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="Don't resume optimizer state",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=1000,
    metavar="N",
    help="number of label classes",
)
parser.add_argument(
    "--time-steps",
    type=int,
    default=4,
    metavar="N",
    help="temporal steps for SNN model",
)
parser.add_argument(
    "--num-heads",
    type=int,
    default=8,
    metavar="N",
    help="attention heads",
)
parser.add_argument(
    "--patch-size", type=int, default=None, metavar="N", help="Image patch size"
)
parser.add_argument(
    "--mlp-ratio",
    type=int,
    default=4,
    metavar="N",
    help="MLP expansion ratio",
)
parser.add_argument(
    "--dim",
    default=512,
    type=int,
    metavar="N",
    help="embedding dimension",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image size (height==width)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Full input dims, e.g. --input-size 3 224 224",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Validation crop percent",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override dataset mean",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override dataset std",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="train batch size",
)
parser.add_argument(
    "-vb",
    "--val-batch-size",
    type=int,
    default=16,
    metavar="N",
    help="val batch size",
)

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd")',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer epsilon",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer betas",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum",
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.0001,
    help="weight decay",
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm",
)
parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode ("norm", "value", "agc")',
)

# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="step",
    type=str,
    metavar="SCHEDULER",
    help="LR scheduler",
)
parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate")
parser.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
parser.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="lr noise limit percent",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="lr noise std-dev",
)
parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="lr cycle len multiplier",
)
parser.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="lr cycle limit",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="warmup lr",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="epochs to train",
)
parser.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual start epoch",
)
parser.add_argument(
    "--decay-epochs",
    type=float,
    default=30,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=3,
    metavar="N",
    help="warmup epochs",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="cooldown epochs at min_lr",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate",
)

# Augmentation & regularization parameters
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale range",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio range",
)
parser.add_argument("--hflip", type=float, default=0.5, help="Horizontal flip prob")
parser.add_argument("--vflip", type=float, default=0.0, help="Vertical flip prob")
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor",
)
parser.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='AutoAugment policy ("v0", "original", etc.)',
)
parser.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (0 or >=2)",
)
parser.add_argument(
    "--jsd",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss (use with --aug-splits)",
)
parser.add_argument(
    "--bce-loss",
    action="store_true",
    default=False,
    help="Enable BCE loss w/ Mixup/CutMix",
)
parser.add_argument(
    "--bce-target-thresh",
    type=float,
    default=None,
    help="Threshold for binarizing softened BCE targets",
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Random erase prob",
)
parser.add_argument("--remode", type=str, default="const", help='Random erase mode')
parser.add_argument("--recount", type=int, default=1, help="Random erase count")
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) aug split",
)
parser.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (>0 enables)")
parser.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha (>0 enables)")
parser.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio (overrides alpha)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Prob to apply mixup or cutmix when enabled",
)
parser.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Switch-to-cutmix prob when both enabled",
)
parser.add_argument(
    "--pre-patchdropout",
    action="store_true",
    default=False,
    help="Enable pre-encoding image-level PatchDropout before the model.",
)

parser.add_argument(
    "--pre-pd-keep",
    type=float,
    default=0.95,
    help="Keep ratio for pre-encoding PatchDropout. Example: 0.95 keeps 95% of image patches.",
)

parser.add_argument(
    "--pre-pd-patch-size",
    type=int,
    default=16,
    help="Patch size for pre-encoding PatchDropout. For Tiny ImageNet 64x64, patch_size=16 gives a 4x4 grid.",
)

parser.add_argument(
    "--pre-pd-prob",
    type=float,
    default=1.0,
    help="Probability of applying pre-encoding PatchDropout per sample.",
)
parser.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='mixup/cutmix application ("batch","pair","elem")',
)
parser.add_argument(
    "--mixup-off-epoch",
    default=0,
    type=int,
    metavar="N",
    help="Turn off mixup after this epoch (0=never off)",
)
parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing")
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation ("random","bilinear","bicubic")',
)
parser.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="Dropout")
parser.add_argument("--drop-path", type=float, default=0.2, metavar="PCT", help="Drop path")
parser.add_argument("--drop-block", type=float, default=None, metavar="PCT", help="Drop block")

# Batch norm parameters
parser.add_argument(
    "--bn-tf",
    action="store_true",
    default=False,
    help="Use Tensorflow BatchNorm defaults where supported",
)
parser.add_argument("--bn-momentum", type=float, default=None, help="BN momentum override")
parser.add_argument("--bn-eps", type=float, default=None, help="BN epsilon override")
parser.add_argument("--sync-bn", action="store_true", help="Enable SyncBatchNorm")
parser.add_argument(
    "--dist-bn",
    type=str,
    default="",
    help='Distribute BN stats between nodes ("broadcast", "reduce", or "")',
)
parser.add_argument("--split-bn", action="store_true", help="Separate BN per aug-split")
parser.add_argument("--linear-prob", action="store_true", help="Linear probe mode")

# Model EMA
parser.add_argument("--model-ema", action="store_true", default=False, help="Enable EMA")
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force EMA on CPU; disables EMA validation",
)
parser.add_argument("--model-ema-decay", type=float, default=0.9998, help="EMA decay")
parser.add_argument(
    "--dvs-resize",
    type=int,
    default=64,
    help="Resize size for DVS datasets.",
)
# Misc
parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed")
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging train status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument("--checkpoint-hist", type=int, default=10, metavar="N", help="checkpoints to keep")
parser.add_argument("-j", "--workers", type=int, default=4, metavar="N", help="dataloader workers")
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save train batch images every log interval",
)
parser.add_argument("--amp", action="store_true", default=False, help="use AMP (apex or native)")
parser.add_argument("--apex-amp", action="store_true", default=False, help="Force NVIDIA Apex AMP")
parser.add_argument("--native-amp", action="store_true", default=False, help="Force native Torch AMP")
parser.add_argument("--channels-last", action="store_true", default=False, help="Channels-last memory layout")
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader",
)
parser.add_argument("--no-prefetcher", action="store_true", default=False, help="disable fast prefetcher")
parser.add_argument("--dvs-aug", action="store_true", default=False, help="DVS Cutout")
parser.add_argument("--dvs-trival-aug", action="store_true", default=False, help="DVS TrivialAugmentWide")
parser.add_argument("--output", default="", type=str, metavar="PATH", help="output folder")
parser.add_argument("--experiment", default="", type=str, metavar="NAME", help="train experiment name")
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric to track (default: "top1")',
)
parser.add_argument("--tta", type=int, default=0, metavar="N", help="test-time augmentation factor")
parser.add_argument("--local_rank", "--local-rank", dest="local_rank", default=0, type=int)
parser.add_argument("--use-multi-epochs-loader", action="store_true", default=False)
parser.add_argument("--torchscript", dest="torchscript", action="store_true", help="torchscript for inference")
parser.add_argument("--log-wandb", action="store_true", default=False, help="log to wandb")

# === PatchDropout CLI flags (updated for token-level PatchDropout) ===
parser.add_argument(
    "--patchdropout",
    action="store_true",
    default=False,
    help="Enable PatchDropoutTokens during TRAINING (token dropping).",
)
parser.add_argument(
    "--pd-keep",
    type=float,
    default=1.0,
    help="Fixed keep rate (0<keep<=1). e.g. 0.5 keeps ~50% of patch tokens.",
)
parser.add_argument(
    "--pd-min-keep",
    type=float,
    default=None,
    help="If set, each forward we sample keep_rate ~ Uniform(pd_min_keep, 1.0).",
)
parser.add_argument(
    "--pd-prefix-tokens",
    type=int,
    default=1,
    help="How many prefix tokens to ALWAYS keep (typically 1 for CLS).",
)
parser.add_argument(
    "--pd-mask-output",
    action="store_true",
    default=False,
    help="Apply PatchDropout as masking output instead of dropping tokens; needed for SDT grid-shape compatibility.",
)
# =====================================================================

_logger = logging.getLogger("train")
stream_handler = logging.StreamHandler()
format_str = "%(asctime)s %(levelname)s: %(message)s"
stream_handler.setFormatter(logging.Formatter(format_str))
_logger.addHandler(stream_handler)
_logger.propagate = False


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


DVS_NAME_SET = {
    "cifar10-dvs-tet",
    "cifar10-dvs",
    "gesture",
    "dvs128gesture",
    "dvs128-gesture",
    "dvs128_gesture",
}

def is_dvs_dataset_name(name):
    ds = str(name).lower().strip()
    extra = set()
    if hasattr(dvs_utils, "DVS_DATASET"):
        extra = {str(x).lower().strip() for x in dvs_utils.DVS_DATASET}
    return ds in DVS_NAME_SET or ds in extra

def _autofill_dataset_defaults(args):
    ds = (args.dataset or "").lower().strip()

    cifar10_names = {"torch/cifar10", "cifar10", "cifar-10"}
    cifar100_names = {"torch/cifar100", "cifar100", "cifar-100"}
    tiny_names = {"tiny-imagenet", "tinyimagenet", "tiny-imagenet-200", "tinyimagenet200"}
    if ds in cifar10_names:
        if args.num_classes != 10:
            args.num_classes = 10
        if args.img_size is None:
            args.img_size = 32
        if args.input_size is None:
            args.input_size = [3, 32, 32]
        if args.mean is None:
            args.mean = [0.4914, 0.4822, 0.4465]
        if args.std is None:
            args.std = [0.2023, 0.1994, 0.2010]
        if args.val_split in ("validation", "val"):
            args.val_split = "test"
    elif ds in cifar100_names:
        if args.num_classes != 100:
            args.num_classes = 100
        if args.img_size is None:
            args.img_size = 32
        if args.input_size is None:
            args.input_size = [3, 32, 32]
        if args.mean is None:
            args.mean = [0.5071, 0.4867, 0.4408]
        if args.std is None:
            args.std = [0.2675, 0.2565, 0.2761]
        if args.val_split in ("validation", "val"):
            args.val_split = "test"
    elif ds in {"gesture", "dvs128gesture", "dvs128-gesture", "dvs128_gesture"}:
        args.num_classes = 11
        args.in_channels = 2

        if args.img_size is None:
            args.img_size = args.dvs_resize

        # DVS input has 2 channels
        args.input_size = [2, args.img_size, args.img_size]

        args.mean = [0.0, 0.0]
        args.std = [1.0, 1.0]
    elif ds in {"cifar10-dvs", "cifar10-dvs-tet"}:
        args.num_classes = 10
        args.in_channels = 2

        if args.img_size is None:
            args.img_size = args.dvs_resize

        # IMPORTANT: force this even if YAML config has 3 channels
        args.input_size = [2, args.img_size, args.img_size]

        # IMPORTANT: DVS has 2 channels
        args.mean = [0.0, 0.0]
        args.std = [1.0, 1.0]

    elif ds in {"cifar100-dvs", "cifar100dvs"}:
            raise ValueError(
                "CIFAR100-DVS is not a standard SpikingJelly dataset. "
                "Use --dataset torch/cifar100 for static CIFAR100, "
                "or add a custom CIFAR100-DVS Dataset class."
            )
    elif ds in tiny_names:
        args.num_classes = 200
        args.in_channels = 3

        if args.img_size is None:
            args.img_size = 64

        if args.input_size is None:
            args.input_size = [3, 64, 64]

        # ImageNet-style normalization is fine for Tiny ImageNet
        if args.mean is None:
            args.mean = [0.485, 0.456, 0.406]

        if args.std is None:
            args.std = [0.229, 0.224, 0.225]

        if args.train_split in ("training",):
            args.train_split = "train"

        if args.val_split in ("validation", "test"):
            args.val_split = "val"
    return args


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    # auto-fill dataset-specific defaults so CIFAR-10 / CIFAR-100 work cleanly
    args = _autofill_dataset_defaults(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0  # global rank

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        _logger.info("Training with a single process on 1 GPU.")
    assert args.rank >= 0

    # AMP resolve
    use_amp = None
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDIA apex or upgrade to PyTorch 1.6"
        )

    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.initial_seed()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random_seed(args.seed, args.rank)

    args.dvs_mode = False
    if is_dvs_dataset_name(args.dataset):
        args.dvs_mode = True

    # ------ build model -------------------------------------------------
    net = create_model(
        args.model,
        T=args.time_steps,
        pretrained=args.pretrained,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        pooling_stat=args.pooling_stat,
        img_size_h=args.img_size,
        img_size_w=args.img_size,
        patch_size=args.patch_size,
        embed_dims=args.dim,
        mlp_ratios=args.mlp_ratio,
        in_channels=args.in_channels,
        qkv_bias=False,
        depths=args.layer,
        sr_ratios=1,
        spike_mode=args.spike_mode,
        dvs_mode=args.dvs_mode,
        TET=args.TET,
    )
    if args.local_rank == 0:
        if args.pd_min_keep is not None:
            _logger.info(
                f"[PatchDropoutTokens] Enabled and hooked: keep ~ U[{args.pd_min_keep},1.0], "
                f"prefix_tokens={args.pd_prefix_tokens}, "
                f"mask_output={args.pd_mask_output}"
            )
        else:
            _logger.info(
                f"[PatchDropoutTokens] Enabled and hooked: fixed keep={args.pd_keep}, "
                f"prefix_tokens={args.pd_prefix_tokens}, "
                f"mask_output={args.pd_mask_output}"
            )

    # attach PatchDropoutTokens module to the model so forward() can use it
    # NOTE: your model class forward() still needs to actually call this layer.
# ============================================================
# REPLACE this block in main()
# ============================================================

    # attach PatchDropoutTokens module to the model and hook it into forward()
    net.patchdrop_layer = None
    net.patchdrop_handles = None
    net.patchdrop_mask_output = False

    if args.patchdropout:
        net.patchdrop_layer = PatchDropoutTokens(
            keep=args.pd_keep,
            min_keep=args.pd_min_keep,
            num_prefix_tokens=args.pd_prefix_tokens,
            mask_output=args.pd_mask_output,
        )

        # Set both, because your SpikeDrivenTransformer may check either one
        net.patchdrop_mask_output = args.pd_mask_output
        net.patchdrop_layer.mask_output = args.pd_mask_output

        net.patchdrop_handles = attach_patchdropout_hook(
        net,
        net.patchdrop_layer,
        preferred_names=("patch_embed", "patch_embed.proj"),
        )

        if not net.patchdrop_handles:
            raise RuntimeError(
                "PatchDropout was requested, but no patch_embed hook was attached. "
                "Print model.named_modules() and choose the correct module name."
            )


        if args.local_rank == 0:
            if args.pd_min_keep is not None:
                _logger.info(
                    f"[PatchDropoutTokens] Enabled and hooked: keep ~ U[{args.pd_min_keep},1.0], "
                    f"prefix_tokens={args.pd_prefix_tokens}"
                )
            else:
                _logger.info(
                    f"[PatchDropoutTokens] Enabled and hooked: fixed keep={args.pd_keep}, "
                    f"prefix_tokens={args.pd_prefix_tokens}"
                )
    if args.num_classes is None:
        assert hasattr(net, "num_classes"), "Model must define num_classes if not set on cmd line/config."
        args.num_classes = net.num_classes

    data_config = resolve_data_config(vars(args), model=net, verbose=args.local_rank == 0)

    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    "data-" + args.dataset.split("/")[-1],
                    f"t-{args.time_steps}",
                    f"spike-{args.spike_mode}",
                ]
            )
        output_dir = get_outdir(args.output if args.output else "./output/train", exp_name)
        file_handler = logging.FileHandler(os.path.join(output_dir, f"{args.model}.log"), "w")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(logging.INFO)
        _logger.addHandler(file_handler)

    if args.local_rank == 0:
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in net.parameters()])}"
        )

    # aug splits
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1
        num_aug_splits = args.aug_splits

    # split BN
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        net = convert_splitbn_model(net, max(num_aug_splits, 2))

    # to GPU
    net.cuda()

    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)

    # sync BN for DDP
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != "native":
            net = convert_syncbn_model(net)
        else:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        if args.local_rank == 0:
            _logger.info("Converted model to SyncBatchNorm.")

    if args.torchscript:
        assert not use_amp == "apex"
        assert not args.sync_bn
        net = torch.jit.script(net)

    optimizer = create_optimizer_v2(net, **optimizer_kwargs(cfg=args))

    # AMP scalers
    amp_autocast = suppress
    loss_scaler = None
    if use_amp == "apex":
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    # resume
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            net,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0,
        )

    # EMA
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(net, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # DDP wrap
    if args.distributed:
        if has_apex and use_amp != "native":
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DDP.")
            net = ApexDDP(net, delay_allreduce=True, find_unused_parameters=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DDP.")
            net = NativeDDP(net, device_ids=[args.local_rank], find_unused_parameters=True)

    # linear probe freezing
    if args.linear_prob:
        freeze_model = net.module if args.distributed else net
        for n, p in freeze_model.named_parameters():
            if "patch_embed" in n:
                p.requires_grad = False

    # scheduler
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None and (not args.linear_prob):
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info("Scheduled epochs: {}".format(num_epochs))

    transforms_train, transforms_eval = None, None

    # datasets
    dataset_train, dataset_eval = None, None
    if args.dataset == "cifar10-dvs-tet":
        dataset_train = dvs_utils.DVSCifar10(root=os.path.join(args.data_dir, "train"), train=True)
        dataset_eval = dvs_utils.DVSCifar10(root=os.path.join(args.data_dir, "test"), train=False)
    elif args.dataset == "cifar10-dvs":
        dataset_all = CIFAR10DVS(
            args.data_dir,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
            transform=dvs_utils.Resize(args.dvs_resize),
        )
        dataset_train, dataset_eval = dvs_utils.split_to_train_test_set(0.9, dataset_all, 10)
    elif str(args.dataset).lower().strip() in {"gesture", "dvs128gesture", "dvs128-gesture", "dvs128_gesture"}:
        dataset_train = DVS128Gesture(
            args.data_dir,
            train=True,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
            transform=dvs_utils.Resize(args.dvs_resize),
        )
        dataset_eval = DVS128Gesture(
            args.data_dir,
            train=False,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
            transform=dvs_utils.Resize(args.dvs_resize),
        )
    elif str(args.dataset).lower() in {"tiny-imagenet", "tiny_imagenet", "tinyimagenet", "tiny-imagenet-200", "tinyimagenet200"}:
        train_root = os.path.join(args.data_dir, "train")
        val_root = os.path.join(args.data_dir, "val")

        _logger.info(f"TinyImageNet train_root = {train_root}")
        _logger.info(f"TinyImageNet val_root   = {val_root}")
        _logger.info(f"train exists = {os.path.exists(train_root)}")
        _logger.info(f"val exists   = {os.path.exists(val_root)}")
        _logger.info(f"val entries  = {sorted(os.listdir(val_root))[:20]}")

        dataset_train = ImageFolder(train_root)
        dataset_eval = ImageFolder(val_root)

        _logger.info(f"TinyImageNet train classes = {len(dataset_train.classes)}")
        _logger.info(f"TinyImageNet val classes   = {len(dataset_eval.classes)}")
        _logger.info(f"TinyImageNet train samples = {len(dataset_train)}")
        _logger.info(f"TinyImageNet val samples   = {len(dataset_eval)}")

        assert dataset_train.classes == dataset_eval.classes, (
            "TinyImageNet train/val class order mismatch. "
            "Check that val/ has the same class folder names as train/."
        )

    else:
        dataset_train = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.train_split,
            is_training=True,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats,
            transform=transforms_train,
        )
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            batch_size=args.batch_size,
            transform=transforms_eval,
        )

    # mixup/cutmix
    collate_fn = None
    train_dvs_aug, train_dvs_trival_aug = None, None
    if args.dvs_aug:
        train_dvs_aug = dvs_utils.Cutout(n_holes=1, length=16)
    if args.dvs_trival_aug:
        train_dvs_trival_aug = dvs_utils.SNNAugmentWide()

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        if args.prefetcher and not is_dvs_dataset_name(args.dataset):
            assert not num_aug_splits
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1 and not is_dvs_dataset_name(args.dataset):
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # loaders
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]

    loader_train, loader_eval, train_idx = None, None, None
    if args.train_split_path is not None:
        train_idx = np.load(args.train_split_path).tolist()

    if is_dvs_dataset_name(args.dataset):
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )
        loader_eval = torch.utils.data.DataLoader(
            dataset_eval, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
    else:
        loader_train = create_loader(
            dataset_train,
            input_size=data_config["input_size"],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
        )

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=args.val_batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=args.pin_mem,
        )
    if args.local_rank == 0:
        _logger.info("Create dataloader: {}".format(args.dataset))

    # losses
    if args.jsd:
        assert num_aug_splits > 1
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # checkpoint saver
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    if args.rank == 0:
        decreasing = False
        saver = CheckpointSaver(
            model=net,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                net,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                dvs_aug=train_dvs_aug,
                dvs_trival_aug=train_dvs_trival_aug,
            )

            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(net, args.world_size, args.dist_bn == "reduce")

            eval_metrics = validate(net, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=" (EMA)"
                )
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    os.path.join(output_dir, "summary.csv"),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if saver is not None:
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


def apply_pre_encoding_patchdropout(
    x: torch.Tensor,
    keep: float = 0.95,
    patch_size: int = 16,
    prob: float = 1.0,
) -> torch.Tensor:
    """
    Pre-encoding PatchDropout before the model.

    Supports:
      (B, C, H, W)       static image batch
      (B, T, C, H, W)    DVS frame batch

    For DVS, the same spatial patch mask is applied across all time steps.
    """
    if x is None or x.dim() not in (4, 5):
        return x

    if keep >= 1.0 or prob <= 0.0:
        return x

    is_dvs_5d = x.dim() == 5

    if is_dvs_5d:
        B, T, C, H, W = x.shape
    else:
        B, C, H, W = x.shape
        T = None

    device = x.device
    dtype = x.dtype

    grid_h = max(1, H // patch_size)
    grid_w = max(1, W // patch_size)
    num_patches = grid_h * grid_w

    keep = max(0.0, min(float(keep), 1.0))
    k = max(1, int(round(keep * num_patches)))

    mask_grid = torch.ones(B, 1, grid_h, grid_w, device=device, dtype=dtype)

    for b in range(B):
        if torch.rand(1, device=device).item() > prob:
            continue

        mask_grid[b].zero_()
        keep_idx = torch.randperm(num_patches, device=device)[:k]
        flat = mask_grid[b, 0].view(-1)
        flat[keep_idx] = 1.0

    mask = torch.nn.functional.interpolate(
        mask_grid,
        size=(H, W),
        mode="nearest",
    )

    if is_dvs_5d:
        # (B, 1, H, W) -> (B, 1, 1, H, W)
        # broadcast over time T and channels C
        mask = mask.unsqueeze(1)

    if not hasattr(apply_pre_encoding_patchdropout, "_printed"):
        if is_dvs_5d:
            msg = (
                f"[PrePatchDropout] Applied to 5D DVS input: "
                f"B={B}, T={T}, C={C}, H={H}, W={W}, "
                f"grid={grid_h}x{grid_w}, patch_size={patch_size}, "
                f"keep={keep}, kept={k}/{num_patches}, prob={prob}"
            )
        else:
            msg = (
                f"[PrePatchDropout] Applied to 4D input: "
                f"B={B}, C={C}, H={H}, W={W}, "
                f"grid={grid_h}x{grid_w}, patch_size={patch_size}, "
                f"keep={keep}, kept={k}/{num_patches}, prob={prob}"
            )

        print(msg, flush=True)
        _logger.info(msg)
        apply_pre_encoding_patchdropout._printed = True

    return x * mask

def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
    dvs_aug=None,
    dvs_trival_aug=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher:
            if hasattr(loader, "mixup_enabled"):
                loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    sample_number = 0
    start_time = time.time()

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    functional.reset_net(model)

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (inp, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        inp = inp.float()

        # Branch 1: no prefetcher OR dvs dataset
        if not args.prefetcher or is_dvs_dataset_name(args.dataset):
            if args.amp and not isinstance(inp, torch.cuda.HalfTensor):
                inp = inp.half()
            inp, target = inp.cuda(), target.cuda()

            if dvs_aug is not None:
                inp = dvs_aug(inp)

            if dvs_trival_aug is not None:
                tmp_out = []
                for i in range(inp.shape[0]):
                    tmp_out.append(dvs_trival_aug(inp[i]))
                inp = torch.stack(tmp_out)
                del tmp_out

            if mixup_fn is not None:
                inp, target = mixup_fn(inp, target)

        # Branch 2: prefetcher=True & non-DVS dataset
        else:
            pass
        
        if args.pre_patchdropout:
            if args.local_rank == 0 and batch_idx % args.log_interval == 0:
                _logger.info(
                    f"[PrePatchDropout] BEFORE batch {batch_idx}: "
                    f"shape={tuple(inp.shape)}, keep={args.pre_pd_keep}, "
                    f"patch_size={args.pre_pd_patch_size}, prob={args.pre_pd_prob}"
                )

            inp = apply_pre_encoding_patchdropout(
                inp,
                keep=args.pre_pd_keep,
                patch_size=args.pre_pd_patch_size,
                prob=args.pre_pd_prob,
            )

            if args.local_rank == 0 and batch_idx % args.log_interval == 0:
                _logger.info(
                    f"[PrePatchDropout] AFTER  batch {batch_idx}: shape={tuple(inp.shape)}"
                )

        if args.channels_last:
            inp = inp.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            out = model(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if args.TET:
                loss = criterion.TET_loss(out, target, loss_fn, means=args.TET_means, lamb=args.TET_lamb)
            else:
                loss = loss_fn(out, target)

        sample_number += inp.shape[0]
        if not args.distributed:
            losses_m.update(loss.item(), inp.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head="agc" in args.clip_mode),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        functional.reset_net(model)
        if model_ema is not None:
            model_ema.update(model)
            functional.reset_net(model_ema)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [pg["lr"] for pg in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), inp.size(0))

            if args.local_rank == 0:
                _logger.info(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}  "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100.0 * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=inp.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=inp.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                    )
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        inp,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()
    if args.local_rank == 0:
        _logger.info(f"samples / s = {sample_number / (time.time() - start_time): .3f}")
    return OrderedDict([("loss", losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=""):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (inp, target) in enumerate(loader):
            inp = inp.float()
            if (target >= 1000).sum() != 0 or (target < 0).sum() != 0:
                print(target)

            last_batch = batch_idx == last_idx
            if not args.prefetcher or is_dvs_dataset_name(args.dataset):
                if args.amp and not isinstance(inp, torch.cuda.HalfTensor):
                    inp = inp.half()
                inp = inp.cuda()
                target = target.cuda()

            if args.channels_last:
                inp = inp.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                out = model(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if args.TET:
                out = out.mean(0)

            reduce_factor = args.tta
            if reduce_factor > 1:
                out = out.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss = loss_fn(out, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(out, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inp.size(0))
            top1_m.update(acc1.item(), out.size(0))
            top5_m.update(acc5.item(), out.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )

    metrics = OrderedDict([("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)])
    return metrics


if __name__ == "__main__":
    main()

torch.cuda.empty_cache()
gc.collect()