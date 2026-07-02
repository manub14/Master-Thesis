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
from pathlib import Path

from spikingjelly.clock_driven import functional
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from PIL import Image
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
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
    resume_checkpoint as timm_resume_checkpoint,
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


# =====================================================================
# Temporal Jitter (POST-ENCODING)
#
# This module is attached to the model as `model.temporal_jitter` and is
# called inside SpikeDrivenTransformer.forward_features, right after the
# spiking patch embedding and before the backbone blocks. At that point the
# feature tensor is (T, B, C, H, W) with time on dim 0. The call is guarded
# by `self.training`, so the shift is applied during training only.
#
# REQUIRED model.py edits (once):
#   In __init__:      self.temporal_jitter = None
#                     self._tj_warned = False
#   In forward_features, after the local_time_shuffle block:
#       if (self.training and getattr(self, "temporal_jitter", None) is not None):
#           x = self.temporal_jitter(x)
# =====================================================================
class TemporalJitter(nn.Module):
    """
    Post-encoding temporal circular shift (wrap-around) on (T, B, C, H, W).

    Global mode:
      one shift k shared by the whole batch. x[t] -> x[(t - k) mod T].

    Per-sample mode:
      each sample b draws its own shift k_b. x[t, b] -> x[(t - k_b) mod T, b].

    Wrap-around keeps the total activity and the per-frame spike-rate
    distribution unchanged, so no artificial empty or repeated frames are
    introduced at the boundary.
    """

    def __init__(self, max_shift: int = 1, p: float = 0.5, per_sample: bool = False):
        super().__init__()
        self.max_shift = int(max_shift)
        self.p = float(p)
        self.per_sample = bool(per_sample)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # Only act on 5D spike features (T, B, C, H, W).
        if self.max_shift <= 0 or x.dim() != 5:
            return x

        # Apply with probability p.
        if self.p < 1.0 and torch.rand((), device=x.device).item() > self.p:
            return x

        T, B, C, H, W = x.shape
        if T <= 1:
            return x

        if not self.per_sample:
            k = int(
                torch.randint(-self.max_shift, self.max_shift + 1, (1,), device=x.device).item()
            )
            if k == 0:
                return x
            # Roll along the time axis (dim 0) with wrap-around.
            return torch.roll(x, shifts=k, dims=0)

        # Per-sample jitter: one shift per sample.
        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (B,), device=x.device)
        t = torch.arange(T, device=x.device).view(T, 1)          # (T, 1)
        idx = (t - shifts.view(1, B)) % T                        # (T, B)
        idx = idx.view(T, B, 1, 1, 1).expand(T, B, C, H, W)      # (T, B, C, H, W)
        return torch.gather(x, dim=0, index=idx)


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
                    resume_epoch += 1  # start at the next epoch

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

# The first arg parser parses out only the --config argument
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
    "--data-dir",
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
    "--dataset-download",
    action="store_true",
    default=False,
    help="Force dataset download when supported by the dataset backend.",
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
    help="",
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
    help="pooling layers in SPS moduls",
)
parser.add_argument(
    "--TET",
    default=False,
    type=bool,
    help="",
)
parser.add_argument(
    "--TET-means",
    default=1.0,
    type=float,
    help="",
)
parser.add_argument(
    "--TET-lamb",
    default=0.0,
    type=float,
    help="",
)
parser.add_argument(
    "--spike-mode",
    default="lif",
    type=str,
    help="",
)
parser.add_argument(
    "--layer",
    default=4,
    type=int,
    help="",
)
parser.add_argument(
    "--in-channels",
    default=3,
    type=int,
    help="",
)
parser.add_argument(
    "--dim",
    default=512,
    type=int,
    metavar="N",
    help="embedding dimension for the model",
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=1000,
    metavar="N",
    help="number of label classes (auto-corrected for known datasets if left at default)",
)
parser.add_argument(
    "--time-steps",
    type=int,
    default=4,
    metavar="N",
    help="",
)
parser.add_argument(
    "--num-heads",
    type=int,
    default=8,
    metavar="N",
    help="",
)
parser.add_argument(
    "--patch-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size",
)
parser.add_argument(
    "--mlp-ratio",
    type=int,
    default=4,
    metavar="N",
    help="expand ration of embedding dimension in MLP block",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size (default: None => model default)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "-vb",
    "--val-batch-size",
    type=int,
    default=16,
    metavar="N",
    help="input val batch size for training (default: 16)",
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
    help="Optimizer Epsilon (default: None, use opt default)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.0001,
    help="weight decay (default: 0.0001)",
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)

# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="step",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step")',
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="learning rate (default: 0.01)",
)
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
    help="learning rate noise limit percent (default: 0.67)",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="warmup learning rate (default: 0.0001)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
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
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10)",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
parser.add_argument(
    "--hflip",
    type=float,
    default=0.5,
    help="Horizontal flip training aug probability",
)
parser.add_argument(
    "--vflip",
    type=float,
    default=0.0,
    help="Vertical flip training aug probability",
)
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: None)',
)
parser.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
)
parser.add_argument(
    "--jsd",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
)
parser.add_argument(
    "--bce-loss",
    action="store_true",
    default=False,
    help="Enable BCE loss w/ Mixup/CutMix use.",
)
parser.add_argument(
    "--bce-target-thresh",
    type=float,
    default=None,
    help="Threshold for binarizing softened BCE targets (default: None, disabled)",
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Random erase prob (default: 0.)",
)
parser.add_argument(
    "--remode",
    type=str,
    default="const",
    help='Random erase mode (default: "const")',
)
parser.add_argument(
    "--recount",
    type=int,
    default=1,
    help="Random erase count (default: 1)",
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.0,
    help="mixup alpha, mixup enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=0.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Probability of performing mixup or cutmix when either/both is enabled",
)
parser.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Probability of switching to cutmix when both mixup and cutmix enabled",
)
parser.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
)
parser.add_argument(
    "--mixup-off-epoch",
    default=0,
    type=int,
    metavar="N",
    help="Turn off mixup after this epoch, disabled if 0 (default: 0)",
)
parser.add_argument(
    "--smoothing",
    type=float,
    default=0.1,
    help="Label smoothing (default: 0.1)",
)
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)
parser.add_argument(
    "--drop",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Dropout rate (default: 0.)",
)
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=0.2,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
parser.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)

# ---- Temporal Jitter (POST-ENCODING) CLI flags ----
parser.add_argument(
    "--tshift",
    action="store_true",
    default=False,
    help="Enable post-encoding Temporal Jitter (circular time shift with wrap-around).",
)
parser.add_argument(
    "--tshift-prob",
    type=float,
    default=0.5,
    help="Probability to apply the temporal shift each step.",
)
parser.add_argument(
    "--tshift-max",
    type=int,
    default=1,
    help="Max absolute shift in timesteps. Sample k in [-max, +max].",
)
parser.add_argument(
    "--tshift-per-sample",
    action="store_true",
    default=False,
    help="Apply a different random shift per sample in the batch (jitter).",
)
# ---------------------------------------------------

# ---- Generic DVS dataset helper flags ----
parser.add_argument(
    "--dvs-train-ratio",
    type=float,
    default=0.9,
    help="Train split ratio for generic DVS datasets when no explicit train/test folders are present.",
)
# -----------------------------------------

# Batch norm parameters
parser.add_argument(
    "--bn-tf",
    action="store_true",
    default=False,
    help="Use Tensorflow BatchNorm defaults for models that support it (default: False)",
)
parser.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
parser.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)
parser.add_argument(
    "--sync-bn",
    action="store_true",
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)
parser.add_argument(
    "--dist-bn",
    type=str,
    default="",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)
parser.add_argument(
    "--split-bn",
    action="store_true",
    help="Enable separate BN layers per augmentation split.",
)
parser.add_argument(
    "--linear-prob",
    action="store_true",
    help="",
)

# Model Exponential Moving Average
parser.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="decay factor for model weights moving average (default: 0.9998)",
)

# Misc
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    metavar="S",
    help="random seed (default: 42)",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument(
    "--checkpoint-hist",
    type=int,
    default=10,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 4)",
)
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--dvs-aug",
    action="store_true",
    default=False,
    help="enable DVS cutout augmentation",
)
parser.add_argument(
    "--dvs-trival-aug",
    action="store_true",
    default=False,
    help="enable DVS trivial augmentation",
)
parser.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1")',
)
parser.add_argument(
    "--tta",
    type=int,
    default=0,
    metavar="N",
    help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
)
parser.add_argument("--local_rank", "--local-rank", dest="local_rank", default=0, type=int)
parser.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)

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


def _is_dvs_dataset(dataset_name):
    dataset_name = str(dataset_name).lower()
    if dataset_name in {"cifar10-dvs-tet", "cifar10-dvs", "cifar100-dvs", "gesture",
                        "dvs128gesture", "dvs128-gesture"}:
        return True
    dvs_list = getattr(dvs_utils, "DVS_DATASET", [])
    return dataset_name in {str(x).lower() for x in dvs_list}


def _apply_dataset_autofixes(args):
    """Auto-fix a few settings so CIFAR / DVS / Tiny-ImageNet work out of the box."""
    dataset_name = str(args.dataset).lower()

    if dataset_name in {"torch/cifar10", "cifar10"}:
        if args.num_classes == 1000:
            args.num_classes = 10
        if args.img_size is None:
            args.img_size = 32
        if args.val_split == "validation":
            args.val_split = "test"
        if args.train_split not in {"train", "training"}:
            args.train_split = "train"
        _logger.info(
            "Auto-configured CIFAR-10 settings: num_classes=10, img_size=32, val_split=test"
        )

    elif dataset_name in {"torch/cifar100", "cifar100"}:
        if args.num_classes == 1000:
            args.num_classes = 100
        if args.img_size is None:
            args.img_size = 32
        if args.val_split == "validation":
            args.val_split = "test"
        if args.train_split not in {"train", "training"}:
            args.train_split = "train"
        _logger.info(
            "Auto-configured CIFAR-100 settings: num_classes=100, img_size=32, val_split=test"
        )

    elif dataset_name in {"tiny-imagenet", "tiny_imagenet", "tinyimagenet"}:
        if args.num_classes == 1000:
            args.num_classes = 200
        if args.img_size is None or args.img_size == 224:
            args.img_size = 64
        if args.in_channels != 3:
            args.in_channels = 3
        if args.train_split not in {"train", "training"}:
            args.train_split = "train"
        if args.val_split == "validation":
            args.val_split = "val"
        _logger.info(
            "Auto-configured Tiny ImageNet settings: num_classes=200, img_size=64, in_channels=3, val_split=val"
        )

    elif dataset_name in {"gesture", "dvs128gesture", "dvs128-gesture"}:
        if args.num_classes == 1000:
            args.num_classes = 11
        if args.img_size is None:
            args.img_size = 64
        if args.in_channels == 3:
            args.in_channels = 2
        if args.input_size is None:
            args.input_size = [args.in_channels, args.img_size, args.img_size]
        if args.mean is None or len(args.mean) != args.in_channels:
            args.mean = [0.0] * args.in_channels
        if args.std is None or len(args.std) != args.in_channels:
            args.std = [1.0] * args.in_channels
        _logger.info(
            "Auto-configured DVS128Gesture settings: "
            "num_classes=11, img_size=64, in_channels=2, mean/std=2-channel"
        )

    elif dataset_name in {"cifar10-dvs", "cifar10-dvs-tet"}:
        if args.num_classes == 1000:
            args.num_classes = 10
        if args.img_size is None:
            args.img_size = 64
        if args.in_channels == 3:
            args.in_channels = 2
        _logger.info(
            "Auto-configured CIFAR10-DVS settings: num_classes=10, img_size=64, in_channels=2"
        )

    elif dataset_name == "cifar100-dvs":
        if args.num_classes == 1000:
            args.num_classes = 100
        if args.img_size is None:
            args.img_size = 64
        if args.in_channels == 3:
            args.in_channels = 2
        _logger.info(
            "Auto-configured CIFAR100-DVS settings: num_classes=100, img_size=64, in_channels=2"
        )

    return args


def _extract_frames_from_object(obj):
    if isinstance(obj, dict):
        for key in ("frames", "data", "x", "img", "tensor", "events"):
            if key in obj:
                obj = obj[key]
                break
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        obj = obj[0]

    if torch.is_tensor(obj):
        x = obj
    else:
        x = torch.as_tensor(obj)
    return x


def _ensure_dvs_tensor_shape(x):
    """
    Convert loaded DVS frame tensor to [T, C, H, W] whenever possible.
    Accepts common layouts like:
      [T, C, H, W], [C, T, H, W], [T, H, W, C], [T, H, W], [C, H, W], [H, W]
    """
    x = _extract_frames_from_object(x)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)

    elif x.ndim == 3:
        if x.shape[0] in (1, 2, 3) and x.shape[1] > 8 and x.shape[2] > 8:
            x = x.unsqueeze(0)   # [C, H, W] -> [1, C, H, W]
        else:
            x = x.unsqueeze(1)   # [T, H, W] -> [T, 1, H, W]

    elif x.ndim == 4:
        if x.shape[1] in (1, 2, 3) and x.shape[2] > 8 and x.shape[3] > 8:
            pass                 # already [T, C, H, W]
        elif x.shape[0] in (1, 2, 3) and x.shape[2] > 8 and x.shape[3] > 8:
            x = x.permute(1, 0, 2, 3).contiguous()   # [C, T, H, W] -> [T, C, H, W]
        elif x.shape[-1] in (1, 2, 3) and x.shape[1] > 8 and x.shape[2] > 8:
            x = x.permute(0, 3, 1, 2).contiguous()   # [T, H, W, C] -> [T, C, H, W]
        else:
            pass
    else:
        raise ValueError(f"Unsupported DVS sample rank: {x.ndim}, expected 2D/3D/4D")

    return x.float()


def _resize_dvs_frames(x, img_size):
    """x: [T, C, H, W]; resize spatial size only."""
    if img_size is None:
        return x
    if x.shape[-2] == img_size and x.shape[-1] == img_size:
        return x
    return F.interpolate(x, size=(img_size, img_size), mode="nearest")


class GenericDVSFrameDataset(Dataset):
    """
    Generic DVS frame dataset for folder layouts like:
      data_dir/train/class_x/*.pt|*.pth|*.npy|*.npz
      data_dir/test/class_x/*.pt|*.pth|*.npy|*.npz
    or a single-level layout that gets split with --dvs-train-ratio.
    """
    EXTENSIONS = {".pt", ".pth", ".npy", ".npz"}

    def __init__(self, root, img_size=None):
        self.root = Path(root)
        self.img_size = img_size

        if not self.root.exists():
            raise FileNotFoundError(f"DVS dataset root does not exist: {self.root}")

        class_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if len(class_dirs) == 0:
            raise RuntimeError(
                f"No class subfolders found under {self.root}. "
                f"Expected folders like {self.root}/class_name/*.pt"
            )

        self.class_to_idx = {p.name: i for i, p in enumerate(class_dirs)}
        self.samples = []
        self.targets = []

        for class_dir in class_dirs:
            cls_idx = self.class_to_idx[class_dir.name]
            files = sorted(
                [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in self.EXTENSIONS]
            )
            for fp in files:
                self.samples.append((str(fp), cls_idx))
                self.targets.append(cls_idx)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No supported sample files found under {self.root}. "
                f"Supported extensions: {sorted(self.EXTENSIONS)}"
            )

        _logger.info(
            f"Loaded GenericDVSFrameDataset from {self.root} with "
            f"{len(self.samples)} samples and {len(self.class_to_idx)} classes."
        )

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, path):
        suffix = Path(path).suffix.lower()

        if suffix in {".pt", ".pth"}:
            obj = torch.load(path, map_location="cpu")
        elif suffix == ".npy":
            obj = np.load(path, allow_pickle=True)
        elif suffix == ".npz":
            obj = np.load(path, allow_pickle=True)
            if isinstance(obj, np.lib.npyio.NpzFile):
                if "frames" in obj.files:
                    obj = obj["frames"]
                elif "data" in obj.files:
                    obj = obj["data"]
                elif "x" in obj.files:
                    obj = obj["x"]
                else:
                    obj = obj[obj.files[0]]
        else:
            raise ValueError(f"Unsupported file extension for DVS sample: {path}")

        x = _ensure_dvs_tensor_shape(obj)
        x = _resize_dvs_frames(x, self.img_size)
        return x

    def __getitem__(self, index):
        path, target = self.samples[index]
        x = self._load_sample(path)
        return x, target


def _stratified_train_test_split(dataset, train_ratio, num_classes, seed=42):
    if not hasattr(dataset, "targets"):
        raise ValueError("Dataset must have a `.targets` attribute for stratified splitting.")

    targets = np.asarray(dataset.targets)
    rng = np.random.RandomState(seed)

    train_indices = []
    test_indices = []

    for c in range(num_classes):
        cls_idx = np.where(targets == c)[0]
        if len(cls_idx) == 0:
            continue
        rng.shuffle(cls_idx)

        if len(cls_idx) == 1:
            n_train = 1
        else:
            n_train = int(round(len(cls_idx) * train_ratio))
            n_train = max(1, min(n_train, len(cls_idx) - 1))

        train_indices.extend(cls_idx[:n_train].tolist())
        test_indices.extend(cls_idx[n_train:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


class TinyImageNetValDataset(Dataset):
    """
    Tiny ImageNet validation dataset for the official layout:
      val/images/*.JPEG
      val/val_annotations.txt
    Class indices are taken from the sibling train/ folder so that train and
    val agree on label ordering. timm's create_loader assigns the transform to
    `self.transform`, so we leave transform=None here.
    """
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.images_dir = self.root / "images"
        self.ann_file = self.root / "val_annotations.txt"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet val images dir: {self.images_dir}")
        if not self.ann_file.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet val annotation file: {self.ann_file}")

        train_dir = self.root.parent / "train"
        class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

        self.samples = []
        with open(self.ann_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_name = parts[0]
                wnid = parts[1]
                img_path = self.images_dir / img_name
                target = self.class_to_idx[wnid]
                self.samples.append((str(img_path), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _get_vis_batch(input_tensor):
    """Convert a possibly-5D DVS batch into a 4D tensor for save_image."""
    if input_tensor.ndim == 4:
        return input_tensor
    if input_tensor.ndim == 5:
        return input_tensor[:, 0]
    return None


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args = _apply_dataset_autofixes(args)

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    if "RANK" in os.environ:
        args.rank = int(os.environ["RANK"])
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

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

    args.device = "cuda"
    args.world_size = 1
    args.rank = 0  # global rank

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            f"Distributed init done: RANK={args.rank} LOCAL_RANK={args.local_rank} "
            f"WORLD_SIZE={args.world_size} -> device {args.device}"
        )
    else:
        _logger.info("Training with a single process on 1 GPUs.")

    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
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
            "Install NVIDIA apex or upgrade to PyTorch 1.6+"
        )

    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.initial_seed()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random_seed(args.seed, args.rank)

    args.dvs_mode = _is_dvs_dataset(args.dataset)

    model = create_model(
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

    # === Attach post-encoding Temporal Jitter to the model ===
    # This must happen before DDP wrapping so the module's forward (which runs
    # on model.module) sees `self.temporal_jitter`. The model forward applies it
    # after the spiking patch embedding, guarded by self.training.
    if args.tshift:
        model.temporal_jitter = TemporalJitter(
            max_shift=args.tshift_max,
            p=args.tshift_prob,
            per_sample=args.tshift_per_sample,
        )
        if args.local_rank == 0:
            _logger.info(
                f"[TemporalJitter] Attached for POST-ENCODING use: "
                f"max_shift={args.tshift_max}, prob={args.tshift_prob}, "
                f"per_sample={args.tshift_per_sample}"
            )
    else:
        model.temporal_jitter = None

    if args.local_rank == 0:
        _logger.info(f"Creating model {args.model}")
        try:
            summary_shape = None
            if args.input_size is not None:
                summary_shape = (2, *args.input_size)
            elif args.img_size is not None:
                if _is_dvs_dataset(args.dataset):
                    summary_shape = (
                        2,
                        args.time_steps,
                        args.in_channels,
                        args.img_size,
                        args.img_size,
                    )
                else:
                    summary_shape = (2, args.in_channels, args.img_size, args.img_size)

            if summary_shape is not None:
                _logger.info(str(torchinfo.summary(model, summary_shape)))
            else:
                _logger.info(
                    "Skipping torchinfo.summary because neither --img-size nor --input-size is set."
                )
        except Exception as e:
            _logger.warning(f"torchinfo.summary skipped due to: {e}")

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

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
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != "native":
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0,
        )

    # setup exponential moving average of model weights
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != "native":
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True, find_unused_parameters=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # for linear probe
    if args.linear_prob:
        base_model = model.module if hasattr(model, "module") else model
        for n, p in base_model.named_parameters():
            if "patch_embed" in n:
                p.requires_grad = False

    # setup learning rate schedule and starting epoch
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

    # create the train and eval datasets
    dataset_train, dataset_eval = None, None
    ds_lower = str(args.dataset).lower()

    if args.dataset == "cifar10-dvs-tet":
        dataset_train = dvs_utils.DVSCifar10(
            root=os.path.join(args.data_dir, "train"),
            train=True,
        )
        dataset_eval = dvs_utils.DVSCifar10(
            root=os.path.join(args.data_dir, "test"),
            train=False,
        )
    elif args.dataset == "cifar10-dvs":
        dataset = CIFAR10DVS(
            args.data_dir,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
            transform=dvs_utils.Resize(args.img_size),
        )
        dataset_train, dataset_eval = dvs_utils.split_to_train_test_set(0.9, dataset, 10)

    elif args.dataset == "cifar100-dvs":
        train_root = os.path.join(args.data_dir, "train")
        test_root = os.path.join(args.data_dir, "test")

        if os.path.isdir(train_root) and os.path.isdir(test_root):
            dataset_train = GenericDVSFrameDataset(train_root, img_size=args.img_size)
            dataset_eval = GenericDVSFrameDataset(test_root, img_size=args.img_size)
        else:
            dataset = GenericDVSFrameDataset(args.data_dir, img_size=args.img_size)
            dataset_train, dataset_eval = _stratified_train_test_split(
                dataset=dataset,
                train_ratio=args.dvs_train_ratio,
                num_classes=args.num_classes,
                seed=args.seed,
            )

    elif ds_lower in {"gesture", "dvs128gesture", "dvs128-gesture"}:
        dataset_train = DVS128Gesture(
            args.data_dir,
            train=True,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
            transform=dvs_utils.Resize(args.img_size),
        )
        dataset_eval = DVS128Gesture(
            args.data_dir,
            train=False,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
            transform=dvs_utils.Resize(args.img_size),
        )

    elif ds_lower in {"tiny-imagenet", "tiny_imagenet", "tinyimagenet"}:
        train_root = os.path.join(args.data_dir, "train")
        val_root = os.path.join(args.data_dir, "val")
        _logger.info(f"TinyImageNet train_root = {train_root}")
        _logger.info(f"TinyImageNet val_root   = {val_root}")
        _logger.info(f"train exists = {os.path.exists(train_root)}")
        _logger.info(f"val exists   = {os.path.exists(val_root)}")

        dataset_train = ImageFolder(train_root)

        # Official Tiny ImageNet val is images/ + val_annotations.txt.
        # Restructured val (class subfolders) works with ImageFolder directly.
        val_ann = os.path.join(val_root, "val_annotations.txt")
        if os.path.isfile(val_ann):
            _logger.info("TinyImageNet val: using val_annotations.txt layout.")
            dataset_eval = TinyImageNetValDataset(val_root, transform=None)
        else:
            _logger.info("TinyImageNet val: using ImageFolder (class-subfolder) layout.")
            dataset_eval = ImageFolder(val_root)

    else:
        auto_download = args.dataset_download or ds_lower in {
            "torch/cifar10",
            "cifar10",
            "torch/cifar100",
            "cifar100",
        }

        dataset_train = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.train_split,
            is_training=True,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats,
            transform=transforms_train,
            download=auto_download,
        )
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            batch_size=args.batch_size,
            transform=transforms_eval,
            download=auto_download,
        )

    # setup mixup / cutmix
    collate_fn = None
    train_dvs_aug, train_dvs_trival_aug = None, None
    if args.dvs_aug:
        train_dvs_aug = dvs_utils.Cutout(n_holes=1, length=16)
    if args.dvs_trival_aug:
        train_dvs_trival_aug = dvs_utils.SNNAugmentWide()

    mixup_fn = None
    is_dvs_dataset = _is_dvs_dataset(args.dataset)
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
        # For DVS (5D) inputs, timm's collate-based mixup does not apply; use the
        # callable form and let the loop apply it to 4D static inputs only.
        if args.prefetcher and not is_dvs_dataset:
            assert not num_aug_splits
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1 and not is_dvs_dataset:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]

    loader_train, loader_eval, train_idx = None, None, None
    if args.train_split_path is not None:
        train_idx = np.load(args.train_split_path).tolist()

    if is_dvs_dataset:
        train_sampler = (
            DistributedSampler(dataset_train, shuffle=True) if args.distributed else None
        )
        eval_sampler = (
            DistributedSampler(dataset_eval, shuffle=False) if args.distributed else None
        )

        loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        loader_eval = DataLoader(
            dataset_eval,
            batch_size=args.val_batch_size,
            shuffle=False,
            sampler=eval_sampler,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=False,
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

    # setup loss function
    if args.jsd:
        assert num_aug_splits > 1
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing, target_threshold=args.bce_target_thresh
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()

    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    if args.rank == 0:
        decreasing = False
        saver = CheckpointSaver(
            model=model,
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
                model,
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
                distribute_bn(model, args.world_size, args.dist_bn == "reduce")

            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")
                ema_eval_metrics = validate(
                    model_ema.module,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    amp_autocast=amp_autocast,
                    log_suffix=" (EMA)",
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
                _logger.info(
                    "*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch)
                )

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


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
    is_dvs_dataset = _is_dvs_dataset(args.dataset)

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        input = input.float()

        if not args.prefetcher or is_dvs_dataset:
            if args.amp and not isinstance(input, torch.cuda.HalfTensor):
                input = input.half()
            input, target = input.cuda(), target.cuda()

            if dvs_aug is not None:
                input = dvs_aug(input)

            if dvs_trival_aug is not None:
                output_aug = []
                for i in range(input.shape[0]):
                    output_aug.append(dvs_trival_aug(input[i]))
                input = torch.stack(output_aug)
                del output_aug

        # Mixup applies to 4D static image batches only. DVS 5D inputs are left
        # untouched here. Temporal Jitter runs later, inside the model forward.
        if mixup_fn is not None and input.ndim == 4:
            input, target = mixup_fn(input, target)

        if args.channels_last and input.ndim == 4:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            # Temporal Jitter is applied POST-ENCODING inside model.forward_features,
            # on the (T, B, C, H, W) spike features, guarded by model.training.
            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            if args.TET:
                loss = criterion.TET_loss(
                    output, target, loss_fn, means=args.TET_means, lamb=args.TET_lamb
                )
            else:
                loss = loss_fn(output, target)

        sample_number += input.shape[0]

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

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
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

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
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                    )
                )

                if args.save_images and output_dir:
                    vis_input = _get_vis_batch(input)
                    if vis_input is not None:
                        torchvision.utils.save_image(
                            vis_input,
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

    model.eval()  # disables Temporal Jitter (guarded by self.training in the model)
    is_dvs_dataset = _is_dvs_dataset(args.dataset)

    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.float()

            if (target >= args.num_classes).sum() != 0 or (target < 0).sum() != 0:
                print(target)

            last_batch = batch_idx == last_idx

            if not args.prefetcher or is_dvs_dataset:
                if args.amp and not isinstance(input, torch.cuda.HalfTensor):
                    input = input.half()
                input = input.cuda()
                target = target.cuda()

            if args.channels_last and input.ndim == 4:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)

            if isinstance(output, (tuple, list)):
                output = output[0]

            if args.TET:
                output = output.mean(0)

            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0: target.size(0): reduce_factor]

            if (target >= args.num_classes).sum() != 0 or (target < 0).sum() != 0:
                print(target)

            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

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

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )

    return metrics


if __name__ == "__main__":
    main()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()