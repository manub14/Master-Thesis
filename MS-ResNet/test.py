import argparse
import time
import os
import re
import json
import logging
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import models.MS_ResNet


# ----------------------------
# Helpers: parsing + logging
# ----------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _safe_slug(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s or "exp"


def _infer_seed_from_weights(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"seed(\d+)", base)
    if m:
        return m.group(1)
    m = re.search(r"_s(\d+)", base)
    if m:
        return m.group(1)
    return "unknown"


def _setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # clear existing handlers so a new log_path always works
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ----------------------------
# AverageMeter + accuracy
# ----------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        out = []
        for k in topk:
            kk = min(k, output.size(1))
            correct_k = correct[:kk].reshape(-1).float().sum(0)
            out.append(correct_k.mul_(100.0 / batch_size))
        return out


# ----------------------------
# Spike hook collector
# ----------------------------
class SpikeStatsCollector:
    """
    Collects layer-wise:
      - non_zero fraction
      - firing_rate (mean activation)
    from forward-hooked modules.

    It supports:
      - TBCHW / TBC...  (time dim = 0)
      - BTCHW / BTC...  (time dim = 1)
      - no explicit time dimension -> logs under 'overall'
    """
    def __init__(self, model, time_steps=4, layout="auto",
                 hook_pattern=r"(lif|ifnode|if|spike|neuron|mem_update|mem)",
                 leaf_only=True):
        self.model = model
        self.time_steps = int(time_steps) if time_steps is not None else None
        self.layout = layout
        self.pattern = re.compile(hook_pattern, re.IGNORECASE)
        self.leaf_only = bool(leaf_only)

        self.handles = []
        self.hooked_module_names = []
        self.reset()

    def reset(self):
        self._stats = {"overall": {}}
        if self.time_steps is not None and self.time_steps > 1:
            for t in range(self.time_steps):
                self._stats[f"t{t}"] = {}

    def close(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _should_hook(self, name, module):
        if self.leaf_only and any(True for _ in module.children()):
            return False

        text = f"{name}::{module.__class__.__name__}"
        return self.pattern.search(text) is not None

    def _extract_tensor(self, obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (tuple, list)):
            for x in obj:
                t = self._extract_tensor(x)
                if t is not None:
                    return t
        if isinstance(obj, dict):
            for _, x in obj.items():
                t = self._extract_tensor(x)
                if t is not None:
                    return t
        return None

    def _bucket(self, group, name):
        d = self._stats.setdefault(group, {})
        if name not in d:
            d[name] = {"sum": 0.0, "nonzero": 0, "numel": 0}
        return d[name]

    def _accumulate(self, bucket, x):
        if x is None:
            return
        if not torch.is_tensor(x):
            return
        if x.numel() == 0:
            return

        x = x.detach()
        if x.is_sparse:
            x = x.to_dense()
        if not torch.is_floating_point(x):
            x = x.float()

        bucket["sum"] += x.sum().item()
        bucket["nonzero"] += torch.count_nonzero(x).item()
        bucket["numel"] += x.numel()

    def _time_dim(self, x):
        if self.layout == "none":
            return None
        if x.ndim < 2 or self.time_steps is None or self.time_steps <= 1:
            return None

        T = self.time_steps

        if self.layout in ("t_first", "TBCHW", "T-first"):
            return 0 if x.shape[0] == T else None
        if self.layout in ("b_first", "BTCHW", "B-first"):
            return 1 if x.ndim > 1 and x.shape[1] == T else None

        # auto
        if x.shape[0] == T and (x.ndim < 2 or x.shape[1] != T):
            return 0
        if x.ndim > 1 and x.shape[1] == T and x.shape[0] != T:
            return 1
        if x.shape[0] == T:
            return 0
        if x.ndim > 1 and x.shape[1] == T:
            return 1

        return None

    def _make_hook(self, name):
        def hook(module, inputs, output):
            x = self._extract_tensor(output)
            if x is None:
                return

            # always keep overall stats
            self._accumulate(self._bucket("overall", name), x)

            tdim = self._time_dim(x)
            if tdim is None:
                return

            max_t = min(self.time_steps, x.shape[tdim])
            for t in range(max_t):
                xt = x.select(dim=tdim, index=t)
                self._accumulate(self._bucket(f"t{t}", name), xt)
        return hook

    def register(self):
        self.close()
        self.hooked_module_names = []

        for name, module in self.model.named_modules():
            if self._should_hook(name, module):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
                self.hooked_module_names.append(name)

        return self.hooked_module_names

    def summary(self):
        non_zero = {}
        firing_rate = {}

        for group, per_layer in self._stats.items():
            non_zero[group] = {}
            firing_rate[group] = {}
            for name, s in per_layer.items():
                if s["numel"] > 0:
                    non_zero[group][name] = float(s["nonzero"] / s["numel"])
                    firing_rate[group][name] = float(s["sum"] / s["numel"])

        return non_zero, firing_rate


# ----------------------------
# CIFAR-100-C Dataset (npy)
# ----------------------------
class CIFAR100CDataset(Dataset):
    """
    Expects CIFAR-100-C directory with files:
      - labels.npy
      - <corruption>.npy  (shape: 50000 x 32 x 32 x 3, uint8)
    Each corruption has 5 severity levels, each 10,000 samples in order.
    """
    def __init__(self, c_dir, corruption, severity, transform=None):
        self.c_dir = c_dir
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        labels_path = os.path.join(c_dir, "labels.npy")
        data_path = os.path.join(c_dir, f"{corruption}.npy")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.npy not found at: {labels_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{corruption}.npy not found at: {data_path}")

        self.labels = np.load(labels_path)
        data = np.load(data_path)

        if data.shape[0] != 50000:
            raise ValueError(f"Expected 50000 images in {data_path}, got {data.shape[0]}")
        if self.labels.shape[0] != 50000:
            raise ValueError(f"Expected 50000 labels in {labels_path}, got {self.labels.shape[0]}")
        if not (1 <= severity <= 5):
            raise ValueError("severity must be in [1..5]")

        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = data[start:end]
        self.targets = self.labels[start:end]

    def __len__(self):
        return int(self.targets.shape[0])

    def __getitem__(self, idx):
        img = self.data[idx]
        label = int(self.targets[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ----------------------------
# DataLoaders
# ----------------------------
def get_imagenet_loader(valdir, batch_size=100, num_workers=4, shuffle=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ds = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    return DataLoader(ds, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)


def get_cifar100_loader(data_root, batch_size=256, num_workers=4, shuffle=False):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
    tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    ds = datasets.CIFAR100(root=data_root, train=False, download=False, transform=tf)
    return DataLoader(ds, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)


def get_cifar100c_loader(c_dir, corruption, severity, batch_size=256, num_workers=4, shuffle=False):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
    tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    ds = CIFAR100CDataset(c_dir=c_dir, corruption=corruption, severity=severity, transform=tf)
    return DataLoader(ds, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)


# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, loss_fn, device, logger=None,
             log_interval=100, log_suffix="", spike_collector=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    if spike_collector is not None:
        spike_collector.reset()

    model.eval()
    end = time.time()
    last_idx = len(loader) - 1

    for batch_idx, (images, target) in enumerate(loader):
        last_batch = batch_idx == last_idx

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = loss_fn(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        losses_m.update(loss.item(), images.size(0))
        top1_m.update(acc1.item(), images.size(0))
        top5_m.update(acc5.item(), images.size(0))
        batch_time_m.update(time.time() - end)
        end = time.time()

        if logger is not None and (last_batch or batch_idx % log_interval == 0):
            log_name = "Test" + log_suffix
            logger.info(
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

    non_zero, firing_rate = ({}, {})
    if spike_collector is not None:
        non_zero, firing_rate = spike_collector.summary()

    return OrderedDict([
        ("loss", losses_m.avg),
        ("top1", top1_m.avg),
        ("top5", top5_m.avg),
        ("non_zero", non_zero),
        ("firing_rate", firing_rate),
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-weights', type=str, default="resnet34.pth",
                        help='the weights file you want to test')
    parser.add_argument('-net', type=str, required=True,
                        choices=["resnet18", "resnet34", "resnet104", "resnet110"])
    parser.add_argument('-gpu', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('-b', type=int, default=100, help='batch size for dataloader')
    parser.add_argument('--log-interval', type=int, default=100)

    parser.add_argument('--experiment', type=str, default="")
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--test-root', type=str, default='test')

    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'cifar100', 'cifar100c'])
    parser.add_argument('--data-root', type=str, default='/media/homes/bist/data')
    parser.add_argument('--valdir', type=str, default='/data1/imagenet/val')
    parser.add_argument('--cifar100c-dir', type=str, default='')
    parser.add_argument('--corruption', type=str, default='all')
    parser.add_argument('--severity', type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--rsm', type=str2bool, nargs='?', const=True, default=False,
                    help='build RSM version of the network')
    parser.add_argument('--rsm-p', type=float, default=0.0,
                        help='RSM probability used during training')

    # hook-based spike logging
    parser.add_argument('--save-spike-stats', type=str2bool, nargs='?', const=True, default=True,
                        help='enable hook-based non_zero / firing_rate logging')
    parser.add_argument('--time-steps', type=int, default=4,
                        help='used only to split hook outputs into t0..t{T-1}')
    parser.add_argument('--spike-layout', type=str, default='auto',
                        choices=['auto', 'TBCHW', 'BTCHW', 'none'],
                        help='layout of hook output if it contains time dimension')
    parser.add_argument('--hook-pattern', type=str,
                        default=r"(lif|ifnode|if|spike|neuron|mem_update|mem)",
                        help='regex over module name/class name to decide which modules to hook')

    args = parser.parse_args()

    exp_name = _safe_slug(args.experiment) if args.experiment else _safe_slug(os.path.splitext(os.path.basename(args.weights))[0])
    seed_name = str(args.seed) if args.seed >= 0 else _infer_seed_from_weights(args.weights)

    out_dir = os.path.join(args.test_root, exp_name, seed_name)
    log_path = os.path.join(out_dir, "sdt.log")
    logger = _setup_logger(log_path)

    logger.info("========== TEST START ==========")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Net: {args.net}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Seed: {seed_name}")
    logger.info("Args:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")

    use_cuda = bool(args.gpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.gpu and not torch.cuda.is_available():
        logger.warning("GPU was requested but CUDA is not available. Falling back to CPU.")

    if args.num_classes is not None:
        num_classes = args.num_classes
    elif args.dataset == "imagenet":
        num_classes = 1000
    else:
        num_classes = 100

    cifar_stem = (args.dataset in ["cifar100", "cifar100c"])

    if args.rsm:
        if args.net == "resnet34":
            net = models.MS_ResNet.resnet34(num_classes=num_classes, cifar_stem=cifar_stem, rsm=True, rsm_p=args.rsm_p)
        elif args.net == "resnet104":
            net = models.MS_ResNet.resnet104(num_classes=num_classes, rsm=True, rsm_p=args.rsm_p)
        elif args.net == "resnet110":
            net = models.MS_ResNet.resnet110(num_classes=num_classes, rsm=True, rsm_p=args.rsm_p)
        else:
            net = models.MS_ResNet.resnet18(num_classes=num_classes, cifar_stem=cifar_stem, rsm=True, rsm_p=args.rsm_p)
    else:
        if args.net == "resnet34":
            net = models.MS_ResNet.resnet34(num_classes=num_classes, cifar_stem=cifar_stem)
        elif args.net == "resnet104":
            net = models.MS_ResNet.resnet104(num_classes=num_classes)
        elif args.net == "resnet110":
            net = models.MS_ResNet.resnet110(num_classes=num_classes)
        else:
            net = models.MS_ResNet.resnet18(num_classes=num_classes, cifar_stem=cifar_stem)

    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict):
        for k in ["state_dict", "model", "net"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    state = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)

    # register hooks before DataParallel wrapping
    spike_collector = None
    if args.save_spike_stats:
        spike_collector = SpikeStatsCollector(
            net,
            time_steps=args.time_steps,
            layout=args.spike_layout,
            hook_pattern=args.hook_pattern,
            leaf_only=True,
        )
        hooked = spike_collector.register()
        logger.info("Spike hook logging: enabled")
        logger.info("Hook pattern: %s", args.hook_pattern)
        logger.info("Hooked modules: %d", len(hooked))
        if len(hooked) > 0:
            for name in hooked:
                logger.info("  hook: %s", name)
        else:
            logger.warning("No modules matched hook pattern. non_zero/firing_rate will be empty.")

    net = net.to(device)
    if use_cuda:
        net = torch.nn.DataParallel(net)

    loss_fn = nn.CrossEntropyLoss().to(device)
    start = time.time()

    results = {
        "ts": datetime.now().isoformat(),
        "experiment": exp_name,
        "seed": seed_name,
        "weights": args.weights,
        "net": args.net,
        "dataset": args.dataset,
        "batch_size": int(args.b),
        "gpu": bool(use_cuda),
        "save_spike_stats": bool(args.save_spike_stats),
    }

    if args.dataset == "imagenet":
        loader = get_imagenet_loader(args.valdir, batch_size=args.b, num_workers=4, shuffle=False)
        eval_metrics = evaluate(
            net, loader, loss_fn, device=device,
            logger=logger, log_interval=args.log_interval,
            spike_collector=spike_collector,
        )
        finish = time.time()

        logger.info("loss: %.4f", eval_metrics["loss"])
        logger.info("top-1: %.4f", eval_metrics["top1"])
        logger.info("top-5: %.4f", eval_metrics["top5"])

        if args.save_spike_stats:
            non_zero_str = json.dumps(eval_metrics["non_zero"], indent=4)
            firing_rate_str = json.dumps(eval_metrics["firing_rate"], indent=4)
            logger.info("non_zero:\n%s", non_zero_str)
            logger.info("firing_rate:\n%s", firing_rate_str)

        logger.info("Time consumed: %.2fs", finish - start)

        results.update({
            "loss": float(eval_metrics["loss"]),
            "top1": float(eval_metrics["top1"]),
            "top5": float(eval_metrics["top5"]),
            "time_s": float(finish - start),
            "non_zero": eval_metrics["non_zero"] if args.save_spike_stats else {},
            "firing_rate": eval_metrics["firing_rate"] if args.save_spike_stats else {},
        })

    elif args.dataset == "cifar100":
        loader = get_cifar100_loader(args.data_root, batch_size=args.b, num_workers=4, shuffle=False)
        eval_metrics = evaluate(
            net, loader, loss_fn, device=device,
            logger=logger, log_interval=args.log_interval,
            spike_collector=spike_collector,
        )
        finish = time.time()

        logger.info("loss: %.4f", eval_metrics["loss"])
        logger.info("top-1: %.4f", eval_metrics["top1"])
        logger.info("top-5: %.4f", eval_metrics["top5"])

        if args.save_spike_stats:
            non_zero_str = json.dumps(eval_metrics["non_zero"], indent=4)
            firing_rate_str = json.dumps(eval_metrics["firing_rate"], indent=4)
            logger.info("non_zero:\n%s", non_zero_str)
            logger.info("firing_rate:\n%s", firing_rate_str)

        logger.info("Time consumed: %.2fs", finish - start)

        results.update({
            "loss": float(eval_metrics["loss"]),
            "top1": float(eval_metrics["top1"]),
            "top5": float(eval_metrics["top5"]),
            "time_s": float(finish - start),
            "non_zero": eval_metrics["non_zero"] if args.save_spike_stats else {},
            "firing_rate": eval_metrics["firing_rate"] if args.save_spike_stats else {},
        })

    else:  # cifar100c
        c_dir = args.cifar100c_dir if args.cifar100c_dir else args.data_root
        results["severity"] = int(args.severity)

        if args.corruption == "all":
            corruptions = [
                "gaussian_noise", "shot_noise", "impulse_noise",
                "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                "snow", "frost", "fog", "brightness",
                "contrast", "elastic_transform", "pixelate", "jpeg_compression"
            ]

            summary_rows = []
            per_corr = {}

            logger.info("Evaluating %d CIFAR-100-C corruptions (severity=%d)...",
                        len(corruptions), int(args.severity))

            for corr in corruptions:
                logger.info("=== Corruption: %s, severity=%d ===", corr, int(args.severity))
                loader = get_cifar100c_loader(
                    c_dir, corr, args.severity,
                    batch_size=args.b, num_workers=4, shuffle=False
                )

                eval_metrics = evaluate(
                    net, loader, loss_fn, device=device,
                    logger=logger, log_interval=args.log_interval,
                    log_suffix=f"-{corr}-sev{int(args.severity)}",
                    spike_collector=spike_collector,
                )

                logger.info("[%-18s] top-1: %.4f | top-5: %.4f | loss: %.4f",
                            corr, eval_metrics["top1"], eval_metrics["top5"], eval_metrics["loss"])

                summary_rows.append((corr, eval_metrics["top1"], eval_metrics["top5"], eval_metrics["loss"]))
                per_corr[corr] = {
                    "loss": float(eval_metrics["loss"]),
                    "top1": float(eval_metrics["top1"]),
                    "top5": float(eval_metrics["top5"]),
                }

            finish = time.time()

            mean_top1 = float(np.mean([r[1] for r in summary_rows])) if summary_rows else 0.0
            mean_top5 = float(np.mean([r[2] for r in summary_rows])) if summary_rows else 0.0
            mean_loss = float(np.mean([r[3] for r in summary_rows])) if summary_rows else 0.0

            logger.info("")
            logger.info("CIFAR-100-C summary (severity=%d):", int(args.severity))
            logger.info("  %-18s | %8s | %8s | %8s", "corruption", "top1", "top5", "loss")
            logger.info("  " + "-" * 56)
            for corr, t1, t5, ls in summary_rows:
                logger.info("  %-18s | %8.4f | %8.4f | %8.4f", corr, t1, t5, ls)
            logger.info("  " + "-" * 56)
            logger.info("  %-18s | %8.4f | %8.4f | %8.4f", "MEAN", mean_top1, mean_top5, mean_loss)
            logger.info("Time consumed: %.2fs", finish - start)

            results.update({
                "corruption": "all",
                "mean_loss": float(mean_loss),
                "mean_top1": float(mean_top1),
                "mean_top5": float(mean_top5),
                "per_corruption": per_corr,
                "time_s": float(finish - start),
            })

        else:
            loader = get_cifar100c_loader(
                c_dir, args.corruption, args.severity,
                batch_size=args.b, num_workers=4, shuffle=False
            )

            eval_metrics = evaluate(
                net, loader, loss_fn, device=device,
                logger=logger, log_interval=args.log_interval,
                log_suffix=f"-{args.corruption}-sev{int(args.severity)}",
                spike_collector=spike_collector,
            )
            finish = time.time()

            logger.info("loss: %.4f", eval_metrics["loss"])
            logger.info("top-1: %.4f", eval_metrics["top1"])
            logger.info("top-5: %.4f", eval_metrics["top5"])

            if args.save_spike_stats:
                non_zero_str = json.dumps(eval_metrics["non_zero"], indent=4)
                firing_rate_str = json.dumps(eval_metrics["firing_rate"], indent=4)
                logger.info("non_zero:\n%s", non_zero_str)
                logger.info("firing_rate:\n%s", firing_rate_str)

            logger.info("Time consumed: %.2fs", finish - start)

            results.update({
                "corruption": args.corruption,
                "severity": int(args.severity),
                "loss": float(eval_metrics["loss"]),
                "top1": float(eval_metrics["top1"]),
                "top5": float(eval_metrics["top5"]),
                "time_s": float(finish - start),
                "non_zero": eval_metrics["non_zero"] if args.save_spike_stats else {},
                "firing_rate": eval_metrics["firing_rate"] if args.save_spike_stats else {},
            })

    core = net.module if hasattr(net, "module") else net
    param_count = sum(p.numel() for p in core.parameters())
    logger.info("Parameter numbers: %d", param_count)
    results["params"] = int(param_count)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("========== TEST END ==========")

    if spike_collector is not None:
        spike_collector.close()