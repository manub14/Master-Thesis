# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms

import models.IPMix_utils as ipmix_utils


class IPMixConfig(object):
    def __init__(self,
                 k=3,
                 t=3,
                 aug_severity=3,
                 no_jsd=False,
                 jsd_weight=12.0,
                 img_size=32,
                 mixing_resize=36):
        self.k = int(k)
        self.t = int(t)
        self.aug_severity = int(aug_severity)
        self.no_jsd = bool(no_jsd)
        self.jsd_weight = float(jsd_weight)
        self.img_size = int(img_size)
        self.mixing_resize = int(mixing_resize)


def build_ipmix_preprocess(mean, std):
    return tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=mean, std=std),
    ])


def build_ipmix_mixing_set(mixing_set_root, img_size=32, mixing_resize=36):
    if (mixing_set_root is None) or (not os.path.isdir(mixing_set_root)):
        raise RuntimeError(
            "IPMix mixing set path is invalid: {}".format(mixing_set_root)
        )

    mixing_transform = tv_transforms.Compose([
        tv_transforms.Resize(mixing_resize),
        tv_transforms.RandomCrop(img_size),
    ])

    mixing_set = tv_datasets.ImageFolder(
        root=mixing_set_root,
        transform=mixing_transform
    )
    return mixing_set


def _ensure_pil_rgb(image):
    if not isinstance(image, Image.Image):
        raise TypeError("IPMix expects PIL images before preprocessing.")
    return image.convert("RGB")


def _augment_input(image, aug_severity):
    aug_list = ipmix_utils.augmentations_all
    op = random.choice(aug_list)
    return op(image.copy(), aug_severity)


def ipmix_one(image, mixing_pic, preprocess, cfg):
    """
    Original IPMix-style single mixed sample.
    """
    image = _ensure_pil_rgb(image)
    mixing_pic = _ensure_pil_rgb(mixing_pic)

    mixings = ipmix_utils.mixings
    patch_mixing = ipmix_utils.patch_mixing

    patch_sizes = [4, 8, 16, 32]
    patch_sizes = [p for p in patch_sizes if p <= cfg.img_size]
    if len(patch_sizes) == 0:
        patch_sizes = [cfg.img_size]

    mixing_modes = ["IMG", "P"]

    ws = np.float32(np.random.dirichlet([1.0] * cfg.k))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(preprocess(image))

    for i in range(cfg.k):
        mixed = image.copy()
        mixing_way = random.choice(mixing_modes)

        if mixing_way == "P":
            for _ in range(np.random.randint(cfg.t + 1)):
                patch_size = int(random.choice(patch_sizes))
                mixed_op = random.choice(mixings)
                mixed = patch_mixing(mixed, mixing_pic, patch_size, mixed_op, beta=1)
        else:
            for _ in range(np.random.randint(cfg.t + 1)):
                mixed = _augment_input(mixed, cfg.aug_severity)

        mix = mix + ws[i] * preprocess(mixed)

    clean = preprocess(image)
    mix_result = (1.0 - m) * clean + m * mix
    return mix_result


class IPMixDataset(Dataset):
    """
    Dataset wrapper for IPMix pre-encoding.

    If no_jsd = False:
        returns ((clean, aug1, aug2), target)

    If no_jsd = True:
        returns (mixed, target)
    """
    def __init__(self, base_dataset, mixing_set, preprocess, cfg):
        self.base_dataset = base_dataset
        self.mixing_set = mixing_set
        self.preprocess = preprocess
        self.cfg = cfg

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, target = self.base_dataset[index]
        image = _ensure_pil_rgb(image)

        rnd_idx = random.randrange(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[rnd_idx]
        mixing_pic = _ensure_pil_rgb(mixing_pic)

        if self.cfg.no_jsd:
            mixed = ipmix_one(image, mixing_pic, self.preprocess, self.cfg)
            return mixed, target

        clean = self.preprocess(image)
        aug1 = ipmix_one(image, mixing_pic, self.preprocess, self.cfg)
        aug2 = ipmix_one(image, mixing_pic, self.preprocess, self.cfg)
        return (clean, aug1, aug2), target


def build_ipmix_cifar_datasets(data_root,
                               dataset_name,
                               mixing_set_root,
                               mean,
                               std,
                               cfg):
    ds_name = str(dataset_name).lower()

    if ds_name == "cifar100":
        dataset_cls = tv_datasets.CIFAR100
    elif ds_name == "cifar10":
        dataset_cls = tv_datasets.CIFAR10
    else:
        raise ValueError(
            "IPMix preencoding currently supports only cifar10/cifar100, got {}.".format(dataset_name)
        )

    preprocess = build_ipmix_preprocess(mean=mean, std=std)

    train_base = dataset_cls(
        root=data_root,
        train=True,
        transform=None,
        download=True
    )

    test_set = dataset_cls(
        root=data_root,
        train=False,
        transform=preprocess,
        download=True
    )

    mixing_set = build_ipmix_mixing_set(
        mixing_set_root=mixing_set_root,
        img_size=cfg.img_size,
        mixing_resize=cfg.mixing_resize
    )

    train_set = IPMixDataset(
        base_dataset=train_base,
        mixing_set=mixing_set,
        preprocess=preprocess,
        cfg=cfg
    )

    return train_set, test_set


def compute_ipmix_loss(logits_all, target, jsd_weight=12.0):
    """
    logits_all: [3B, C]
    target:     [B]
    returns: total_loss, logits_clean
    """
    batch_size = target.size(0)
    expected = 3 * batch_size

    if logits_all.size(0) != expected:
        raise ValueError(
            "Expected 3*B logits in IPMix JSD path, got {} logits for batch size {}.".format(
                logits_all.size(0), batch_size
            )
        )

    logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, batch_size, dim=0)

    ce = F.cross_entropy(logits_clean, target)

    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)

    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1.0).log()

    jsd = (
        F.kl_div(p_mixture, p_clean, reduction="batchmean") +
        F.kl_div(p_mixture, p_aug1, reduction="batchmean") +
        F.kl_div(p_mixture, p_aug2, reduction="batchmean")
    ) / 3.0

    return ce + float(jsd_weight) * jsd, logits_clean