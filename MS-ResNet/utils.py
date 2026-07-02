# utils.py
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler


def get_network(args):
    """
    Build the network given args.net and pass dataset-specific knobs:
      - num_classes
      - cifar_stem
      - post-encoding patchdropout params: patchdrop_keep, patchdrop_size
      - post-encoding timeshuffle params: tshift_*
      - post-encoding timemask params: tmask_*
      - post-encoding patchshuffle params: pshuf_*
      - pre-encoding timemask params: pre_tmask_*
      - pre-encoding timeshuffle params: pre_tshift_*
      - pre-encoding temporal jitter params: pre_tjitter_*
      - pre-encoding timemix params: pre_tmix_*
      - pre-encoding fulldimmix params: pre_fdmix_*

    NOTE (PatchMix / HoleFill / LocalTimeShuffle / Center Patch MinLift / other post-encoding augs):
      These are attached or applied in train_amp.py via net.postenc_aug or
      in the outer image batch path before net(images), so no constructor args
      are required here for them unless you explicitly want them hard-wired
      inside MS_ResNet.py.

    NOTE (Center Patch MinLift specifically):
      Center Patch MinLift is a PRE-encoding image-domain augmentation applied
      directly on BCHW image tensors in train_amp.py before the network forward.
      Therefore it should NOT be forwarded through get_network(...) into
      MS_ResNet.py unless you later decide to hard-wire it inside the model.
    """
    # dataset knobs
    num_classes = getattr(args, "num_classes", 1000)
    cifar_stem = getattr(args, "cifar_stem", False)

    # PatchDropout
    patchdrop_keep = getattr(args, "patchdrop_keep", 1.0)
    patchdrop_size = getattr(args, "patchdrop_size", 4)

    # TimeShuffle
    tshift_p = getattr(args, "tshift_p", 0.0)
    tshift_max = getattr(args, "tshift_max", 1)
    tshift_fold_k = getattr(args, "tshift_fold_k", 32)
    tshift_alpha = getattr(args, "tshift_alpha", 0.3)

    # TimeMask
    tmask_p = getattr(args, "tmask_p", 0.0)
    tmask_num = getattr(args, "tmask_num", 1)
    tmask_max_frac = getattr(args, "tmask_max_frac", 0.25)
    tmask_min_len = getattr(args, "tmask_min_len", 1)
    tmask_mode = getattr(args, "tmask_mode", "zero")
    tmask_noise_std = getattr(args, "tmask_noise_std", 0.05)
    tmask_layout = getattr(args, "tmask_layout", "TB")
    tmask_same_on_batch = getattr(args, "tmask_same_on_batch", False)
    tmask_per_channel = getattr(args, "tmask_per_channel", False)
    tmask_channel_groups = getattr(args, "tmask_channel_groups", 1)

    # PatchShuffle
    pshuf_p = getattr(args, "pshuf_p", 0.0)
    pshuf_size = getattr(args, "pshuf_size", 2)
    pshuf_layout = getattr(args, "pshuf_layout", "TB")
    pshuf_per_time = getattr(args, "pshuf_per_time", False)
    pshuf_same_on_batch = getattr(args, "pshuf_same_on_batch", False)

    # NEW: Pre-encoding TimeMask
    pre_tmask_enable = getattr(args, "pre_tmask_enable", False)
    pre_tmask_p = getattr(args, "pre_tmask_p", 0.0)
    pre_tmask_num = getattr(args, "pre_tmask_num", 1)
    pre_tmask_max_frac = getattr(args, "pre_tmask_max_frac", 0.25)
    pre_tmask_min_len = getattr(args, "pre_tmask_min_len", 1)
    pre_tmask_mode = getattr(args, "pre_tmask_mode", "zero")
    pre_tmask_noise_std = getattr(args, "pre_tmask_noise_std", 0.05)
    pre_tmask_layout = getattr(args, "pre_tmask_layout", "TBCHW")
    pre_tmask_same_on_batch = getattr(args, "pre_tmask_same_on_batch", False)
    pre_tmask_per_channel = getattr(args, "pre_tmask_per_channel", False)
    pre_tmask_channel_groups = getattr(args, "pre_tmask_channel_groups", 1)
    pre_tmask_apply_in_eval = getattr(args, "pre_tmask_apply_in_eval", False)

    # NEW: Pre-encoding TimeShuffle
    pre_tshift_enable = getattr(args, "pre_tshift_enable", False)
    pre_tshift_p = getattr(args, "pre_tshift_p", 0.0)
    pre_tshift_max = getattr(args, "pre_tshift_max", 1)
    pre_tshift_foldk = getattr(args, "pre_tshift_foldk", 32)
    pre_tshift_alpha = getattr(args, "pre_tshift_alpha", 0.3)
    pre_tshift_apply_in_eval = getattr(args, "pre_tshift_apply_in_eval", False)

    # NEW: Pre-encoding Temporal Jitter
    pre_tjitter_enable = getattr(args, "pre_tjitter_enable", False)
    pre_tjitter_p = getattr(args, "pre_tjitter_p", 0.0)
    pre_tjitter_max = getattr(args, "pre_tjitter_max", 1)
    pre_tjitter_per_sample = getattr(args, "pre_tjitter_per_sample", False)
    pre_tjitter_layout = getattr(args, "pre_tjitter_layout", "TBCHW")
    pre_tjitter_apply_in_eval = getattr(args, "pre_tjitter_apply_in_eval", False)

    # NEW: Pre-encoding TimeMix
    pre_tmix_enable = getattr(args, "pre_tmix_enable", False)
    pre_tmix_p = getattr(args, "pre_tmix_p", 0.0)
    pre_tmix_alpha = getattr(args, "pre_tmix_alpha", 0.3)
    pre_tmix_groups = getattr(args, "pre_tmix_groups", 32)
    pre_tmix_random_split = getattr(args, "pre_tmix_random_split", True)
    pre_tmix_apply_in_eval = getattr(args, "pre_tmix_apply_in_eval", False)

    # NEW: Pre-encoding FullDimMix
    pre_fdmix_enable = getattr(args, "pre_fdmix_enable", False)
    pre_fdmix_p = getattr(args, "pre_fdmix_p", 0.0)
    pre_fdmix_alpha = getattr(args, "pre_fdmix_alpha", 0.5)
    pre_fdmix_layout = getattr(args, "pre_fdmix_layout", "TBCHW")
    pre_fdmix_apply_in_eval = getattr(args, "pre_fdmix_apply_in_eval", False)

    if args.net == "resnet18":
        from models.MS_ResNet import resnet18
        net = resnet18(
            num_classes=num_classes,
            cifar_stem=cifar_stem,
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
        )

    elif args.net == "resnet34":
        from models.MS_ResNet import resnet34
        net = resnet34(
            num_classes=num_classes,
            cifar_stem=cifar_stem,
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
        )

    elif args.net == "resnet104":
        from models.MS_ResNet import resnet104
        net = resnet104(num_classes=num_classes)

    elif args.net == "resnet110":
        from models.MS_ResNet import resnet110
        net = resnet110(
            num_classes=num_classes,
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
        )

    elif args.net == "resnet56":
        from models.MS_ResNet import resnet56
        net = resnet56(
            num_classes=num_classes,
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
        )

    elif args.net == "resnet44":
        from models.MS_ResNet import resnet44
        net = resnet44(
            num_classes=num_classes,
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
        )

    elif args.net == "resnet32":
        from models.MS_ResNet import resnet32
        net = resnet32(
            num_classes=num_classes,
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
        )

    else:
        print(f'network "{args.net}" is not supported')
        sys.exit(1)

    return net


def _make_sampler(ds, enabled, shuffle):
    if enabled:
        return DistributedSampler(ds, shuffle=shuffle)
    return None


def get_training_dataloader(dataset,
                            data_root=None,
                            traindir=None,
                            sampler=None,
                            batch_size=16,
                            num_workers=2,
                            shuffle=True):
    dataset = dataset.lower()

    if dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                normalize,
            ])
        )

    elif dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(
            root=data_root, train=True, download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )

    elif dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(
            root=data_root, train=True, download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_sampler = _make_sampler(train_set, sampler is not None, shuffle=True)
    if train_sampler is not None:
        shuffle = False

    train_loader = DataLoader(
        train_set,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        sampler=train_sampler,
    )
    return train_loader, train_sampler


def get_test_dataloader(dataset,
                        data_root=None,
                        valdir=None,
                        sampler=None,
                        batch_size=16,
                        num_workers=2,
                        shuffle=False):
    dataset = dataset.lower()

    if dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        test_set = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        )

    elif dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
        test_set = datasets.CIFAR10(
            root=data_root, train=False, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

    elif dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        test_set = datasets.CIFAR100(
            root=data_root, train=False, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    test_sampler = _make_sampler(test_set, sampler is not None, shuffle=False)
    if test_sampler is not None:
        shuffle = False

    test_loader = DataLoader(
        test_set,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        sampler=test_sampler,
    )
    return test_loader, test_sampler