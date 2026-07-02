# models/postenc_pipeline.py

import torch.nn as nn


class PostEncAugPipeline(nn.Module):
    """
    Sequential pipeline for post-encoding augmentations.
    Keeps augmentations modular and avoids clashes.
    """

    def __init__(self, aug_list):
        super(PostEncAugPipeline, self).__init__()
        self.augs = nn.ModuleList([a for a in aug_list if a is not None])

    def forward(self, x):
        for aug in self.augs:
            x = aug(x)
        return x