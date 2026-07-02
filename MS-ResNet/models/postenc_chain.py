import torch.nn as nn

class PostEncAugChain(nn.Module):
    def __init__(self, augs):
        super().__init__()
        self.augs = nn.ModuleList([a for a in augs if a is not None])

    def forward(self, spikes):
        out = spikes
        for aug in self.augs:
            out = aug(out)
        return out