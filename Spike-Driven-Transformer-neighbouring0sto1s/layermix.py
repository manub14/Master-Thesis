# layermix.py
import os, random
from typing import List
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

# ---- light augmentation ops (tensor in [0,1]) ----
def _rand_brightness(x: Tensor, mag: int):  # mag ~ 0..30-ish
    factor = 1.0 + random.uniform(-0.2, 0.2) * (mag / 10)
    return torch.clamp(F.adjust_brightness(x, factor), 0, 1)

def _rand_contrast(x: Tensor, mag: int):
    factor = 1.0 + random.uniform(-0.2, 0.2) * (mag / 10)
    return torch.clamp(F.adjust_contrast(x, factor), 0, 1)

def _rand_saturation(x: Tensor, mag: int):
    factor = 1.0 + random.uniform(-0.2, 0.2) * (mag / 10)
    return torch.clamp(F.adjust_saturation(x, factor), 0, 1)

def _rand_hue(x: Tensor, mag: int):
    # hue expects [-0.5, 0.5]
    delta = 0.02 * (mag / 10) * random.choice([-1, 1])
    return torch.clamp(F.adjust_hue(x, delta), 0, 1)

AUG_FNS = [_rand_brightness, _rand_contrast, _rand_saturation, _rand_hue]

# ---- blending ops ----
def _mixup(a: Tensor, b: Tensor, alpha: float):
    return a * (1 - alpha) + b * alpha

def _add(a: Tensor, b: Tensor, alpha: float):
    return torch.clamp(a + alpha * (b - 0.5), 0, 1)

def _mul(a: Tensor, b: Tensor, alpha: float):
    return torch.clamp(a * (1 - alpha) + a * b * alpha, 0, 1)

BLEND_FNS = [_mixup, _add, _mul]

class RandomLayerMixTransform:
    """
    Apply the LayerMix pipeline to a single image tensor in [0,1].
    Requires a mixing-set directory with images (e.g., fractals).
    """
    def __init__(self, mixing_set_dir: str, p: float = 1.0,
                 ratio: float = 0.5, mag: int = 10, out_size: int = 32):
        assert os.path.isdir(mixing_set_dir), f"mixing-set not found: {mixing_set_dir}"
        self.p = p
        self.ratio = ratio
        self.mag = mag
        self.out_size = out_size

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        self.paths: List[str] = []
        # NEW: search recursively
        for root, _, files in os.walk(mixing_set_dir):
            for fn in files:
                if fn.lower().endswith(exts):
                    self.paths.append(os.path.join(root, fn))

        assert len(self.paths) > 0, \
            f"No images under mixing-set dir (searched recursively): {mixing_set_dir}"

        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((out_size, out_size), antialias=True)

    def _load_mixing_pic(self) -> Tensor:
        p = random.choice(self.paths)
        img = Image.open(p).convert("RGB")
        img = self.resize(img)
        return self.to_tensor(img)  # [0,1]

    def _one_aug(self, x: Tensor) -> Tensor:
        fn = random.choice(AUG_FNS)
        return fn(x, self.mag)

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x
        # Step 0: pick ops
        step = random.randint(0, 2)  # like ref impl: 0,1,2 branches
        x0 = x.clone()
        aug = self._one_aug

        # first aug
        x = aug(x)
        if step == 0:
            return x

        # blend with another augmented copy
        x2 = aug(x0)
        blend = random.choice(BLEND_FNS)
        x = torch.clamp(blend(x, x2, self.ratio), 0, 1)
        if step == 1:
            return x

        # blend with a mixing-set image + final aug
        mix_pic = self._load_mixing_pic()
        mix_pic = mix_pic.to(x.device)
        x = torch.clamp(blend(x, mix_pic, self.ratio), 0, 1)
        x = aug(x)
        return x


# ---- CIFAR-100-C helper Dataset for evaluation ----
class CIFAR100C(Dataset):
    """
    Loads one corruption+severity slice (10k images) from CIFAR-100-C.
    root contains e.g. snow.npy, gaussian_noise.npy, ... and labels.npy
    """
    def __init__(self, root: str, corruption: str, severity: int, transform=None):
        assert 1 <= severity <= 5
        self.transform = transform
        x = np.load(os.path.join(root, f"{corruption}.npy"))  # [50000, 32, 32, 3], uint8
        y = np.load(os.path.join(root, "labels.npy"))
        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = x[start:end]
        self.targets = y[start:end].astype(np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])
