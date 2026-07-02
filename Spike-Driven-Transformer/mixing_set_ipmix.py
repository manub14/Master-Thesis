#!/usr/bin/env python3
import os, math, argparse, random
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageColor

# ---------------------------
# Utils
# ---------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def lerp(a, b, t):
    return a + (b - a) * t

def rand_choice(seq):
    return seq[random.randrange(len(seq))]

def rand_color():
    # Return an RGB tuple; prefer vivid-ish colors
    h = random.random()
    s = random.uniform(0.5, 1.0)
    v = random.uniform(0.7, 1.0)
    return hsv_to_rgb(h, s, v)

def hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = int(255 * v * (1 - s))
    q = int(255 * v * (1 - f * s))
    t = int(255 * v * (1 - (1 - f) * s))
    v = int(255 * v)
    i = i % 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

def make_random_palette(n=256):
    """Random gradient palette."""
    # choose 3-5 anchor colors & interpolate
    k = random.randint(3, 5)
    anchors = [rand_color() for _ in range(k)]
    # positions
    xs = sorted([random.random() for _ in range(k)])
    # build palette
    pal = []
    for i in range(n):
        t = i / (n - 1)
        # find segment
        j = 0
        while j < k - 1 and not (xs[j] <= t <= xs[j+1]):
            j += 1
        if j == k - 1:  # last anchor
            c = anchors[-1]
        else:
            t0, t1 = xs[j], xs[j+1]
            w = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            a, b = anchors[j], anchors[j+1]
            c = (int(lerp(a[0], b[0], w)),
                 int(lerp(a[1], b[1], w)),
                 int(lerp(a[2], b[2], w)))
        pal.append(c)
    return pal

# ---------------------------
# Escape-time fractals (Mandelbrot / Julia)
# ---------------------------

@dataclass
class EscapeTimeParams:
    max_iter: int
    center: complex
    scale: float   # viewport half-width in complex plane
    c: complex = None  # for Julia; None means Mandelbrot

def random_escape_time_params(kind: str, img_size: int) -> EscapeTimeParams:
    # Random viewport and iterations; zoom ranges chosen to diversify patterns
    max_iter = random.randint(64, 256)
    # Random zoom/center; slightly favor interesting regions
    if kind == "mandelbrot":
        # classic viewport around (-0.5, 0)
        base_center = complex(-0.5, 0.0)
        jitter = complex(random.uniform(-0.6, 0.6), random.uniform(-0.6, 0.6))
        center = base_center + jitter * random.uniform(0.0, 0.5)
        # scale determines zoom; smaller -> deeper zoom
        scale = 1.5 * (0.5 ** random.uniform(0.0, 4.0))  # ~1.5 down to ~0.09
        return EscapeTimeParams(max_iter=max_iter, center=center, scale=scale)
    else:
        # Julia: pick c near Mandelbrot cardioid/bulb for richer structure
        # sample c by picking Mandelbrot-like point:
        rho = random.uniform(0.0, 1.0)
        theta = random.uniform(0, 2*math.pi)
        c = 0.7885 * rho * complex(math.cos(theta), math.sin(theta))
        center = complex(0.0, 0.0)
        scale = 1.5 * (0.5 ** random.uniform(0.0, 4.0))
        return EscapeTimeParams(max_iter=max_iter, center=center, scale=scale, c=c)

def render_escape_time(kind: str, size: int, palette: List[Tuple[int,int,int]], bg=None, ss=2) -> Image.Image:
    """Render Mandelbrot or Julia at resolution `size` with simple supersampling."""
    p = random_escape_time_params(kind, size)
    W = H = size * ss
    xs = np.linspace(p.center.real - p.scale, p.center.real + p.scale, W, dtype=np.float32)
    ys = np.linspace(p.center.imag - p.scale, p.center.imag + p.scale, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    Z0 = X + 1j*Y

    if p.c is None:
        C = Z0.copy()
        Z = np.zeros_like(Z0)
    else:
        C = np.full_like(Z0, p.c, dtype=np.complex64)
        Z = Z0.copy()

    it = np.zeros(Z0.shape, dtype=np.int32)
    mask = np.ones(Z0.shape, dtype=bool)
    for k in range(p.max_iter):
        Z[mask] = Z[mask]*Z[mask] + C[mask]
        escaped = np.abs(Z) > 2.0
        newly = escaped & mask
        it[newly] = k
        mask &= ~newly
        if not mask.any(): break
    it[mask] = p.max_iter

    it_norm = it.astype(np.float32) / p.max_iter
    # colorize
    pal_idx = np.clip((it_norm * (len(palette)-1)).astype(np.int32), 0, len(palette)-1)
    rgb = np.stack([np.array([palette[i][ch] for i in pal_idx.ravel()]).reshape(pal_idx.shape) for ch in range(3)], axis=-1)
    # optional background override for interior points (not escaped)
    if bg is not None:
        interior = (it == p.max_iter)
        for ch in range(3):
            rgb[..., ch][interior] = bg[ch]
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    if ss > 1:
        img = img.resize((size, size), Image.LANCZOS)
    return img

# ---------------------------
# IFS fractals
# ---------------------------

@dataclass
class Affine:
    a: float; b: float; c: float; d: float; e: float; f: float
    p: float  # probability

def barnsley_fern() -> List[Affine]:
    # classic fern
    return [
        Affine(0.0,    0.0,    0.0, 0.16, 0.0, 0.0, 0.01),
        Affine(0.85,   0.04,  -0.04, 0.85, 0.0, 1.6, 0.85),
        Affine(0.2,   -0.26,   0.23, 0.22, 0.0, 1.6, 0.07),
        Affine(-0.15,  0.28,   0.26, 0.24, 0.0, 0.44,0.07),
    ]

def sierpinski() -> List[Affine]:
    s = 0.5
    return [
        Affine(s,0,0,s,0,0,1/3),
        Affine(s,0,0,s,1,0,1/3),
        Affine(s,0,0,s,0.5,math.sqrt(3)/2,1/3),
    ]

def spiral_set() -> List[Affine]:
    # simple spiral-ish system
    return [
        Affine(0.787879, -0.424242, 0.242424, 0.859848, 1.758647, 1.408065, 0.9),
        Affine(-0.121212, 0.257576, 0.151515, 0.053030, -6.721654, 1.377236, 0.05),
        Affine(0.181818, -0.136364, 0.090909, 0.181818, 6.086107, 1.568035, 0.05),
    ]

def random_affines(n=3) -> List[Affine]:
    # generate a stable random IFS (contractive maps)
    aff = []
    probs = np.random.dirichlet(np.ones(n))
    for i in range(n):
        # small scalings for contractiveness
        a = random.uniform(-0.6, 0.6)
        d = random.uniform(-0.6, 0.6)
        b = random.uniform(-0.3, 0.3)
        c = random.uniform(-0.3, 0.3)
        e = random.uniform(-1.0, 1.0)
        f = random.uniform(-1.0, 1.0)
        aff.append(Affine(a,b,c,d,e,f,float(probs[i])))
    return aff

def sample_ifs(affs: List[Affine], num_points=60000, burn_in=100) -> np.ndarray:
    # pick according to cumulative probs
    ps = np.array([a.p for a in affs], dtype=np.float64)
    ps = ps / ps.sum()
    cdf = np.cumsum(ps)
    x, y = 0.0, 0.0
    pts = []
    for i in range(num_points + burn_in):
        r = random.random()
        j = int(np.searchsorted(cdf, r))
        A = affs[j]
        x, y = A.a * x + A.b * y + A.e, A.c * x + A.d * y + A.f
        if i >= burn_in:
            pts.append((x, y))
    return np.array(pts, dtype=np.float32)

def render_ifs(size: int, palette: List[Tuple[int,int,int]], bg=None) -> Image.Image:
    # choose a system
    system = rand_choice([barnsley_fern(), sierpinski(), spiral_set(), random_affines(random.randint(3,5))])
    pts = sample_ifs(system, num_points=random.randint(40000, 90000))
    # normalize to [0,1]x[0,1] with small margins
    x = pts[:,0]; y = pts[:,1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx = xmax - xmin; dy = ymax - ymin
    x = (x - xmin) / (dx + 1e-8)
    y = (y - ymin) / (dy + 1e-8)
    # rasterize density (log)
    H = W = size
    img = np.zeros((H, W), dtype=np.float32)
    xs = np.clip((x * (W - 1)).astype(np.int32), 0, W - 1)
    ys = np.clip((1.0 - y) * (H - 1), 0, H - 1).astype(np.int32)
    for i in range(xs.shape[0]):
        img[ys[i], xs[i]] += 1.0
    img = np.log1p(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    pal_idx = np.clip((img * (len(palette)-1)).astype(np.int32), 0, len(palette)-1)

    # Build rgb as float32 for safe blending, cast to uint8 at the end
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for ch in range(3):
        rgb[..., ch] = np.array([palette[i][ch] for i in pal_idx.ravel()]).reshape(H, W)

    if bg is not None:
        # blend towards bg using low-density mask (vectorized)
        mask = img < 0.1
        for ch in range(3):
            rgb_ch = rgb[..., ch]
            # elementwise blend
            rgb_ch[mask] = 0.7 * rgb_ch[mask] + 0.3 * float(bg[ch])
            rgb[..., ch] = rgb_ch

    # clip and convert to uint8
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")

# ---------------------------
# Main generator
# ---------------------------

def make_one(size: int) -> Image.Image:
    # choose type
    palette = make_random_palette()
    bg = rand_color() if random.random() < 0.6 else None
    choice = random.random()
    if choice < 0.25:
        return render_escape_time("mandelbrot", size, palette, bg=bg, ss=2)
    elif choice < 0.50:
        return render_escape_time("julia", size, palette, bg=bg, ss=2)
    else:
        return render_ifs(size, palette, bg=bg)

def main():
    ap = argparse.ArgumentParser("Generate an IPMix-style fractal mixing set (ImageFolder).")
    ap.add_argument("--out", required=True, help="Output directory for ImageFolder (e.g., /data/mixing_set)")
    ap.add_argument("--total", type=int, default=13000, help="Total images to generate (default 13000).")
    ap.add_argument("--img-size", type=int, default=32, help="Image size (default 32).")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ImageFolder layout: /out/fractal/*.png
    class_dir = os.path.join(args.out, "fractal")
    ensure_dir(class_dir)

    print(f"Generating {args.total} images into: {class_dir}")
    for i in range(args.total):
        img = make_one(args.img_size)
        img.save(os.path.join(class_dir, f"{i:06d}.png"))
        if (i+1) % 500 == 0:
            print(f"... {i+1}/{args.total}")

    print("Done.")

if __name__ == "__main__":
    main()
