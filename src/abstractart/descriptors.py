"""Interpretable visual descriptors, used to describe clusters (not to build them).

Twenty measurable quantities, each with a plain meaning, computed on the painting at its
native aspect ratio (long side resized to 512 px, no padding). They are a curated, cleaned
subset of the v1 handcrafted features: no file-size or HOG features, and no reflective
padding that would fake symmetry.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

SIZE = 512
EDGE_THRESHOLD = 0.1      # Sobel magnitude (normalised to ~[0, 1.4]) above which a pixel is an edge
CHROMA_THRESHOLD = 15.0   # CIELAB chroma above which a pixel counts as coloured
AXIS_TOLERANCE = np.deg2rad(10)

DESCRIPTIONS = {
    # colour
    "lightness": "mean CIELAB lightness (0 = black, 100 = white)",
    "lightness_contrast": "std of lightness",
    "chroma": "mean CIELAB chroma (colour intensity)",
    "colourfulness": "Hasler & Süsstrunk colourfulness",
    "chromatic_share": "share of clearly coloured pixels",
    "warm_ratio": "share of coloured pixels with warm hues (red, orange, yellow)",
    "hue_diversity": "entropy of the hue histogram of coloured pixels (0-1)",
    "palette_size": "number of colours (216-colour quantisation) covering 90% of the canvas",
    # surface and detail
    "edge_density": "share of edge pixels",
    "local_entropy": "mean grey-level entropy of 32x32 patches (texture complexity)",
    "flat_ratio": "share of 16x16 patches of almost uniform tone (flat areas)",
    "spectral_slope": "slope of the power spectrum (high = smooth, low = fine detail)",
    "fractal_dimension": "box-counting dimension of the edge map",
    # geometry and composition
    "rectilinearity": "share of edge energy aligned with the horizontal/vertical axes",
    "horizontal_bias": "horizontal vs vertical edge energy (0.5 = balanced)",
    "orientation_entropy": "entropy of edge orientations (1 = all directions)",
    "symmetry_lr": "left-right mirror similarity (1 = symmetric)",
    "centre_focus": "edge density of the central ninth relative to the whole canvas",
    "periodicity": "strength of repeated motifs (grids, dots, stripes); ~0 for lines and gradients",
    "aspect_ratio": "width / height of the canvas",
}
NAMES = list(DESCRIPTIONS)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _blocks(x: np.ndarray, size: int) -> np.ndarray:
    h, w = (x.shape[0] // size) * size, (x.shape[1] // size) * size
    return x[:h, :w].reshape(h // size, size, w // size, size).swapaxes(1, 2).reshape(-1, size, size)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def _periodicity(gray: np.ndarray, size: int = 256, n_directions: int = 36) -> float:
    """Strongest "rebound" of the autocorrelation along a direction.

    A repeated motif makes the autocorrelation drop then rise again at the motif period;
    a long straight edge or a gradient only makes it decrease, so they score ~0. Working at
    256 px with a band-pass filter removes JPEG 8x8 blocking, which otherwise looks periodic
    on flat canvases.
    """
    h, w = gray.shape
    scale = size / max(h, w)
    g = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize(
        (max(16, round(w * scale)), max(16, round(h * scale))), Image.LANCZOS), dtype=np.float64) / 255
    band = ndimage.gaussian_filter(g, 0.7) - ndimage.gaussian_filter(g, 2.5)
    if band.std() < 0.01:  # no fine structure at all
        return 0.0
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(band - band.mean())) ** 2).real)
    ac /= ac.max()
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    t = np.arange(1, min(cy, cx) // 2)
    best = 0.0
    for angle in np.linspace(0, np.pi, n_directions, endpoint=False):
        profile = ndimage.map_coordinates(ac, [cy + t * np.sin(angle), cx + t * np.cos(angle)], order=1)
        best = max(best, float((profile - np.minimum.accumulate(profile)).max()))
    return best


def load(path: Path) -> np.ndarray:
    img = Image.open(path)
    img.draft("RGB", (2 * SIZE, 2 * SIZE))
    img = img.convert("RGB")
    scale = SIZE / max(img.size)
    img = img.resize((max(16, round(img.width * scale)), max(16, round(img.height * scale))), Image.BICUBIC)
    return np.asarray(img, dtype=np.float64) / 255.0


def compute(rgb: np.ndarray) -> dict[str, float]:
    h, w = rgb.shape[:2]
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    lab = _srgb_to_lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    coloured = chroma > CHROMA_THRESHOLD
    hue = np.degrees(np.arctan2(b, a)) % 360

    R, G, B = (255 * rgb).transpose(2, 0, 1)
    rg, yb = R - G, 0.5 * (R + G) - B
    colourfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    quant = (np.clip(rgb, 0, 0.999) * 6).astype(int) @ np.array([36, 6, 1])
    counts = np.sort(np.bincount(quant.ravel(), minlength=216))[::-1]
    palette_size = int(np.searchsorted(np.cumsum(counts), 0.9 * counts.sum()) + 1)

    hue_hist = np.histogram(hue[coloured], bins=12, range=(0, 360))[0]
    warm = (hue[coloured] >= 330) | (hue[coloured] <= 100)

    # gradients (Sobel response to a unit step is 4)
    gx, gy = ndimage.sobel(gray, axis=1) / 4, ndimage.sobel(gray, axis=0) / 4
    mag = np.hypot(gx, gy)
    edges = mag > EDGE_THRESHOLD
    theta = np.arctan2(gy, gx)
    m = mag[edges]
    vertical_edges = np.abs(np.cos(theta[edges])) > np.cos(AXIS_TOLERANCE)    # gradient along x
    horizontal_edges = np.abs(np.sin(theta[edges])) > np.cos(AXIS_TOLERANCE)  # gradient along y
    e_v, e_h, e_all = m[vertical_edges].sum(), m[horizontal_edges].sum(), m.sum() + 1e-9
    orient_hist = np.histogram(np.degrees(theta[edges]) % 180, bins=18, range=(0, 180), weights=m)[0]

    # texture
    patches32 = (_blocks(gray, 32) * 31.99).astype(int)
    local_entropy = np.mean([_entropy(np.bincount(p.ravel(), minlength=32).astype(float)) for p in patches32])
    flat_ratio = (_blocks(gray, 16).std(axis=(1, 2)) < 0.02).mean()

    # power spectrum slope on a Hann-windowed image
    win = np.outer(np.hanning(h), np.hanning(w))
    power = np.abs(np.fft.fft2((gray - gray.mean()) * win)) ** 2
    r = np.hypot(*np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h)))
    bins = np.logspace(np.log10(2 / min(h, w)), np.log10(0.5), 20)
    idx = np.digitize(r.ravel(), bins)
    radial = np.array([power.ravel()[idx == i].mean() for i in range(1, len(bins))])
    centres = np.sqrt(bins[:-1] * bins[1:])
    ok = radial > 0
    spectral_slope = -np.polyfit(np.log(centres[ok]), np.log(radial[ok]), 1)[0] if ok.sum() > 3 else np.nan

    # box-counting dimension of the edge map
    sizes, boxes = [], []
    for s in (2, 4, 8, 16, 32, 64):
        n = _blocks(edges.astype(float), s).max(axis=(1, 2)).sum()
        if n > 0:
            sizes.append(s)
            boxes.append(n)
    fractal = -np.polyfit(np.log(sizes), np.log(boxes), 1)[0] if len(sizes) >= 3 else np.nan

    # composition
    small = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((64, max(8, round(64 * h / w)))),
                       dtype=float) / 255
    spread = np.abs(small - small.mean()).mean() + 1e-6
    symmetry = float(np.clip(1 - np.abs(small - small[:, ::-1]).mean() / (2 * spread), 0, 1))
    ch, cw = h // 3, w // 3
    centre_focus = edges[ch:2 * ch, cw:2 * cw].mean() / (edges.mean() + 1e-6)

    periodicity = _periodicity(gray)

    return {
        "lightness": L.mean(), "lightness_contrast": L.std(), "chroma": chroma.mean(),
        "colourfulness": colourfulness, "chromatic_share": coloured.mean(),
        "warm_ratio": warm.mean() if coloured.any() else np.nan,
        "hue_diversity": _entropy(hue_hist.astype(float)) / np.log2(12) if coloured.sum() > 50 else 0.0,
        "palette_size": palette_size,
        "edge_density": edges.mean(), "local_entropy": local_entropy, "flat_ratio": flat_ratio,
        "spectral_slope": spectral_slope, "fractal_dimension": fractal,
        "rectilinearity": (e_v + e_h) / e_all, "horizontal_bias": e_h / (e_h + e_v + 1e-9),
        "orientation_entropy": _entropy(orient_hist) / np.log2(18) if orient_hist.sum() > 0 else 0.0,
        "symmetry_lr": symmetry, "centre_focus": centre_focus, "periodicity": periodicity,
        "aspect_ratio": w / h,
    }


def compute_all(paths: list[Path]) -> pd.DataFrame:
    rows = [compute(load(p)) for p in tqdm(paths, desc="descriptors")]
    return pd.DataFrame(rows, columns=NAMES)
