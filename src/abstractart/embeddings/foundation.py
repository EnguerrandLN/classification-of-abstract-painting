"""Embeddings from frozen vision foundation models (DINOv2, SigLIP 2).

Each painting is encoded at its native aspect ratio (no padding, no square resize)
and yields three descriptors:

- global: image-level embedding (CLS / attention-pooled token + mean patch token);
- stats:  per-channel mean and std of the patch tokens of several intermediate
          layers (AdaIN-style first/second-order style statistics);
- gram:   Gram matrix of the patch tokens of the same layers, after projecting the
          channels on a PCA basis fitted on the corpus. This is the v1 VGG16 Gram
          signature, moved to a transformer backbone.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

STYLE_LAYERS = (3, 6, 9, 12)
GRAM_RANK = 48
MAX_PATCHES = 1024
PCA_FIT_IMAGES = 256
PCA_TOKENS_PER_IMAGE = 64

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img.draft("RGB", (1024, 1024))  # fast JPEG decode at reduced scale, still >= 2x the model input
    return img.convert("RGB")


def native_grid(width: int, height: int, patch: int, max_patches: int) -> tuple[int, int]:
    """Largest (rows, cols) patch grid with the image aspect ratio and at most `max_patches` cells."""
    scale = np.sqrt(max_patches / ((width / patch) * (height / patch)))
    cols = max(4, int(width / patch * scale))
    rows = max(4, int(height / patch * scale))
    while rows * cols > max_patches:
        if cols >= rows:
            cols -= 1
        else:
            rows -= 1
    return rows, cols


@dataclass
class Encoded:
    global_vec: torch.Tensor        # (D_global,)
    layer_tokens: list[torch.Tensor]  # one (N_valid, D) tensor per style layer


class Dinov2Encoder:
    patch = 14

    def __init__(self, name: str = "facebook/dinov2-base", device: str = "cuda"):
        from transformers import AutoModel
        self.name, self.device = name, device
        self.model = AutoModel.from_pretrained(name).to(device).eval()

    def encode(self, img: Image.Image) -> Encoded:
        rows, cols = native_grid(*img.size, self.patch, MAX_PATCHES)
        img = img.resize((cols * self.patch, rows * self.patch), Image.BICUBIC)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = self.model(pixel_values=x, output_hidden_states=True)
        last = out.last_hidden_state[0].float()
        global_vec = torch.cat([last[0], last[1:].mean(0)])  # CLS + mean patch token
        tokens = [out.hidden_states[l][0, 1:].float() for l in STYLE_LAYERS]
        return Encoded(global_vec, tokens)


class Siglip2Encoder:
    def __init__(self, name: str = "google/siglip2-base-patch16-naflex", device: str = "cuda"):
        from transformers import AutoImageProcessor, Siglip2VisionModel
        self.name, self.device = name, device
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = Siglip2VisionModel.from_pretrained(name).to(device).eval()

    def encode(self, img: Image.Image) -> Encoded:
        inputs = self.processor(images=img, max_num_patches=MAX_PATCHES, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = self.model(**inputs, output_hidden_states=True)
        valid = inputs["pixel_attention_mask"][0].bool()
        last = out.last_hidden_state[0, valid].float()
        global_vec = torch.cat([out.pooler_output[0].float(), last.mean(0)])  # MAP-pooled + mean patch token
        tokens = [out.hidden_states[l][0, valid].float() for l in STYLE_LAYERS]
        return Encoded(global_vec, tokens)


ENCODERS = {"dinov2": Dinov2Encoder, "siglip2": Siglip2Encoder}


def _fit_token_bases(encoder, paths: list[Path], seed: int = 0) -> list[dict]:
    """Per style layer: channel mean/std and a PCA basis of standardised patch tokens."""
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(paths), size=min(PCA_FIT_IMAGES, len(paths)), replace=False)
    samples = [[] for _ in STYLE_LAYERS]
    for i in tqdm(subset, desc=f"{encoder.name}: fitting token bases"):
        enc = encoder.encode(load_image(paths[i]))
        for s, tok in zip(samples, enc.layer_tokens):
            pick = torch.from_numpy(rng.choice(len(tok), size=min(PCA_TOKENS_PER_IMAGE, len(tok)), replace=False))
            s.append(tok[pick.to(tok.device)])
    bases = []
    for s in samples:
        T = torch.cat(s)
        mu, sd = T.mean(0), T.std(0) + 1e-6
        _, _, V = torch.pca_lowrank((T - mu) / sd, q=GRAM_RANK, center=False)
        bases.append({"mu": mu, "sd": sd, "P": V[:, :GRAM_RANK]})
    return bases


def extract(model: str, paths: list[Path], device: str = "cuda") -> dict[str, np.ndarray]:
    encoder = ENCODERS[model](device=device)
    bases = _fit_token_bases(encoder, paths)
    iu = torch.triu_indices(GRAM_RANK, GRAM_RANK)

    out = {"global": [], "stats": [], "gram": []}
    for path in tqdm(paths, desc=f"{encoder.name}: encoding"):
        enc = encoder.encode(load_image(path))
        stats, grams = [], []
        for tok, b in zip(enc.layer_tokens, bases):
            stats += [tok.mean(0), tok.std(0)]
            Y = ((tok - b["mu"]) / b["sd"]) @ b["P"]
            grams.append((Y.T @ Y / len(Y))[iu[0], iu[1]])
        out["global"].append(enc.global_vec.cpu().numpy())
        out["stats"].append(torch.cat(stats).cpu().numpy())
        out["gram"].append(torch.cat(grams).cpu().numpy())
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}
