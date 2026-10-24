"""Formal-analysis vocabulary scored with SigLIP 2's text tower (zero-shot, local).

SigLIP 2's pooled image embedding (the first half of the `global` descriptor) lives in the
same space as its text embeddings, so every painting can be scored against short formal
descriptions without any external API. Terms are grouped in four facets (form, colour,
texture, composition), the dimensions used by formal analysis and by Abstract4D.
"""

import numpy as np

from . import OUTPUT_DIR

MODEL_NAME = "google/siglip2-base-patch16-naflex"
TEMPLATE = "an abstract painting with {}"
TEXT_CACHE = OUTPUT_DIR / "embeddings" / "siglip2_text.npz"
IMAGE_DIM = 768  # the `global` descriptor is [pooled image embedding (768) | mean patch token (768)]

FACETS = {
    "form": [
        "hard-edged geometric shapes", "circles and concentric rings", "a grid of squares", "a pattern of dots",
        "diagonal lines", "biomorphic organic shapes", "fragmented cubist planes", "one large simple shape",
        "thin straight lines", "calligraphic signs", "scribbles and drawn marks", "figures", "a landscape horizon",
    ],
    "colour": [
        "a monochrome surface", "black and white", "vivid saturated colours", "primary colours",
        "earthy brown and ochre tones", "cool blue tones", "warm red and orange tones", "pastel colours",
        "dark sombre colours", "soft luminous fields of colour", "gold and metallic colours",
    ],
    "texture": [
        "thick impasto paint", "dripped and splattered paint", "loose gestural brushstrokes", "flat even paint",
        "a rough material surface", "a slashed or punctured canvas", "smooth gradients", "slabs of paint applied with a knife",
        "fine detailed linework", "collage of paper and materials", "watercolour washes",
    ],
    "composition": [
        "an all-over composition", "a centred composition", "horizontal bands", "vertical stripes",
        "a symmetrical composition", "a dense crowded composition", "a sparse minimal composition",
        "a repetitive optical pattern", "a dynamic swirling composition", "a lot of empty space",
    ],
}
TERMS = [(facet, term) for facet, terms in FACETS.items() for term in terms]

# Only used to check that the text tower is grounded on this corpus (see scripts/explain_clusters.py)
MOVEMENT_PROMPTS = {
    "geometric": "a geometric abstract painting",
    "action_painting": "an abstract expressionist action painting",
    "tachisme_informel": "an art informel painting with heavy matter",
    "surrealisme": "a surrealist painting",
    "futurisme_orphisme": "a futurist or orphist painting",
    "colour_field": "a colour field painting",
    "nouveau_realisme": "a nouveau realisme artwork",
    "street_art": "graffiti street art",
    "art_brut": "an outsider art painting",
    "expressionnisme": "an expressionist painting",
    "abstraction_lyrique": "a lyrical abstraction painting",
    "arte_povera_conceptuel": "a conceptual art or arte povera work",
    "kinetic_op": "an op art painting",
    "non_occidental": "a non-western abstract artwork",
}


def encode_texts(prompts: list[str], device: str = "cpu") -> np.ndarray:
    """L2-normalised SigLIP 2 text embeddings, cached on disk by prompt."""
    cache = dict(np.load(TEXT_CACHE)) if TEXT_CACHE.exists() else {}
    missing = [p for p in prompts if p not in cache]
    if missing:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
        # SigLIP was trained on lower-cased text padded to 64 tokens
        inputs = tokenizer([p.lower() for p in missing], padding="max_length", max_length=64,
                           truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        out = out if isinstance(out, torch.Tensor) else out.pooler_output
        emb = torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()
        cache.update(zip(missing, emb))
        TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(TEXT_CACHE, **cache)
    return np.stack([cache[p] for p in prompts])


def image_text_similarity(global_descriptor: np.ndarray, prompts: list[str]) -> np.ndarray:
    """Cosine similarity (n_images, n_prompts) between paintings and prompts."""
    img = global_descriptor[:, :IMAGE_DIM]
    img = img / np.linalg.norm(img, axis=1, keepdims=True)
    return img @ encode_texts(prompts).T


def term_prompts() -> list[str]:
    return [TEMPLATE.format(term) for _, term in TERMS]
