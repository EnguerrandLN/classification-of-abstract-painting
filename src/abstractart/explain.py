"""Cluster explanations: what distinguishes a cluster from a reference set of paintings.

Every signal (an interpretable descriptor or a vocabulary term) is z-scored over the
reference set, then averaged over the cluster's core members: +1.0 means "one standard
deviation above the reference". Families (coarsest layer) are contrasted with the whole
corpus and sub-groups with their parent family, so each card says what is specific at its
own level of the hierarchy.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .vocabulary import FACETS, TERMS

MIN_TERM_Z = 0.3
MIN_DESCRIPTOR_Z = 0.4


def contrast(values: np.ndarray, members: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Mean z-score of `members` for every column, z-scored over `reference` rows."""
    ref = values[reference]
    mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0) + 1e-9
    return np.nanmean((values[members] - mu) / sd, axis=0)


@dataclass
class Card:
    layer: int
    cluster: int
    parent: int | None
    size: int
    core: int
    terms: dict[str, list[tuple[str, float]]] = field(default_factory=dict)  # facet -> [(term, z)]
    descriptors: list[tuple[str, float]] = field(default_factory=list)       # strongest |z| first
    artists: list[tuple[str, int]] = field(default_factory=list)
    movement: tuple[str, float] | None = None
    exemplars: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        """The two most distinctive terms, from different facets."""
        best = sorted(((z, t) for facet in self.terms.values() for t, z in facet[:1]), reverse=True)[:2]
        return " · ".join(t for _, t in best) or "no distinctive term"


def make_card(layer: int, cluster: int, parent: int | None, members: np.ndarray, core: np.ndarray,
              reference: np.ndarray, term_sim: np.ndarray, descriptors: pd.DataFrame,
              manifest: pd.DataFrame, movements: dict[str, str], embedding: np.ndarray,
              excluded_terms: frozenset = frozenset(), n_terms: int = 2, n_descriptors: int = 5,
              n_exemplars: int = 5) -> Card:
    """`excluded_terms`: vocabulary terms that failed the grounding check, never shown."""
    profile_rows = members & core if (members & core).sum() >= 3 else members

    tz = contrast(term_sim, profile_rows, reference)
    terms = {}
    for facet in FACETS:
        idx = [i for i, (f, t) in enumerate(TERMS) if f == facet and t not in excluded_terms]
        ranked = sorted(((tz[i], TERMS[i][1]) for i in idx), reverse=True)
        terms[facet] = [(t, float(z)) for z, t in ranked[:n_terms] if z >= MIN_TERM_Z]

    dz = contrast(descriptors.to_numpy(dtype=float), profile_rows, reference)
    order = np.argsort(-np.abs(dz))
    desc = [(descriptors.columns[i], float(dz[i])) for i in order[:n_descriptors] if abs(dz[i]) >= MIN_DESCRIPTOR_Z]

    in_cluster = manifest[members]
    mv = in_cluster["artist"].map(movements).dropna().value_counts()
    centroid = embedding[profile_rows].mean(0)
    rows = np.flatnonzero(profile_rows)
    central = rows[np.argsort(-(embedding[rows] @ centroid))][:n_exemplars]
    return Card(
        layer=layer, cluster=cluster, parent=parent, size=int(members.sum()), core=int((members & core).sum()),
        terms=terms, descriptors=desc,
        artists=[(a, int(n)) for a, n in in_cluster["artist"].value_counts().head(5).items()],
        movement=(mv.index[0], float(mv.iloc[0] / mv.sum())) if len(mv) else None,
        exemplars=manifest["filename"].to_numpy()[central].tolist(),
    )
