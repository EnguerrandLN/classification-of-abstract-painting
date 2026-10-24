"""Conformal sets of plausible groups for every painting, adapted from two 2026 papers.

Nath, Hur & Allen, "Inference for Clustering: Conformal Sets for Cluster Labels" (arXiv 2604.03488)
and "Weighted Conformal Clustering" (arXiv 2606.00436).

Their recipe (split conformal clustering with stochastic labels):
  fit a SOFT clustering; sample each label from the soft memberships instead of taking the argmax
  (hard labels are neither consistent nor stable and under-cover); train a classifier pi_hat on the
  sampled labels of a training split; on a calibration split, sample labels again and compute the
  generalised inverse-quantile score of Romano, Sesia & Candes (2020),
      s(x, y) = sum of pi_hat(x)_k over the labels k ranked strictly above y   (top label: 0);
  return C(x) = {y : s(x, y) <= q}, q the ceil((1 - alpha)(n_cal + 1))-th smallest calibration score.
  Guarantee: P(latent label in C(X)) >= 1 - alpha - (consistency and stability terms of the soft
  clustering). The weighted version reweights calibration point i by p_hat(Y_i|X_i) / A(X_i)_{Y_i}
  and the test label y by p_hat(y|x) / A(x)_y (A = soft clustering, p_hat = pi_hat), and uses the
  weighted quantile with the test mass at +infinity, which corrects the gap between the synthetic
  labels and the true conditional label distribution.

Adaptations for this project:
1. Anchored soft clustering. EVoC is hard, transductive and has noise. Its finest groups anchor a
   Gaussian mixture with FIXED means (centroids of each group's core paintings) and one shared
   spherical variance and mixing weights estimated by EM, in a d-dimensional PCA space:
   A(x)_c = softmax_c(log w_c - |x - mu_c|^2 / (2 sigma^2)). Every fit shares the anchors, so the
   papers' label-alignment (Hungarian) step is the identity.
2. A set for every painting, by cross-fitting: each painting gets its set from a classifier and a
   calibration split it was not part of (the papers only return sets for held-out points). EVoC itself
   saw every painting, which matches the "augmented" clustering of the weighted paper, where the query
   point is included in the clustering fit.
3. Hierarchy (not in the papers). Labels are sampled at the finest layer and propagated through the
   tree, so sampled labels are nested; a coarser group's probability is the sum of its children's.
   Each of the L layers gets its own threshold at level alpha / L, then sets are restricted top-down
   (a group is kept only if its parent is kept). Claim: P(every restricted set covers its label) >=
   1 - alpha. Proof: by the union bound, with probability >= 1 - alpha every unrestricted set C_l
   covers its label y_l. On that event, the coarsest restricted set equals C_L and covers y_L; if the
   restricted set at layer l+1 covers y_{l+1} = parent(y_l), then y_l, which is in C_l and whose parent
   is kept, survives the restriction. By induction every layer is covered. Restricted sets are also
   coherent: the parent of every kept group is kept.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression


@dataclass
class AnchoredMixture:
    """Gaussian mixture with fixed means, shared spherical variance and weights fitted by EM."""
    means: np.ndarray
    log_weights: np.ndarray
    sigma2: float

    @classmethod
    def fit(cls, X: np.ndarray, means: np.ndarray, n_iter: int = 200) -> "AnchoredMixture":
        K, d = means.shape
        D2 = _sq_dist(X, means)
        log_w, sigma2 = np.full(K, -np.log(K)), D2.min(1).mean() / d
        for _ in range(n_iter):
            R = softmax(log_w - D2 / (2 * sigma2), axis=1)
            log_w = np.log(R.mean(0) + 1e-12)
            sigma2 = (R * D2).sum() / (len(X) * d)
        return cls(means, log_w, sigma2)

    def memberships(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.log_weights - _sq_dist(X, self.means) / (2 * self.sigma2), axis=1)

    def sample(self, n: int, rng: np.random.Generator, df: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Synthetic data for the coverage simulations: (X, latent labels).

        df=None draws Gaussian noise (the mixture is then exactly right); a finite df draws Student-t
        noise with the same variance, heavy-tailed, so that the mixture is misspecified.
        """
        labels = rng.choice(len(self.means), size=n, p=np.exp(self.log_weights) / np.exp(self.log_weights).sum())
        shape = (n, self.means.shape[1])
        noise = rng.normal(size=shape) if df is None else rng.standard_t(df, size=shape) / np.sqrt(df / (df - 2))
        return self.means[labels] + np.sqrt(self.sigma2) * noise, labels


def _sq_dist(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (X ** 2).sum(1)[:, None] - 2 * X @ M.T + (M ** 2).sum(1)[None, :]


def sample_labels(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One categorical draw per row of P."""
    u = rng.random(len(P))[:, None]
    return np.minimum((P.cumsum(1) < u).sum(1), P.shape[1] - 1)


def fit_classifier(X: np.ndarray, soft_targets: np.ndarray, Cs=(0.1, 1.0, 10.0), seed: int = 0) -> LogisticRegression:
    """Multinomial logistic regression trained on the expected loss of the sampled labels.

    Every point is repeated once per label with weight = its soft membership (same expected objective
    as training on sampled labels, without the sampling noise). C is chosen by expected log-loss on a
    20% split of the training points.
    """
    K = soft_targets.shape[1]
    rng = np.random.default_rng(seed)
    val = rng.random(len(X)) < 0.2

    def fit(rows, C):
        Xr = np.repeat(X[rows], K, axis=0)
        yr = np.tile(np.arange(K), rows.sum())
        wr = soft_targets[rows].ravel()
        keep = wr > 1e-6
        return LogisticRegression(C=C, max_iter=2000).fit(Xr[keep], yr[keep], sample_weight=wr[keep])

    def loss(model):
        P = full_proba(model, X[val], K)
        return -(soft_targets[val] * np.log(P + 1e-12)).sum(1).mean()

    best = min(Cs, key=lambda C: loss(fit(~val, C)))
    return fit(np.ones(len(X), bool), best)


def full_proba(model: LogisticRegression, X: np.ndarray, K: int) -> np.ndarray:
    """Class probabilities over all K labels (labels absent from training get probability 0)."""
    P = np.zeros((len(X), K))
    P[:, model.classes_] = model.predict_proba(X)
    return P


def aps_scores(P: np.ndarray) -> np.ndarray:
    """Generalised inverse-quantile score s(x, y) for every label: mass of the labels ranked above y."""
    order = np.argsort(-P, axis=1, kind="stable")
    sorted_p = np.take_along_axis(P, order, axis=1)
    before = np.cumsum(sorted_p, axis=1) - sorted_p
    S = np.empty_like(P)
    np.put_along_axis(S, order, before, axis=1)
    return S


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    n = len(cal_scores)
    rank = int(np.ceil((1 - alpha) * (n + 1)))
    return np.inf if rank > n else float(np.sort(cal_scores)[rank - 1])


def weighted_thresholds(cal_scores: np.ndarray, cal_weights: np.ndarray, test_weights: np.ndarray,
                        alpha: float) -> np.ndarray:
    """Weighted conformal quantile, test mass at +infinity, for every (test point, label) pair."""
    order = np.argsort(cal_scores)
    s, w = cal_scores[order], cal_weights[order]
    cum = np.cumsum(w)
    needed = (1 - alpha) * (w.sum() + test_weights)
    idx = np.searchsorted(cum, needed - 1e-12)
    return np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)


def nested_matrices(parents: list[np.ndarray], sizes: list[int]) -> list[np.ndarray]:
    """Indicator matrices (K_l, K_{l+1}) that map a layer's probabilities to the next coarser layer."""
    mats = []
    for l, par in enumerate(parents):
        M = np.zeros((sizes[l], sizes[l + 1]))
        M[np.arange(sizes[l]), par] = 1
        mats.append(M)
    return mats


def layer_probs(P0: np.ndarray, mats: list[np.ndarray]) -> list[np.ndarray]:
    out = [P0]
    for M in mats:
        out.append(out[-1] @ M)
    return out


def restrict_top_down(sets: list[np.ndarray], parents: list[np.ndarray]) -> list[np.ndarray]:
    """Keep a group only if its parent is kept, from the coarsest layer down. sets: boolean (n, K_l)."""
    out = [None] * len(sets)
    out[-1] = sets[-1].copy()
    for l in range(len(sets) - 2, -1, -1):
        out[l] = sets[l] & out[l + 1][:, parents[l]]
    return out


def cross_fitted_sets(X: np.ndarray, anchors: np.ndarray, parents: list[np.ndarray], sizes: list[int],
                      alpha: float, method: str = "split", score_model: str = "mixture", n_folds: int = 5,
                      seed: int = 0) -> dict:
    """Conformal sets for every point, each from a score model and calibration split it was not part of.

    method: "weighted" (weighted conformal clustering), "split" (stochastic-label split conformal),
    "hard" (naive: argmax labels, the baseline the papers show to under-cover) or "cutoff" (smallest
    top-k soft memberships reaching 1 - alpha, uncalibrated).
    score_model: "classifier" (logistic regression trained on the training split's soft labels, the
    papers' Algorithm 1) or "mixture" (the anchored mixture fitted on the training split, which the
    first paper's Remark 1 allows when the soft clustering can score new points).
    Returns per-layer boolean sets: "marginal" (each layer at level alpha) and "hierarchical"
    (alpha / L per layer, restricted top-down), plus the soft memberships used.
    """
    rng = np.random.default_rng(seed)
    n, L = len(X), len(sizes)
    mats = nested_matrices(parents, sizes)
    folds = rng.permutation(n) % n_folds
    marginal = [np.zeros((n, k), bool) for k in sizes]
    simultaneous = [np.zeros((n, k), bool) for k in sizes]
    memberships = np.zeros((n, sizes[0]))

    for f in range(n_folds):
        test = folds == f
        rest = np.flatnonzero(~test)
        rng.shuffle(rest)
        tr, ca = np.zeros(n, bool), np.zeros(n, bool)
        tr[rest[: len(rest) // 2]], ca[rest[len(rest) // 2:]] = True, True

        mix_tr = AnchoredMixture.fit(X[tr], anchors)
        mix_ca = AnchoredMixture.fit(X[ca], anchors)
        A_ca, A_te = mix_ca.memberships(X[ca]), mix_ca.memberships(X[test])
        memberships[test] = A_te

        if method == "cutoff":
            for l, P in enumerate(layer_probs(A_te, mats)):
                for sets, a in ((marginal, alpha), (simultaneous, alpha / L)):
                    sets[l][test] = aps_scores(P) < 1 - a
            continue

        if score_model == "mixture" and method != "hard":
            P_ca = layer_probs(mix_tr.memberships(X[ca]), mats)
            P_te = layer_probs(mix_tr.memberships(X[test]), mats)
        else:
            targets = mix_tr.memberships(X[tr])
            if method == "hard":
                targets = np.eye(sizes[0])[targets.argmax(1)]
            clf = fit_classifier(X[tr], targets, seed=seed + f)
            P_ca = layer_probs(full_proba(clf, X[ca], sizes[0]), mats)
            P_te = layer_probs(full_proba(clf, X[test], sizes[0]), mats)
        y0 = A_ca.argmax(1) if method == "hard" else sample_labels(A_ca, rng)
        A_ca_l, A_te_l = layer_probs(A_ca, mats), layer_probs(A_te, mats)
        y = [y0]
        for par in parents:
            y.append(par[y[-1]])

        unrestricted = []
        for l in range(L):
            S_ca = aps_scores(P_ca[l])[np.arange(ca.sum()), y[l]]
            S_te = aps_scores(P_te[l])
            for sets, a in ((marginal, alpha), (unrestricted, alpha / L)):
                if method == "weighted":
                    w_ca = P_ca[l][np.arange(ca.sum()), y[l]] / np.maximum(A_ca_l[l][np.arange(ca.sum()), y[l]], 1e-12)
                    w_te = P_te[l] / np.maximum(A_te_l[l], 1e-12)
                    t = weighted_thresholds(S_ca, w_ca, w_te, a)
                else:
                    t = conformal_quantile(S_ca, a)
                if sets is marginal:
                    sets[l][test] = S_te <= t
                else:
                    unrestricted.append(S_te <= t)
        for l, s in enumerate(restrict_top_down(unrestricted, parents)):
            simultaneous[l][test] = s

    if method == "cutoff":
        simultaneous = restrict_top_down(simultaneous, parents)
    return {"marginal": marginal, "hierarchical": simultaneous, "memberships": memberships}
