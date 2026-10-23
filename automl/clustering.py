"""Clustering with a required number of groups and no labels.

1. Prune features the partition does not separate (noise columns).
2. Build a pool of partitions: k-means, Ward, and Gaussian mixtures (diagonal
   and full covariance, warm-started from k-means and best-of-several inits).
3. Return the consensus partition: the one that agrees most (mean ARI) with
   the rest of the pool, silhouette as a tie-break.

On 19 labelled benchmark problems this consensus rule averaged ARI 0.874 against
0.836 for picking by silhouette alone (see README)."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

FIT_ROWS = 20000   # mixtures are fitted on at most this many rows, then predict every row
WARD_ROWS = 8000   # Ward needs O(n^2) memory


def _kmeans(X, k, seed=0):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def _warm_gmm(X, k, cov, seed=0):
    init = _kmeans(X, k, seed)
    means = np.vstack([X[init == j].mean(0) if np.any(init == j) else X[j % len(X)] for j in range(k)])
    return GaussianMixture(n_components=k, covariance_type=cov, means_init=means, reg_covar=1e-4,
                           max_iter=300, random_state=seed).fit(X)


def _best_gmm(X, k, cov):
    best = None
    for init, seed in (("kmeans", 0), ("k-means++", 1), ("random_from_data", 2), ("random_from_data", 3),
                       ("k-means++", 4)):
        g = GaussianMixture(n_components=k, covariance_type=cov, init_params=init, reg_covar=1e-4,
                            max_iter=300, random_state=seed).fit(X)
        if best is None or g.lower_bound_ > best.lower_bound_:
            best = g
    return best


def _silhouette(X, labels, seed=0):
    if len(np.unique(labels)) < 2:
        return -1.0
    return float(silhouette_score(X, labels, sample_size=min(len(X), 3000), random_state=seed))


def separation_per_feature(X, labels):
    """Share of each feature's variance explained by the partition (eta squared)."""
    total = X.var(0) + 1e-12
    between = np.zeros(X.shape[1])
    for j in np.unique(labels):
        m = labels == j
        between += m.mean() * (X[m].mean(0) - X.mean(0)) ** 2
    return between / total


def prune_features(X, k):
    """Keep features that a k-means partition separates; repeat once in the reduced space."""
    eta = separation_per_feature(X, _kmeans(X, k))
    keep = eta >= max(0.02, 0.1 * eta.max())
    if not (1 <= keep.sum() < X.shape[1]):
        return np.ones(X.shape[1], bool)
    eta2 = separation_per_feature(X, _kmeans(X[:, keep], k))
    keep2 = eta2 >= max(0.02, 0.1 * eta2.max())
    return keep2 if keep2.sum() >= 1 else keep


def choose_k(X, budget, k_max=10):
    best_k, best_s = 2, -np.inf
    for k in range(2, min(k_max, len(X) - 1) + 1):
        if budget.left() < 8:
            break
        s = _silhouette(X, _kmeans(X, k))
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def run_clustering(X, k, budget, log=print):
    report = {}
    if k is None:
        k = choose_k(X, budget)
        report["n_clusters_chosen_by_silhouette"] = k
    k = int(max(1, min(k, len(X))))
    if k == 1 or X.shape[1] == 0:
        return np.zeros(len(X), dtype=int), {"n_clusters": k, "selected_model": "trivial"}

    keep = prune_features(X, k)
    Xs = X[:, keep]
    report["features_used"] = int(keep.sum())
    report["features_pruned"] = int((~keep).sum())
    if Xs.shape[1] > 20:
        Xs = PCA(n_components=20, random_state=0).fit_transform(Xs)
        report["pca_components"] = 20

    rng = np.random.RandomState(0)
    fit_idx = np.sort(rng.choice(len(Xs), FIT_ROWS, replace=False)) if len(Xs) > FIT_ROWS else slice(None)
    Xf = Xs[fit_idx]

    parts = {"kmeans": _kmeans(Xs, k)}
    jobs = [("gmm_diag_warm", lambda: _warm_gmm(Xf, k, "diag").predict(Xs)),
            ("gmm_full_warm", lambda: _warm_gmm(Xf, k, "full").predict(Xs)),
            ("ward", lambda: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)),
            ("gmm_full_best", lambda: _best_gmm(Xf, k, "full").predict(Xs)),
            ("gmm_diag_best", lambda: _best_gmm(Xf, k, "diag").predict(Xs))]
    for name, fn in jobs:
        if budget.left() < 3.0:
            break
        if name == "ward" and len(Xs) > WARD_ROWS:
            continue
        if "full" in name and len(Xf) < 3 * k * (Xs.shape[1] + 1):
            continue  # too few rows to estimate full covariances
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lab = fn()
            if len(np.unique(lab)) == k:
                parts[name] = lab
        except Exception as exc:
            log(f"  {name} failed: {exc}")

    names = list(parts)
    sil = {m: _silhouette(Xs, parts[m]) for m in names}
    agree = {m: float(np.mean([adjusted_rand_score(parts[m], parts[o]) for o in names if o != m]))
             if len(names) > 1 else 1.0 for m in names}
    best = max(names, key=lambda m: agree[m] + 0.1 * sil[m])
    for m in names:
        log(f"  {m:<14} silhouette={sil[m]:.4f}  agreement={agree[m]:.4f}")
    report["candidates"] = {m: {"silhouette": round(sil[m], 6), "mean_agreement_ari": round(agree[m], 6)} for m in names}
    report["selected_model"] = best
    report["selection_rule"] = "max mean ARI agreement with the other candidates (+0.1 x silhouette)"
    report["silhouette"] = round(sil[best], 6)
    report["n_clusters"] = k

    labels = parts[best]
    # stable label numbering: clusters ordered by size, largest = 0
    order = np.argsort(-np.bincount(labels, minlength=k))
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[labels], report
