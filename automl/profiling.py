"""Structured data profile written next to the predictions (profile.json)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance

from .prep import parse_datetime


def _f(x):
    """JSON-safe float."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, 6)


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return _f(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v if isinstance(v, (int, str, bool)) or v is None else str(v)


def encode_simple(X: pd.DataFrame, info: dict):
    """Quick numeric encoding per raw column (used for relevance and correlations)."""
    cols, discrete = {}, []
    for c, e in info.items():
        if e["drop"] is not None or c not in X.columns:
            continue
        s = X[c]
        if e["type"] == "numeric":
            v = pd.to_numeric(s, errors="coerce").astype(float)
            cols[c] = v.fillna(v.median() if v.notna().any() else 0.0).values
            discrete.append(False)
        elif e["type"] == "datetime":
            d = parse_datetime(s)
            v = (d - d.min()).dt.total_seconds()
            cols[c] = v.fillna(v.median() if v.notna().any() else 0.0).values
            discrete.append(False)
        elif e["type"] == "text":
            cols[c] = s.astype(str).str.len().fillna(0).astype(float).values
            discrete.append(False)
        else:
            cols[c] = pd.Series(pd.factorize(s.astype(str).where(s.notna(), "__nan"))[0]).astype(float).values
            discrete.append(True)
    return pd.DataFrame(cols, index=X.index), np.array(discrete, dtype=bool)


def feature_relevance(X: pd.DataFrame, y: np.ndarray, info: dict, task: str, max_rows=4000, seed=0):
    """Per-feature relevance to the target.

    The decision uses held-out permutation importance of a quick gradient-boosting
    model (it sees interactions and is far less noisy than k-NN mutual information);
    mutual information is reported alongside for reference."""
    E, discrete = encode_simple(X, info)
    if E.shape[1] == 0:
        return {}
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(E), size=min(max_rows, len(E)), replace=False)
    A, t = E.values[idx], y[idx]
    fn = mutual_info_classif if task == "classification" else mutual_info_regression
    d = E.shape[1]
    mi_rows = 2500 if d <= 50 else 1000          # cost grows with rows x features
    repeats = 5 if d <= 50 else 3 if d <= 150 else 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = fn(A[:mi_rows], t[:mi_rows], discrete_features=discrete, random_state=seed)

        cut = int(len(A) * 0.7)
        cat = discrete & (E.nunique().values <= 250)
        params = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=seed,
                      categorical_features=cat if cat.any() else None, early_stopping=False)
        if task == "classification":
            model = HistGradientBoostingClassifier(**params)
            scoring = "neg_log_loss" if len(np.unique(t[:cut])) == len(np.unique(t)) else "accuracy"
        else:
            model = HistGradientBoostingRegressor(**params)
            scoring = "r2"
        imp_mean = imp_std = None
        base = 1.0
        try:
            model.fit(A[:cut], t[:cut])
            pi = permutation_importance(model, A[cut:], t[cut:], scoring=scoring, n_repeats=repeats,
                                        random_state=seed, n_jobs=1)
            imp_mean, imp_std = pi.importances_mean, pi.importances_std
            base = abs(model.score(A[cut:], t[cut:])) if scoring == "r2" else 1.0
        except Exception:
            pass
    out = {}
    for j, c in enumerate(E.columns):
        entry = {"mutual_information": _f(mi[j])}
        if imp_mean is not None:
            m, s = imp_mean[j], imp_std[j]
            entry["permutation_importance"] = _f(m)
            entry["relevant"] = bool(m > max(2 * s, 0.002 * max(base, 0.05)))
        else:
            entry["relevant"] = bool(mi[j] > 0.02)
        out[c] = entry
    return out


def build_profile(name, task, target, id_column, n_clusters, train, test, X, y, info, metric, relevance):
    rows = len(train)
    cols_out = []
    numeric, categorical, datetime_cols, text_cols = [], [], [], []
    missing_cols, constant, duplicates, ids, near_constant = [], [], {}, [], []
    stds = {}
    for c, e in info.items():
        s = X[c]
        n_missing = int(s.isna().sum())
        n_missing_test = int(test[c].isna().sum()) if test is not None and c in test.columns else None
        nun = int(s.nunique(dropna=True))
        col = {
            "name": c,
            "inferred_type": e["type"],
            "declared_type": e["declared_type"],
            "dtype": str(train[c].dtype) if c in train.columns else str(s.dtype),
            "n_missing": n_missing,
            "missing_fraction": _f(n_missing / max(rows, 1)),
            "n_missing_test": n_missing_test,
            "n_unique": nun,
            "unique_fraction": _f(nun / max(rows - n_missing, 1)),
            "is_constant": e["drop"] == "constant",
            "is_duplicate": e["drop"] == "duplicate",
            "duplicate_of": e["duplicate_of"],
            "is_identifier": e["drop"] == "identifier",
            "used_as_feature": e["drop"] is None,
            "drop_reason": e["drop"],
        }
        top_share = s.value_counts(normalize=True, dropna=False).iloc[0] if rows else 0
        col["is_near_constant"] = bool(top_share >= 0.98 and e["drop"] is None)
        if e["type"] == "numeric":
            v = pd.to_numeric(s, errors="coerce")
            col.update({
                "mean": _f(v.mean()), "std": _f(v.std()), "min": _f(v.min()), "max": _f(v.max()),
                "median": _f(v.median()), "skewness": _f(v.skew()),
                "n_outliers_iqr": int(((v < v.quantile(.25) - 3 * (v.quantile(.75) - v.quantile(.25))) |
                                       (v > v.quantile(.75) + 3 * (v.quantile(.75) - v.quantile(.25)))).sum()),
                "is_integer": bool(v.dropna().size and np.allclose(v.dropna(), np.round(v.dropna()))),
            })
            if e["drop"] is None and v.std() > 0:
                stds[c] = float(v.std())
        elif e["type"] in ("categorical", "text"):
            vc = s.astype(str).where(s.notna()).value_counts()
            col["top_values"] = {str(k): int(v) for k, v in vc.head(5).items()}
            col["cardinality"] = "high" if nun > 50 else "low"
        elif e["type"] == "datetime":
            d = parse_datetime(s)
            col["min"], col["max"] = str(d.min()), str(d.max())
        if relevance and c in relevance:
            col.update(relevance[c])
        cols_out.append(col)

        {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols, "text": text_cols}[e["type"]].append(c)
        if n_missing or n_missing_test:
            missing_cols.append(c)
        if e["drop"] == "constant":
            constant.append(c)
        elif e["drop"] == "duplicate":
            duplicates[c] = e["duplicate_of"]
        elif e["drop"] == "identifier":
            ids.append(c)
        if col["is_near_constant"]:
            near_constant.append(c)

    n_cells = max(X.size, 1)
    scale_ratio = (max(stds.values()) / min(stds.values())) if len(stds) >= 2 else 1.0
    used = [c for c, e in info.items() if e["drop"] is None]
    low_rel = [c for c in used if relevance and c in relevance and not relevance[c]["relevant"]]

    # strongly correlated numeric pairs (report only)
    corr_pairs = []
    num_used = [c for c in used if info[c]["type"] == "numeric"][:200]
    if len(num_used) >= 2:
        sample = X[num_used].apply(pd.to_numeric, errors="coerce")
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=0)
        cm = sample.corr().abs().values
        iu = np.triu_indices(len(num_used), 1)
        for i, j in zip(*iu):
            if cm[i, j] >= 0.95:
                corr_pairs.append([num_used[i], num_used[j], _f(cm[i, j])])

    profile = {
        "dataset": name,
        "task": task,
        "target": target,
        "id_column": id_column,
        "metric": metric,
        "n_rows": rows,
        "n_rows_train": rows,
        "n_rows_test": None if test is None else int(len(test)),
        "n_columns": int(X.shape[1]),
        "n_features": int(X.shape[1]),
        "n_features_used": len(used),
        "feature_types": {c: info[c]["type"] for c in info},
        "numeric_features": numeric,
        "categorical_features": categorical,
        "datetime_features": datetime_cols,
        "text_features": text_cols,
        "missing_values": {
            "total_missing_cells": int(X.isna().sum().sum()),
            "missing_cell_fraction": _f(X.isna().sum().sum() / n_cells),
            "columns_with_missing": missing_cols,
            "rows_with_missing": int(X.isna().any(axis=1).sum()),
            "has_missing": bool(missing_cols),
        },
        "constant_columns": constant,
        "near_constant_columns": near_constant,
        "duplicate_columns": duplicates,
        "identifier_columns": ids,
        "dropped_columns": {c: info[c]["drop"] for c in info if info[c]["drop"]},
        "duplicate_rows": int(X.duplicated().sum()),
        "highly_correlated_pairs": corr_pairs[:50],
        "low_relevance_features": low_rel,
        "scaling": {
            "std_ratio_max_over_min": _f(scale_ratio),
            "differently_scaled": bool(scale_ratio > 10),
        },
        "columns": cols_out,
    }

    if task == "classification":
        vals, counts = np.unique(y, return_counts=True)
        order = np.argsort(-counts)
        ratio = counts.max() / max(counts.min(), 1)
        profile["target_summary"] = {
            "type": "categorical",
            "n_classes": int(len(vals)),
            "classes": [_py(v) for v in vals[order]],
            "class_counts": {str(_py(vals[i])): int(counts[i]) for i in order},
            "class_fractions": {str(_py(vals[i])): _f(counts[i] / counts.sum()) for i in order},
            "majority_class": _py(vals[order[0]]),
            "minority_class": _py(vals[order[-1]]),
            "imbalance_ratio": _f(ratio),
            "is_imbalanced": bool(ratio >= 3 or counts.min() / counts.sum() < 0.2 and len(vals) == 2),
        }
    elif task == "regression":
        yv = pd.Series(y, dtype=float)
        profile["target_summary"] = {
            "type": "numeric",
            "mean": _f(yv.mean()), "std": _f(yv.std()), "min": _f(yv.min()), "max": _f(yv.max()),
            "median": _f(yv.median()), "skewness": _f(yv.skew()),
            "is_skewed": bool(abs(yv.skew()) > 1),
            "all_non_negative": bool((yv >= 0).all()),
        }
    else:
        profile["target_summary"] = None
        profile["n_clusters"] = n_clusters
    return profile
