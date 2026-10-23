"""End-to-end run on one dataset folder: load -> profile -> model -> write."""
from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np
import pandas as pd

from .budget import Budget
from .clustering import run_clustering
from .data import Dataset, load_dataset, resolve_task
from .prep import ID_NAME, Preprocessor, analyze_columns, clean_frame, to_numeric_frame
from .profiling import build_profile, feature_relevance
from .supervised import normalize_metric, run_supervised

DEFAULT_TIME_LIMIT = 60.0
SAFETY_MARGIN = 15.0  # seconds kept free for writing outputs and graders slower than our machines


def _ids(ds: Dataset, frame: pd.DataFrame, info: dict):
    spec = ds.spec
    if spec.id_column and spec.id_column in frame.columns:
        return spec.id_column, frame[spec.id_column].values
    for c, e in info.items():
        if e["drop"] == "identifier" and c in frame.columns and ID_NAME.search(c) and not c.startswith("Unnamed"):
            return c, frame[c].values
    return "id", np.arange(len(frame))


def _write_predictions(path, id_name, ids, values, ds: Dataset, pred_name=None):
    cols = ds.submission_columns
    if cols and len(cols) >= 2 and pred_name is None:
        id_name, pred_name = cols[0], cols[1]
    pred_name = pred_name or ds.spec.prediction_column or "prediction"
    if pred_name == id_name:
        pred_name = "prediction"
    pd.DataFrame({id_name: ids, pred_name: values}).to_csv(path, index=False)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _restore_label_type(classes: np.ndarray):
    """Labels read as floats only because of missing rows go back to integers."""
    try:
        f = classes.astype(float)
        if np.all(np.isfinite(f)) and np.allclose(f, np.round(f)):
            return np.round(f).astype(np.int64)
    except (TypeError, ValueError):
        pass
    return classes


def solve(dataset_dir=None, output_dir=None, time_limit=DEFAULT_TIME_LIMIT, train_path=None, test_path=None,
          meta_path=None, pred_column=None, verbose=True, start=None):
    log = print if verbose else (lambda *a, **k: None)
    budget = Budget(max(5.0, time_limit - SAFETY_MARGIN), start=start)
    t0 = time.perf_counter()
    stages, mark = {}, [t0]

    def stage(name):
        now = time.perf_counter()
        stages[name] = round(now - mark[0], 3)
        mark[0] = now

    ds = load_dataset(dataset_dir, train_path, test_path, meta_path)
    resolve_task(ds)
    spec = ds.spec
    task = spec.task
    output_dir = output_dir or os.path.join("outputs", spec.name)
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "predictions.csv")
    log(f"[{spec.name}] task={task} target={spec.target} k={spec.n_clusters} rows={len(ds.train)}"
        f"{'' if ds.test is None else f' test_rows={len(ds.test)}'}")

    train = ds.train
    if task == "clustering":
        test_frame = train if ds.test is None else ds.test
        X = clean_frame(train)
        X_test = None if ds.test is None else clean_frame(ds.test.reindex(columns=train.columns))
        y = None
    else:
        y_raw = train[spec.target]
        if task == "regression":
            y_raw = pd.to_numeric(y_raw, errors="coerce")
        keep = y_raw.notna().values
        train = train[keep].reset_index(drop=True)
        y_raw = y_raw[keep].reset_index(drop=True)
        X = clean_frame(train.drop(columns=[spec.target]))
        test_frame = ds.test if ds.test is not None else train
        X_test = clean_frame(test_frame).reindex(columns=X.columns)
    metric = normalize_metric(spec.metric, task) if task != "clustering" else (spec.metric or "adjusted_rand_index")

    # Fallback predictions first, so a crash or time-out still leaves a valid file behind.
    if task == "classification":
        try:
            classes, y = np.unique(np.asarray(y_raw), return_inverse=True)
        except TypeError:  # mixed label types
            classes, y = np.unique(np.asarray(y_raw.astype(str)), return_inverse=True)
        classes = _restore_label_type(classes)
        fallback = np.repeat(classes[np.bincount(y).argmax()], len(test_frame))
    elif task == "regression":
        y = y_raw.values.astype(float)
        fallback = np.repeat(float(np.median(y)), len(test_frame))
    else:
        fallback = np.zeros(len(test_frame), dtype=int)
    _write_predictions(pred_path, "id", np.arange(len(test_frame)), fallback, ds, pred_column)
    stage("load")

    # Column analysis and profile.
    info = analyze_columns(X, X_test, spec.feature_types, spec.id_column)
    X = to_numeric_frame(X, info)
    if X_test is not None:
        X_test = to_numeric_frame(X_test, info)
    stage("column_analysis")
    relevance = None
    if task != "clustering" and budget.left() > 0.6 * budget.total:
        try:
            relevance = feature_relevance(X, y, info, task)
        except Exception as exc:
            log(f"  relevance skipped: {exc}")
    stage("feature_relevance")
    profile = build_profile(spec.name, task, spec.target, spec.id_column, spec.n_clusters, train,
                            X_test, X, None if y is None else (classes[y] if task == "classification" else y),
                            info, metric, relevance)
    profile_path = os.path.join(output_dir, "profile.json")

    def write_profile():
        profile["timing"] = {"seconds_total": round(time.perf_counter() - t0, 3),
                             "seconds_since_process_start": round(budget.elapsed(), 3),
                             "stages": stages}
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, default=_json_default)

    stage("profile")
    write_profile()
    id_name, ids = _ids(ds, test_frame, info)

    try:
        prep = Preprocessor(info).fit(X)
        profile["preprocessing"] = {
            "dropped_columns": {c: e["drop"] for c, e in info.items() if e["drop"]},
            "numeric_features": prep.num_cols, "categorical_features": prep.cat_cols,
            "datetime_features": prep.dt_cols, "text_features": prep.text_cols,
            "log_transformed_for_linear_models": [f for f, flag in zip(prep.num_feats, prep.log_flag) if flag],
            "imputation": "median + missing indicators (linear view); native NaN routing (gradient boosting)",
            "scaling": "standardisation after log1p of skewed non-negative features (linear view)",
            "categorical_encoding": "integer codes for trees, one-hot (top levels + other + missing) for linear models",
            "n_tree_features": len(prep.tree_names), "n_linear_features": len(prep.lin_names),
        }

        if task == "clustering":
            Xall = prep.linear_matrix(X)
            labels, report = run_clustering(Xall, spec.n_clusters, budget, log=log)
            if X_test is not None:  # separate rows to label: nearest cluster centre
                Xt = prep.linear_matrix(X_test)
                centers = np.vstack([Xall[labels == j].mean(0) for j in range(labels.max() + 1)])
                labels = np.argmin(((Xt[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
            profile["model"] = report
            _write_predictions(pred_path, id_name, ids, labels, ds, pred_column)
        else:
            views_train = {"tree": prep.tree_matrix(X), "linear": prep.linear_matrix(X)}
            views_train["filled"] = prep.filled_tree_matrix(X)
            views_test = {"tree": prep.tree_matrix(X_test), "linear": prep.linear_matrix(X_test),
                          "filled": prep.filled_tree_matrix(X_test)}
            relevance_cols = None
            if relevance:
                bad = {c for c, r in relevance.items() if not r["relevant"]}
                if bad:
                    relevance_cols = {
                        "tree": [i for i, s in enumerate(prep.tree_src) if s not in bad],
                        "filled": [i for i, s in enumerate(prep.tree_src) if s not in bad],
                        "linear": [i for i, s in enumerate(prep.lin_src) if s not in bad],
                    }
            imbalanced = bool(profile["target_summary"].get("is_imbalanced")) if task == "classification" else False
            preds, proba, report = run_supervised(task, views_train, views_test, y, prep.cat_mask, budget, metric,
                                                  imbalanced=imbalanced, relevance_cols=relevance_cols, log=log)
            profile["model"] = report
            profile["nonlinearity"] = report.get("nonlinearity")
            if task == "classification":
                out = classes[preds]
                pd.DataFrame(proba, columns=[f"proba_{c}" for c in classes]).assign(
                    **{id_name: ids}).iloc[:, [-1] + list(range(len(classes)))].to_csv(
                    os.path.join(output_dir, "probabilities.csv"), index=False)
            else:
                lo, hi = float(np.min(y)), float(np.max(y))
                pad = 0.1 * (hi - lo)
                out = np.clip(preds, max(lo - pad, 0.0) if lo >= 0 else lo - pad, hi + pad)
                if np.allclose(y, np.round(y)) and lo >= 0:
                    profile["model"]["target_is_count_like"] = True
            _write_predictions(pred_path, id_name, ids, out, ds, pred_column)
        profile["status"] = "ok"
    except Exception as exc:
        profile["status"] = f"fallback predictions used: {type(exc).__name__}: {exc}"
        profile["error"] = traceback.format_exc(limit=5)
        log(f"  ERROR -> fallback predictions kept: {exc}")
        _write_predictions(pred_path, id_name, ids, fallback, ds, pred_column)

    stage("modelling")
    write_profile()
    log(f"[{spec.name}] done in {time.perf_counter() - t0:.1f}s -> {output_dir}")
    return profile
