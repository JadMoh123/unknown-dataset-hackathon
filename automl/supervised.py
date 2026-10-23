"""Classification and regression: a time-boxed model search with out-of-fold
scoring, greedy ensembling, and a final refit on all training rows."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

# --------------------------------------------------------------------------- metrics

CLS_METRICS = {
    "accuracy": "accuracy", "acc": "accuracy",
    "balancedaccuracy": "balanced_accuracy", "bacc": "balanced_accuracy",
    "f1": "f1_macro", "f1macro": "f1_macro", "macrof1": "f1_macro", "f1score": "f1_macro",
    "f1binary": "f1_binary", "f1weighted": "f1_weighted", "weightedf1": "f1_weighted",
    "rocauc": "roc_auc", "auc": "roc_auc", "auroc": "roc_auc",
    "logloss": "log_loss", "crossentropy": "log_loss",
}
REG_METRICS = {
    "rmse": "rmse", "rootmeansquarederror": "rmse", "mse": "rmse", "meansquarederror": "rmse",
    "mae": "mae", "meanabsoluteerror": "mae",
    "r2": "r2", "rsquared": "r2", "r2score": "r2",
    "rmsle": "rmsle", "msle": "rmsle", "rootmeansquaredlogerror": "rmsle",
    "mape": "mae",
}
DEFAULT_METRIC = {"classification": "f1_macro", "regression": "rmse"}


def normalize_metric(name, task):
    if not isinstance(name, str):
        return DEFAULT_METRIC[task]
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    table = CLS_METRICS if task == "classification" else REG_METRICS
    if key in table:
        return table[key]
    for k, v in table.items():
        if k in key:
            return v
    return DEFAULT_METRIC[task]


def _label_metric(metric, y, labels, k):
    """Confusion-matrix metrics in numpy; called hundreds of times while tuning."""
    cm = np.bincount(y * k + labels, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    if metric == "accuracy":
        return tp.sum() / cm.sum()
    if metric == "balanced_accuracy":
        present = support > 0
        return float(np.mean(tp[present] / support[present]))
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    if metric == "f1_binary" and k == 2:
        return float(f1[1])
    if metric == "f1_weighted":
        return float((f1 * support).sum() / support.sum())
    present = (support + predicted) > 0  # sklearn macro-F1 averages over labels seen in y or predictions
    return float(f1[present].mean())


def score_cls(metric, y, proba, labels):
    """Higher is better. `labels` are the predicted class indices."""
    if metric in ("accuracy", "balanced_accuracy", "f1_weighted", "f1_binary", "f1_macro"):
        return _label_metric(metric, y, labels, proba.shape[1])
    if metric == "roc_auc":
        try:
            if proba.shape[1] == 2:
                return roc_auc_score(y, proba[:, 1])
            return roc_auc_score(y, proba, multi_class="ovr", labels=np.arange(proba.shape[1]))
        except ValueError:
            return accuracy_score(y, labels)
    if metric == "log_loss":
        return -log_loss(y, np.clip(proba, 1e-7, 1), labels=np.arange(proba.shape[1]))
    return f1_score(y, labels, average="macro")


def score_reg(metric, y, pred):
    if metric == "mae":
        return -mean_absolute_error(y, pred)
    if metric == "r2":
        return r2_score(y, pred)
    if metric == "rmsle":
        return -np.sqrt(mean_squared_error(np.log1p(np.clip(y, 0, None)), np.log1p(np.clip(pred, 0, None))))
    return -np.sqrt(mean_squared_error(y, pred))


# --------------------------------------------------------------------------- candidates

@dataclass
class Candidate:
    name: str
    view: str                      # "tree" | "filled" | "linear"
    build: Callable                # build(ctx) -> unfitted estimator
    log_target: bool = False       # regression: fit on log1p(y)
    scale_target: bool = False     # regression: fit on standardised y
    cols: np.ndarray | None = None  # column subset of the view (feature selection variant)
    oof: np.ndarray | None = None
    score: float = -np.inf
    fold_seconds: float = 0.0
    meta: dict = field(default_factory=dict)


def _hgb_params(n, cat_mask):
    small = n < 1500
    return dict(
        learning_rate=0.05 if small else 0.08,
        max_iter=250 if small else 600,
        max_leaf_nodes=15 if small else 31,
        min_samples_leaf=max(5, min(20, n // 100)),
        l2_regularization=0.5 if small else 0.1,
        early_stopping=not small,
        validation_fraction=0.12,
        n_iter_no_change=30,
        categorical_features=cat_mask if cat_mask is not None and cat_mask.any() else None,
        random_state=0,
    )


def _trees(n):
    return 300 if n <= 5000 else 200 if n <= 15000 else 100


def _drop_slow(cands, n):
    # Random forest is the slowest learner and on large tables rarely beats boosting;
    # extra-trees keeps a bagged model in the pool at a fraction of the cost.
    return [c for c in cands if not (c.name == "random_forest" and n > 15000)]


def classification_candidates(n, n_classes, imbalanced):
    c = [
        Candidate("hist_gradient_boosting", "tree",
                  lambda ctx: HistGradientBoostingClassifier(**_hgb_params(ctx["n"], ctx["cat_mask"]))),
        Candidate("logistic_regression", "linear",
                  lambda ctx: LogisticRegression(C=0.5, max_iter=2000)),
        Candidate("random_forest", "filled",
                  lambda ctx: RandomForestClassifier(n_estimators=_trees(ctx["n"]), max_features="sqrt",
                                                     min_samples_leaf=1, n_jobs=-1, random_state=0)),
        Candidate("extra_trees", "filled",
                  lambda ctx: ExtraTreesClassifier(n_estimators=_trees(ctx["n"]), max_features="sqrt",
                                                   min_samples_leaf=1, n_jobs=-1, random_state=0)),
    ]
    if n <= 5000:
        c.append(Candidate("svm_rbf", "linear",
                           lambda ctx: SVC(C=1.0, gamma="scale", probability=True, random_state=0)))
    if n <= 20000:
        c.append(Candidate("knn", "linear",
                           lambda ctx: KNeighborsClassifier(n_neighbors=min(25, max(5, int(np.sqrt(ctx["n"]) / 2))),
                                                            weights="distance", n_jobs=-1)))
    if imbalanced:
        c.insert(1, Candidate("hist_gradient_boosting_balanced", "tree",
                              lambda ctx: HistGradientBoostingClassifier(class_weight="balanced",
                                                                         **_hgb_params(ctx["n"], ctx["cat_mask"]))))
    return c


def regression_candidates(n, log_target):
    c = [
        Candidate("hist_gradient_boosting", "tree",
                  lambda ctx: HistGradientBoostingRegressor(**_hgb_params(ctx["n"], ctx["cat_mask"])),
                  log_target=log_target),
        Candidate("ridge", "linear", lambda ctx: RidgeCV(alphas=np.logspace(-3, 3, 13)), log_target=log_target),
        Candidate("random_forest", "filled",
                  lambda ctx: RandomForestRegressor(n_estimators=_trees(ctx["n"]), max_features=0.5,
                                                    n_jobs=-1, random_state=0), log_target=log_target),
        Candidate("extra_trees", "filled",
                  lambda ctx: ExtraTreesRegressor(n_estimators=_trees(ctx["n"]), max_features=0.7,
                                                  n_jobs=-1, random_state=0), log_target=log_target),
    ]
    if n <= 5000:
        c.append(Candidate("svr_rbf", "linear", lambda ctx: SVR(C=3.0, epsilon=0.05, gamma="scale"),
                           log_target=log_target, scale_target=True))
    if n <= 20000:
        c.append(Candidate("knn", "linear",
                           lambda ctx: KNeighborsRegressor(n_neighbors=min(25, max(5, int(np.sqrt(ctx["n"]) / 2))),
                                                           weights="distance", n_jobs=-1),
                           log_target=log_target))
    return c


# --------------------------------------------------------------------------- fitting helpers

def _ctx(cand, n, cat_mask):
    mask = cat_mask if cand.view == "tree" else None
    if mask is not None and cand.cols is not None:
        mask = mask[cand.cols]
    return {"n": n, "cat_mask": mask}


def _view(views, cand, rows=None):
    X = views[cand.view]
    if cand.cols is not None:
        X = X[:, cand.cols]
    return X if rows is None else X[rows]


def fit_predict(cand, Xtr, ytr, Xte, task, n_classes, cat_mask):
    est = cand.build(_ctx(cand, len(Xtr), cat_mask))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "classification":
            est.fit(Xtr, ytr)
            p = est.predict_proba(Xte)
            out = np.zeros((len(Xte), n_classes))
            out[:, est.classes_.astype(int)] = p
            return out
        t = np.log1p(np.clip(ytr, 0, None)) if cand.log_target else ytr.astype(float)
        mu, sd = (t.mean(), t.std() or 1.0) if cand.scale_target else (0.0, 1.0)
        est.fit(Xtr, (t - mu) / sd)
        p = est.predict(Xte) * sd + mu
        return np.expm1(np.clip(p, None, 50)) if cand.log_target else p


def make_folds(y, task, n, seed=0):
    k = 5 if n <= 2000 else 4 if n <= 8000 else 3
    if task == "classification":
        counts = np.bincount(y)
        if counts[counts > 0].min() >= k:
            return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), y))
    return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n)))


# --------------------------------------------------------------------------- decision rules

def decide(proba, rule):
    if rule.get("threshold") is not None:
        return (proba[:, 1] >= rule["threshold"]).astype(int)
    w = rule.get("class_weights")
    return np.argmax(proba * w if w is not None else proba, axis=1)


def tune_rule(metric, y, proba):
    """Pick the decision rule (binary threshold or class re-weighting) that maximises the metric."""
    base = {"threshold": None, "class_weights": None}
    if metric in ("roc_auc", "log_loss", "accuracy"):
        return base, score_cls(metric, y, proba, decide(proba, base))
    best, best_s = base, score_cls(metric, y, proba, decide(proba, base))
    if proba.shape[1] == 2:
        grid = np.unique(np.quantile(proba[:, 1], np.linspace(0.02, 0.98, 49)))
        scores = np.array([score_cls(metric, y, proba, (proba[:, 1] >= t).astype(int)) for t in grid])
        if len(grid) and scores.max() > best_s + 1e-9:
            # median of the near-optimal thresholds is less noisy than the arg-max
            t = float(np.median(grid[scores >= scores.max() - 0.002]))
            r = {"threshold": t, "class_weights": None}
            s = score_cls(metric, y, proba, decide(proba, r))
            if s > best_s:
                best, best_s = r, s
    else:
        prior = np.bincount(y, minlength=proba.shape[1]) / len(y)
        for a in (0.25, 0.5, 0.75, 1.0):
            r = {"threshold": None, "class_weights": 1.0 / np.power(np.clip(prior, 1e-6, None), a)}
            s = score_cls(metric, y, proba, decide(proba, r))
            if s > best_s + 1e-9:
                best, best_s = r, s
    return best, best_s


# --------------------------------------------------------------------------- search

def _blend(cands, weights):
    tot = sum(weights.values())
    return sum(c.oof * (w / tot) for c, w in zip(cands, weights.values()) if w) if tot else None


def greedy_ensemble(cands, scorer, rounds=12):
    """Caruana-style forward selection with replacement on out-of-fold predictions."""
    best = max(cands, key=lambda c: c.score)
    weights = {c.name: 0 for c in cands}
    weights[best.name] = 1
    best_score = best.score
    current = dict(weights)
    for _ in range(rounds):
        trial_best, trial_name = -np.inf, None
        for c in cands:
            trial = dict(current)
            trial[c.name] += 1
            s = scorer(_blend(cands, trial))
            if s > trial_best:
                trial_best, trial_name = s, c.name
        current[trial_name] += 1
        if trial_best > best_score + 1e-6:
            best_score, weights = trial_best, dict(current)
    return weights, best_score


def run_supervised(task, views_train, views_test, y, cat_mask, budget, metric, imbalanced=False,
                   relevance_cols=None, log=print):
    n = len(y)
    n_classes = int(y.max()) + 1 if task == "classification" else 0
    report = {"metric": metric, "candidates": {}, "skipped": []}

    # Model selection runs on at most SEL_ROWS rows; the final refit uses all of them.
    SEL_ROWS = 25000
    if n > SEL_ROWS:
        rng = np.random.RandomState(0)
        sel = np.sort(rng.choice(n, SEL_ROWS, replace=False))
    else:
        sel = np.arange(n)
    ys = y[sel]
    vs = {k: v[sel] for k, v in views_train.items()}
    folds = make_folds(ys, task, len(sel))
    report["cv_folds"] = len(folds)
    report["selection_rows"] = int(len(sel))

    log_target = False
    if task == "regression":
        skew = float(np.nan_to_num(((ys - ys.mean()) ** 3).mean() / (ys.std() ** 3 + 1e-12)))
        log_target = bool(ys.min() >= 0 and (skew > 1.0 or metric == "rmsle"))
        cands = regression_candidates(len(sel), log_target)
        if log_target and metric != "rmsle":  # also try the raw scale for the main learner
            raw = regression_candidates(len(sel), False)[0]
            raw.name += "_raw_target"
            cands.insert(1, raw)
        report["log_target"] = log_target
    else:
        cands = classification_candidates(len(sel), n_classes, imbalanced)
    cands = _drop_slow(cands, len(sel))

    def oof_score(c_oof):
        if task == "classification":
            return tune_rule(metric, ys, c_oof)[1]
        return score_reg(metric, ys, c_oof)

    reserve = max(4.0, 0.22 * budget.total)   # kept for the final refit + writing outputs
    done = []

    def evaluate(cand):
        oof = np.zeros((len(sel), n_classes)) if task == "classification" else np.zeros(len(sel))
        t0 = time.perf_counter()
        for i, (tr, va) in enumerate(folds):
            if budget.left() < reserve:
                report["skipped"].append(cand.name)
                return False
            oof[va] = fit_predict(cand, _view(vs, cand, tr), ys[tr], _view(vs, cand, va), task, n_classes, cat_mask)
            # project the remaining folds; abandon early if they cannot fit
            per = (time.perf_counter() - t0) / (i + 1)
            if budget.left() - per * (len(folds) - i - 1) < reserve:
                report["skipped"].append(cand.name)
                return False
        cand.fold_seconds = (time.perf_counter() - t0) / len(folds)
        cand.oof = oof
        cand.score = float(oof_score(oof))
        done.append(cand)
        report["candidates"][cand.name] = {"cv_score": round(cand.score, 6), "fold_seconds": round(cand.fold_seconds, 3)}
        log(f"  {cand.name:<34} {metric}={cand.score:.5f}  ({cand.fold_seconds:.2f}s/fold)")
        return True

    for cand in cands:
        if budget.left() < reserve + 1:
            report["skipped"].append(cand.name)
            continue
        try:
            evaluate(cand)
        except Exception as exc:  # one failing learner must never sink the run
            report["skipped"].append(f"{cand.name}: {type(exc).__name__}")
            log(f"  {cand.name} failed: {exc}")

    if not done:
        raise RuntimeError("no candidate finished within the time budget")

    # Feature-selection variant: drop features with no measurable relation to the target.
    best = max(done, key=lambda c: c.score)
    if relevance_cols is not None and budget.left() > reserve + 3 * best.fold_seconds * len(folds):
        keep = relevance_cols.get(best.view)
        if keep is not None and 0 < len(keep) < views_train[best.view].shape[1]:
            sel_c = Candidate(best.name + "_selected_features", best.view, best.build, best.log_target,
                              best.scale_target, cols=np.asarray(keep))
            try:
                evaluate(sel_c)
            except Exception as exc:
                log(f"  feature-selection variant failed: {exc}")

    # Greedy ensemble on the out-of-fold predictions.
    weights, ens_score = greedy_ensemble(done, oof_score)
    best = max(done, key=lambda c: c.score)
    use_ensemble = ens_score > best.score + 1e-4 and sum(1 for w in weights.values() if w) > 1
    if not use_ensemble:
        weights = {c.name: int(c is best) for c in done}
    chosen = [c for c in done if weights[c.name]]
    report["ensemble"] = {c.name: weights[c.name] for c in chosen}
    report["selected_cv_score"] = round(float(ens_score if use_ensemble else best.score), 6)
    report["best_single_model"] = best.name

    # Decision rule for classification, tuned on the blended out-of-fold probabilities.
    rule = {"threshold": None, "class_weights": None}
    if task == "classification":
        rule, _ = tune_rule(metric, ys, _blend(done, weights))
        report["decision_rule"] = {"threshold": rule["threshold"],
                                   "class_weights": None if rule["class_weights"] is None
                                   else [round(float(w), 4) for w in rule["class_weights"]]}

    # Refit on every training row; drop the slowest members if time is short.
    scale = n / max(len(sel) * (1 - 1 / len(folds)), 1)
    chosen.sort(key=lambda c: -weights[c.name])
    while len(chosen) > 1 and sum(c.fold_seconds * scale * 1.3 for c in chosen) > budget.left() - 3:
        dropped = max(chosen[1:], key=lambda c: c.fold_seconds)
        chosen.remove(dropped)
        weights[dropped.name] = 0
    final_rows = np.arange(n)
    if chosen[0].fold_seconds * scale * 1.3 > budget.left() - 3:
        final_rows = sel  # not enough time to refit on everything: reuse the selection sample
        report["final_fit_rows"] = int(len(sel))

    preds, tot = None, 0
    for c in chosen:
        if c is not chosen[0] and budget.left() < 3:
            break
        Xtr = _view(views_train, c, final_rows)
        p = fit_predict(c, Xtr, y[final_rows], _view(views_test, c), task, n_classes, cat_mask)
        preds = p * weights[c.name] if preds is None else preds + p * weights[c.name]
        tot += weights[c.name]
    preds = preds / tot
    report["final_models"] = [c.name for c in chosen]

    linear_names = {"logistic_regression", "ridge"}
    tree_names = {"hist_gradient_boosting", "random_forest", "extra_trees"}
    lin = max([c.score for c in done if c.name in linear_names], default=None)
    tre = max([c.score for c in done if c.name in tree_names], default=None)
    report["nonlinearity"] = {
        "linear_cv_score": lin, "tree_cv_score": tre,
        "nonlinear_relationships_detected": bool(lin is not None and tre is not None and tre - lin > 0.01 * max(abs(lin), 1e-3)),
    }

    if task == "classification":
        return decide(preds, rule), preds, report
    return preds, None, report
