"""Find and load a dataset folder: its tables and its metadata.

The kickoff contract was not available while this was written, so loading is
deliberately tolerant: it accepts the common layouts (train/test files,
X/y splits, a single table for clustering, JSON or YAML metadata under many
key spellings) and falls back to inference when something is not declared.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TABLE_EXT = (".csv", ".tsv", ".txt", ".data", ".parquet", ".pq", ".feather", ".xlsx", ".xls")
META_EXT = (".json", ".yaml", ".yml")


@dataclass
class Spec:
    name: str
    task: str | None = None
    target: str | None = None
    id_column: str | None = None
    n_clusters: int | None = None
    metric: str | None = None
    feature_types: dict = field(default_factory=dict)
    prediction_column: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class Dataset:
    spec: Spec
    train: pd.DataFrame
    test: pd.DataFrame | None
    folder: str
    submission_columns: list | None = None


def _norm(key) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _walk(obj, depth=0):
    """Yield (normalised key, value) pairs breadth-first so top-level keys win."""
    level = [obj]
    while level and depth < 4:
        nxt = []
        for item in level:
            if isinstance(item, dict):
                for k, v in item.items():
                    yield _norm(k), v
                    nxt.append(v)
            elif isinstance(item, list):
                nxt.extend(x for x in item if isinstance(x, (dict, list)))
        level = nxt
        depth += 1


def _find(meta, names, accept=lambda v: v is not None):
    names = {_norm(n) for n in names}
    for k, v in _walk(meta):
        if k in names and accept(v):
            return v
    return None


def normalize_task(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.lower()
    if "clust" in v or "segment" in v or "unsupervised" in v:
        return "clustering"
    if "regress" in v:
        return "regression"
    if "class" in v or v in ("binary", "multiclass", "multi-class"):
        return "classification"
    return None


def _as_int(v):
    try:
        i = int(v)
        return i if i > 0 and float(v) == i else None
    except (TypeError, ValueError):
        return None


def _parse_feature_types(meta) -> dict:
    types: dict = {}

    def add(col, kind):
        if isinstance(col, str) and isinstance(kind, str):
            types.setdefault(col, kind)

    # {"feature_types": {"col": "numeric"}} or a list of {"name":..., "type":...}
    for key in ("feature_types", "featuretypes", "column_types", "columntypes", "dtypes", "types",
                "features", "columns", "schema", "feature_schema"):
        v = _find(meta, [key], accept=lambda x: isinstance(x, (dict, list)))
        if isinstance(v, dict):
            for col, kind in v.items():
                if isinstance(kind, str):
                    add(col, kind)
                elif isinstance(kind, list):  # {"numeric": [cols], "categorical": [cols]}
                    for c in kind:
                        add(c, col)
                elif isinstance(kind, dict):
                    add(col, kind.get("type") or kind.get("dtype") or kind.get("kind"))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("column") or item.get("feature")
                    add(name, item.get("type") or item.get("dtype") or item.get("kind"))

    # {"numeric_features": [...], "categorical_features": [...]}
    for k, v in _walk(meta):
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            for kind in ("numeric", "numerical", "continuous", "categorical", "nominal", "ordinal",
                         "datetime", "date", "text", "boolean", "binary", "id"):
                if k.startswith(kind) and ("feature" in k or "column" in k or "col" in k or k == kind):
                    for c in v:
                        add(c, kind)
    return types


def parse_spec(meta: dict, name: str) -> Spec:
    spec = Spec(name=name, raw=meta)
    spec.task = normalize_task(_find(meta, ["task", "task_type", "problem_type", "problem",
                                            "objective", "learning_task", "ml_task"],
                                     accept=lambda v: normalize_task(v) is not None))
    if spec.task is None:  # generic keys only count at the top level, not inside feature lists
        spec.task = next((normalize_task(v) for k, v in meta.items()
                          if _norm(k) in ("type", "kind") and normalize_task(v)), None)
    tgt = _find(meta, ["target", "target_column", "target_col", "target_name", "label",
                       "label_column", "label_col", "y", "y_column", "outcome"],
                accept=lambda v: isinstance(v, (str, list)))
    if isinstance(tgt, list):
        tgt = tgt[0] if len(tgt) == 1 and isinstance(tgt[0], str) else None
    spec.target = tgt
    idc = _find(meta, ["id_column", "id_col", "id", "index_column", "row_id", "key"],
                accept=lambda v: isinstance(v, str))
    spec.id_column = idc
    spec.n_clusters = _as_int(_find(meta, ["n_clusters", "num_clusters", "number_of_clusters", "k",
                                           "n_groups", "num_groups", "number_of_groups", "groups",
                                           "clusters", "n_segments", "required_groups"],
                                    accept=lambda v: _as_int(v) is not None))
    metric = _find(meta, ["metric", "eval_metric", "evaluation_metric", "scoring", "score", "metrics"],
                   accept=lambda v: isinstance(v, (str, list)))
    if isinstance(metric, list):
        metric = next((m for m in metric if isinstance(m, str)), None)
    spec.metric = metric
    spec.prediction_column = _find(meta, ["prediction_column", "output_column", "prediction_col"],
                                   accept=lambda v: isinstance(v, str))
    spec.feature_types = _parse_feature_types(meta)
    if spec.task == "clustering" and spec.n_clusters is None:
        spec.n_clusters = _as_int(_find(meta, ["n", "count"], accept=lambda v: _as_int(v) is not None))
    return spec


def read_meta(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as fh:
        text = fh.read()
    if path.lower().endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # optional
        return yaml.safe_load(text) or {}
    except ImportError:
        out = {}
        for line in text.splitlines():  # flat "key: value" fallback
            m = re.match(r"\s*([^#:]+):\s*(.+?)\s*$", line)
            if m:
                out[m.group(1).strip()] = m.group(2).strip().strip("'\"")
        return out


def _sniff_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        head = [fh.readline() for _ in range(5)]
    counts = {sep: sum(line.count(sep) for line in head) for sep in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_table(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if low.endswith(".feather"):
        return pd.read_feather(path)
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=_sniff_sep(path), encoding="utf-8-sig", low_memory=False)


def _files(folder: str, exts, max_depth=2):
    out = []
    base_depth = folder.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in ("output", "outputs", "predictions")]
        if root.count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
        for f in files:
            if f.lower().endswith(exts) and not f.startswith("."):
                out.append(os.path.join(root, f))
    return sorted(out)


def is_dataset_folder(folder: str) -> bool:
    return bool(_files(folder, TABLE_EXT, max_depth=1))


def _role(path: str) -> str:
    stem = _norm(os.path.splitext(os.path.basename(path))[0])
    if "submission" in stem or "sample" in stem and "sub" in stem:
        return "submission"
    if any(w in stem for w in ("prediction", "output", "solution", "answer")):
        return "ignore"
    is_train = "train" in stem
    is_test = any(w in stem for w in ("test", "eval", "holdout", "predict", "score", "unlabeled", "unlabelled"))
    is_y = stem.startswith("y") and (is_train or is_test) or any(
        w in stem for w in ("label", "target")) and (is_train or is_test)
    if is_train and not is_test:
        return "train_y" if is_y else "train"
    if is_test and not is_train:
        return "test_y" if is_y else "test"
    if any(w in stem for w in ("data", "dataset", "features", "x", "all", "full")):
        return "data"
    return "other"


def load_dataset(folder: str | None = None, train_path=None, test_path=None, meta_path=None) -> Dataset:
    folder = folder or os.path.dirname(os.path.abspath(train_path))
    name = os.path.basename(os.path.normpath(folder))

    if meta_path is None:
        metas = _files(folder, META_EXT)
        preferred = [m for m in metas if any(w in _norm(os.path.basename(m)) for w in
                                              ("meta", "info", "task", "spec", "schema", "config", "dataset"))]
        meta_path = (preferred or metas or [None])[0]
    meta = read_meta(meta_path) if meta_path else {}
    if not isinstance(meta, dict):
        meta = {"items": meta}
    spec = parse_spec(meta, name)

    roles: dict = {}
    if train_path is None:
        for p in _files(folder, TABLE_EXT):
            roles.setdefault(_role(p), p)
        tables = [p for p in _files(folder, TABLE_EXT) if _role(p) not in ("submission", "ignore")]
        train_path = roles.get("train") or roles.get("data") or (tables[0] if len(tables) == 1 else None)
        if train_path is None:  # pick the largest remaining table
            rest = [p for p in tables if p != roles.get("test")]
            train_path = max(rest, key=os.path.getsize) if rest else None
        if test_path is None:
            test_path = roles.get("test")
    if train_path is None:
        raise FileNotFoundError(f"No data table found in {folder}")

    train = read_table(train_path)
    test = read_table(test_path) if test_path else None

    if "train_y" in roles:
        y = read_table(roles["train_y"])
        y = y.drop(columns=[c for c in y.columns if spec.id_column and c == spec.id_column or
                            str(c).lower().startswith("unnamed")], errors="ignore")
        ycol = spec.target if spec.target in y.columns else y.columns[-1]
        spec.target = spec.target or str(ycol)
        train[spec.target] = y[ycol].values

    submission_columns = None
    if "submission" in roles:
        try:
            submission_columns = list(pd.read_csv(roles["submission"], nrows=2).columns)
        except Exception:
            submission_columns = None

    # A single table whose target is missing for some rows: those rows are the test set.
    if test is None and spec.target in train.columns and spec.task != "clustering":
        unlabeled = train[spec.target].isna()
        if 0 < unlabeled.sum() < len(train):
            test = train[unlabeled].drop(columns=[spec.target]).reset_index(drop=True)
            train = train[~unlabeled].reset_index(drop=True)

    return Dataset(spec=spec, train=train, test=test, folder=folder, submission_columns=submission_columns)


def resolve_task(ds: Dataset) -> None:
    """Fill in target / task / n_clusters when the metadata left them out."""
    spec, train, test = ds.spec, ds.train, ds.test
    cols = [str(c) for c in train.columns]
    train.columns = cols
    if test is not None:
        test.columns = [str(c) for c in test.columns]

    if spec.target and spec.target not in cols:
        match = [c for c in cols if _norm(c) == _norm(spec.target)]
        spec.target = match[0] if match else None

    if spec.target is None and spec.task != "clustering":
        extra = [c for c in cols if test is not None and c not in test.columns]
        named = [c for c in cols if _norm(c) in ("target", "label", "y", "class", "outcome", "churn")]
        if len(extra) == 1:
            spec.target = extra[0]
        elif named:
            spec.target = named[-1]
        elif spec.task in ("classification", "regression") or test is not None:
            spec.target = cols[-1]

    if spec.task is None:
        if spec.target is None or spec.n_clusters:
            spec.task = "clustering"
        else:
            y = train[spec.target].dropna()
            numeric = pd.api.types.is_numeric_dtype(y)
            nun = y.nunique()
            integral = numeric and np.allclose(y.astype(float), np.round(y.astype(float)))
            if numeric and (not integral and nun > 10 or nun > max(20, 0.05 * len(y))):
                spec.task = "regression"
            else:
                spec.task = "classification"

    if spec.task == "clustering":
        # Labels are never used for clustering, even if a column was left in.
        if spec.target in train.columns:
            train.drop(columns=[spec.target], inplace=True)
        spec.target = None
        if not spec.n_clusters:
            spec.n_clusters = None  # chosen later from the data

    if spec.id_column and spec.id_column not in cols:
        match = [c for c in cols if _norm(c) == _norm(spec.id_column)]
        spec.id_column = match[0] if match else None
