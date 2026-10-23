"""Column analysis, type inference and feature encoding shared by every task.

`analyze_columns` decides what each raw column is and which ones to drop
(constant, duplicate, identifier). `Preprocessor` then turns the kept columns
into two numeric views of the same rows:

* tree view   - raw numbers with NaN kept, categoricals as integer codes
                (for gradient boosting / random forests);
* linear view - imputed, log-compressed when skewed, standardised numbers
                plus one-hot categoricals (for linear models, SVM, k-NN and
                clustering).
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "null", "none", "-", "--", "#n/a", "missing", "undefined", "nil"}

TYPE_ALIASES = {
    "numeric": "numeric", "numerical": "numeric", "number": "numeric", "num": "numeric",
    "continuous": "numeric", "float": "numeric", "int": "numeric", "integer": "numeric",
    "real": "numeric", "double": "numeric", "count": "numeric", "quantitative": "numeric",
    "categorical": "categorical", "category": "categorical", "cat": "categorical",
    "nominal": "categorical", "ordinal": "categorical", "string": "categorical", "str": "categorical",
    "object": "categorical", "bool": "categorical", "boolean": "categorical", "binary": "categorical",
    "discrete": "categorical", "qualitative": "categorical",
    "datetime": "datetime", "date": "datetime", "time": "datetime", "timestamp": "datetime",
    "text": "text", "freetext": "text",
    "id": "id", "identifier": "id", "index": "id",
}

ID_NAME = re.compile(
    r"^(?i:id|index|idx|row_?id|row_?num(ber)?|record_?id|key|uuid|guid)$"
    r"|^Unnamed: ?\d+$"
    r"|[_\s\-.](?i:id)$"
    r"|[a-z0-9](ID|Id)$"
    r"|^(?i:id)[_\s\-.]"
)


def canonical_type(declared) -> str | None:
    if not isinstance(declared, str):
        return None
    key = re.sub(r"[^a-z]", "", declared.lower())
    if key in TYPE_ALIASES:
        return TYPE_ALIASES[key]
    for prefix, kind in (("float", "numeric"), ("int", "numeric"), ("uint", "numeric"),
                         ("datetime", "datetime"), ("date", "datetime"), ("categor", "categorical")):
        if key.startswith(prefix):
            return kind
    return None


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Strip strings and turn the usual missing-value spellings into NaN."""
    df = df.copy()
    names, seen = [], {}
    for c in df.columns:
        c = str(c).strip()
        if c in seen:
            seen[c] += 1
            c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        names.append(c)
    df.columns = names
    for c in df.columns:
        s = df[c]
        if ptypes.is_numeric_dtype(s) or ptypes.is_bool_dtype(s) or ptypes.is_datetime64_any_dtype(s):
            if ptypes.is_float_dtype(s):
                df[c] = s.replace([np.inf, -np.inf], np.nan)
            continue
        mask = s.notna()
        st = s[mask].astype(str).str.strip()
        st = st.where(~st.str.lower().isin(MISSING_TOKENS))
        out = pd.Series(np.nan, index=s.index, dtype=object)
        out[mask] = st.astype(object)
        df[c] = out
    return df


def parse_datetime(s: pd.Series) -> pd.Series:
    if ptypes.is_datetime64_any_dtype(s):
        return s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(s, errors="coerce")


def infer_type(s: pd.Series) -> str:
    if ptypes.is_bool_dtype(s):
        return "categorical"
    if ptypes.is_numeric_dtype(s):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(s):
        return "datetime"
    nonnull = s.dropna()
    if len(nonnull) == 0:
        return "categorical"
    if pd.to_numeric(nonnull, errors="coerce").notna().mean() >= 0.95:
        return "numeric"
    sample = nonnull.astype(str)
    if len(sample) > 400:
        sample = sample.sample(400, random_state=0)
    if sample.str.contains(r"\d").mean() > 0.9 and sample.str.contains(r"[-/:]").mean() > 0.9:
        parsed = parse_datetime(sample)
        ok = parsed.notna() & parsed.dt.year.between(1800, 2200)
        if ok.mean() >= 0.9:
            return "datetime"
    words = sample.str.split().str.len()
    if words.mean() >= 4 and nonnull.nunique() > 0.5 * len(nonnull):
        return "text"
    return "categorical"


def _value_key(s: pd.Series):
    v = pd.to_numeric(s, errors="coerce") if ptypes.is_numeric_dtype(s) else s.astype(str).where(s.notna(), "\x00")
    h = pd.util.hash_pandas_object(v.reset_index(drop=True), index=False).values
    return h.tobytes()


def is_id_like(name: str, s: pd.Series, kind: str) -> bool:
    n = len(s)
    nonnull = s.dropna()
    if n < 20 or len(nonnull) == 0:
        return False
    unique_ratio = nonnull.nunique() / len(nonnull)
    if ID_NAME.search(name) and unique_ratio >= 0.9:
        return True
    if kind == "categorical" and unique_ratio >= 0.99 and n >= 50:
        return True  # every row has its own code: carries no transferable signal
    if kind == "numeric" and unique_ratio == 1.0 and len(nonnull) == n:
        v = pd.to_numeric(nonnull, errors="coerce")
        if np.allclose(v, np.round(v)):
            d = np.diff(v.values.astype(float))
            if len(d) and (np.all(d == 1) or np.all(d == -1)):
                return True  # a row counter
    return False


def analyze_columns(X: pd.DataFrame, X_test: pd.DataFrame | None, declared: dict,
                    id_column: str | None = None) -> dict:
    """Return {column: {"type", "declared_type", "drop", "duplicate_of"}}."""
    info: dict = {}
    keys: dict = {}
    for c in X.columns:
        decl = canonical_type(declared.get(c))
        kind = decl if decl in ("numeric", "categorical", "datetime", "text") else infer_type(X[c])
        entry = {"type": kind, "declared_type": declared.get(c), "drop": None, "duplicate_of": None}
        s = X[c]
        if c == id_column or decl == "id":
            entry["drop"] = "identifier"
        elif s.nunique(dropna=False) <= 1:
            entry["drop"] = "constant"
        elif is_id_like(c, s, kind):
            entry["drop"] = "identifier"
        else:
            key = _value_key(s)
            if key in keys and X[keys[key]].astype(str).equals(s.astype(str)):
                entry["drop"] = "duplicate"
                entry["duplicate_of"] = keys[key]
            else:
                keys.setdefault(key, c)
        info[c] = entry
    return info


def to_numeric_frame(X: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Coerce columns typed numeric (e.g. numbers stored as text) to floats."""
    X = X.copy()
    for c, e in info.items():
        if c in X.columns and e["type"] == "numeric" and not ptypes.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        elif c in X.columns and e["type"] == "numeric" and ptypes.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)
    return X


class Preprocessor:
    MAX_TREE_LEVELS = 254       # HistGradientBoosting accepts category codes < 255
    MAX_ONEHOT_LEVELS = 30

    def __init__(self, info: dict):
        kept = {c: e for c, e in info.items() if e["drop"] is None}
        self.num_cols = [c for c, e in kept.items() if e["type"] == "numeric"]
        self.cat_cols = [c for c, e in kept.items() if e["type"] == "categorical"]
        self.dt_cols = [c for c, e in kept.items() if e["type"] == "datetime"]
        self.text_cols = [c for c, e in kept.items() if e["type"] == "text"]

    # ---- derived numeric block (numbers, date parts, text statistics) ----
    def _numeric_block(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cols, src = {}, {}
        for c in self.num_cols:
            cols[c] = pd.to_numeric(X[c], errors="coerce").astype(float).values
            src[c] = c
        if fit:
            self.dt_parts = {}
        for c in self.dt_cols:
            d = parse_datetime(X[c])
            parts = {
                "year": d.dt.year, "month": d.dt.month, "day": d.dt.day, "dayofweek": d.dt.dayofweek,
                "hour": d.dt.hour, "minute": d.dt.minute, "dayofyear": d.dt.dayofyear,
            }
            ref = pd.Timestamp("2000-01-01", tz=d.dt.tz) if getattr(d.dt, "tz", None) else pd.Timestamp("2000-01-01")
            parts["elapsed_days"] = (d - ref).dt.total_seconds() / 86400.0
            if fit:
                self.dt_parts[c] = [p for p, v in parts.items() if pd.Series(v).nunique() > 1]
            for p in self.dt_parts[c]:
                cols[f"{c}__{p}"] = pd.Series(parts[p]).astype(float).values
                src[f"{c}__{p}"] = c
        for c in self.text_cols:
            t = X[c].astype(str).where(X[c].notna(), "")
            cols[f"{c}__len"] = t.str.len().astype(float).values
            cols[f"{c}__words"] = t.str.split().str.len().astype(float).values
            src[f"{c}__len"] = src[f"{c}__words"] = c
        if fit:
            self.num_src = src
        return pd.DataFrame(cols, index=X.index)

    def _codes(self, X: pd.DataFrame, c: str) -> np.ndarray:
        s = X[c]
        lookup = self.levels[c]
        mapped = s.astype(str).map(lookup).where(s.notna())
        if len(lookup) >= self.MAX_TREE_LEVELS:  # rare levels of a high-cardinality column
            mapped = mapped.where(~(s.notna() & mapped.isna()), self.MAX_TREE_LEVELS)
        return mapped.astype(float).values

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        num = self._numeric_block(X, fit=True)
        self.num_feats = list(num.columns)
        arr = num.values.astype(float) if len(self.num_feats) else np.zeros((len(X), 0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.medians = np.nan_to_num(np.nanmedian(arr, axis=0)) if arr.shape[1] else np.zeros(0)
            self.missing_flag = np.isnan(arr).any(axis=0) if arr.shape[1] else np.zeros(0, bool)
            lo = np.nanmin(arr, axis=0) if arr.shape[1] else np.zeros(0)
            filled = np.where(np.isnan(arr), self.medians, arr)
            skew = pd.DataFrame(filled).skew().fillna(0).values if arr.shape[1] else np.zeros(0)
            self.log_flag = (np.nan_to_num(lo) >= 0) & (np.abs(skew) > 2.0)
            t = np.where(self.log_flag, np.log1p(np.clip(filled, 0, None)), filled)
            self.mu = t.mean(axis=0) if arr.shape[1] else np.zeros(0)
            sd = t.std(axis=0) if arr.shape[1] else np.zeros(0)
            self.sd = np.where(sd > 1e-12, sd, 1.0)
            self.tree_fill = np.where(np.isfinite(lo), lo - 1.0 - np.abs(lo) * 0.01, -1.0) if arr.shape[1] else np.zeros(0)

        self.levels, self.onehot, self.freq = {}, {}, {}
        for c in self.cat_cols:
            vc = X[c].dropna().astype(str).value_counts()
            self.levels[c] = {v: i for i, v in enumerate(vc.index[: self.MAX_TREE_LEVELS])}
            self.freq[c] = (vc / max(len(X), 1)).to_dict()
            keep = vc[(vc >= max(3, 0.005 * len(X)))].index[: self.MAX_ONEHOT_LEVELS]
            self.onehot[c] = {
                "levels": {v: i for i, v in enumerate(keep)},
                "other": len(keep) < len(vc),
                "missing": bool(X[c].isna().any()),
            }
        self.high_card = [c for c in self.cat_cols if len(self.levels[c]) >= self.MAX_TREE_LEVELS]
        self.freq_stats = {}
        for c in self.high_card:
            f = X[c].astype(str).map(self.freq[c]).fillna(0).astype(float).values
            self.freq_stats[c] = (f.mean(), f.std() + 1e-12)

        # bookkeeping: which raw column each output column comes from
        self.tree_src = [self.num_src[f] for f in self.num_feats] + list(self.cat_cols) + list(self.high_card)
        self.tree_names = list(self.num_feats) + list(self.cat_cols) + [f"{c}__freq" for c in self.high_card]
        self.cat_mask = np.array([False] * len(self.num_feats) + [True] * len(self.cat_cols)
                                 + [False] * len(self.high_card), dtype=bool)
        lin_src, lin_names = [], []
        for f, m in zip(self.num_feats, self.missing_flag):
            lin_src.append(self.num_src[f]); lin_names.append(f)
        for f, m in zip(self.num_feats, self.missing_flag):
            if m:
                lin_src.append(self.num_src[f]); lin_names.append(f"{f}__missing")
        for c in self.cat_cols:
            oh = self.onehot[c]
            labels = list(oh["levels"]) + (["__other"] if oh["other"] else []) + (["__missing"] if oh["missing"] else [])
            for lab in labels:
                lin_src.append(c); lin_names.append(f"{c}={lab}")
            if c in self.high_card:
                lin_src.append(c); lin_names.append(f"{c}__freq")
        self.lin_src, self.lin_names = lin_src, lin_names
        return self

    def tree_matrix(self, X: pd.DataFrame) -> np.ndarray:
        num = self._numeric_block(X, fit=False)
        blocks = [num[self.num_feats].values.astype(float)] if self.num_feats else []
        blocks += [self._codes(X, c)[:, None] for c in self.cat_cols]
        blocks += [X[c].astype(str).map(self.freq[c]).fillna(0).astype(float).values[:, None] for c in self.high_card]
        return np.hstack(blocks) if blocks else np.zeros((len(X), 0))

    def filled_tree_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Tree view without NaN, for estimators that cannot route missing values."""
        T = self.tree_matrix(X)
        fill = np.concatenate([self.tree_fill, -np.ones(T.shape[1] - len(self.tree_fill))])
        return np.where(np.isnan(T), fill, T)

    def linear_matrix(self, X: pd.DataFrame) -> np.ndarray:
        num = self._numeric_block(X, fit=False)
        blocks = []
        if self.num_feats:
            arr = num[self.num_feats].values.astype(float)
            miss = np.isnan(arr)
            arr = np.where(miss, self.medians, arr)
            arr = np.where(self.log_flag, np.log1p(np.clip(arr, 0, None)), arr)
            arr = np.clip((arr - self.mu) / self.sd, -6, 6)
            blocks.append(arr)
            if self.missing_flag.any():
                blocks.append(miss[:, self.missing_flag].astype(float))
        for c in self.cat_cols:
            oh = self.onehot[c]
            s = X[c]
            idx = s.astype(str).map(oh["levels"]).where(s.notna())
            width = len(oh["levels"]) + oh["other"] + oh["missing"]
            M = np.zeros((len(X), width))
            known = idx.notna().values
            M[np.where(known)[0], idx[known].astype(int).values] = 1.0
            if oh["other"]:
                M[(~known) & s.notna().values, len(oh["levels"])] = 1.0
            if oh["missing"]:
                M[s.isna().values, width - 1] = 1.0
            blocks.append(M)
            if c in self.high_card:
                f = s.astype(str).map(self.freq[c]).fillna(0).astype(float).values
                mu, sd = self.freq_stats[c]
                blocks.append(((f - mu) / sd)[:, None])
        return np.hstack(blocks) if blocks else np.zeros((len(X), 0))
