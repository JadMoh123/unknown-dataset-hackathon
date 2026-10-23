"""Score a deliberately plain baseline on the practice datasets and compare it with
"benchmark results.csv" (run tools/benchmark.py first).

Baseline: default HistGradientBoosting on ordinal-encoded columns for supervised
tasks; k-means on standardised numeric columns for clustering.

    python "tools/compare with baseline.py" [PRACTICE_FOLDER]
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, r2_score, adjusted_rand_score
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "practice"); D, A = os.path.join(root, "datasets"), os.path.join(root, "answers")
res = pd.read_csv(os.path.join(root, "benchmark results.csv"))
out = []
for name in sorted(os.listdir(D)):
    meta = json.load(open(os.path.join(D, name, "metadata.json")))
    truth = pd.read_csv(os.path.join(A, name + ".csv")).iloc[:, 0].values
    r = res[res.dataset == name].iloc[0]
    if os.path.exists(os.path.join(D, name, "data.csv")):
        X = pd.read_csv(os.path.join(D, name, "data.csv")).select_dtypes("number")
        X = X.fillna(X.median())
        k = len(np.unique(truth))
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(X))
        out.append((name, "ARI", adjusted_rand_score(truth, lab), r.ari)); continue
    tr = pd.read_csv(os.path.join(D, name, "train.csv")); te = pd.read_csv(os.path.join(D, name, "test.csv"))
    tgt = [c for c in tr.columns if c not in te.columns][0]
    y = tr.pop(tgt)
    for c in tr.columns:
        if not pd.api.types.is_numeric_dtype(tr[c]):
            cats = {v: i for i, v in enumerate(tr[c].astype(str).unique())}
            tr[c] = tr[c].astype(str).map(cats); te[c] = te[c].astype(str).map(cats)
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        p = HistGradientBoostingRegressor(random_state=0).fit(tr, y).predict(te)
        out.append((name, "R2", r2_score(truth, p), r.r2))
    else:
        p = HistGradientBoostingClassifier(random_state=0).fit(tr, y.astype(str)).predict(te)
        out.append((name, "macro-F1", f1_score(truth.astype(str), p, average="macro"), r.macro_f1))
O = pd.DataFrame(out, columns=["dataset", "metric", "baseline", "ours"]); O["gain"] = O.ours - O.baseline
print(O.round(4).to_string(index=False)); print(O.groupby("metric")[["baseline", "ours", "gain"]].mean().round(4))
O.round(4).to_csv(os.path.join(root, "baseline comparison.csv"), index=False)
