"""Run main.py on every practice dataset in a fresh process and score it.

    python tools/benchmark.py [PRACTICE_FOLDER] [--only NAME_PART]

Each run is timed end to end (interpreter start-up included) like the grader's
60-second limit; scores use the hidden answers written by the generator.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score, f1_score,
                             mean_squared_error, normalized_mutual_info_score, r2_score)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.join(REPO, "practice"))
    ap.add_argument("--only", help="run datasets whose name contains this text")
    a = ap.parse_args()
    root, only = a.root, a.only
    data, answers = os.path.join(root, "datasets"), os.path.join(root, "answers")
    out_root = os.path.join(root, "outputs")
    rows = []
    for name in sorted(os.listdir(data)):
        if only and only not in name:
            continue
        out = os.path.join(out_root, name)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, os.path.join(REPO, "main.py"), os.path.join(data, name), out, "--quiet"],
                              capture_output=True, text=True)
        secs = time.perf_counter() - t0
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
        prof = json.load(open(os.path.join(out, "profile.json")))
        pred = pd.read_csv(os.path.join(out, "predictions.csv")).iloc[:, -1].values
        truth = pd.read_csv(os.path.join(answers, f"{name}.csv")).iloc[:, 0].values
        task = prof["task"]
        row = {"dataset": name, "task": task, "seconds": round(secs, 1), "status": prof.get("status", "?")[:30],
               "model": ",".join(prof.get("model", {}).get("ensemble", {}) or [prof.get("model", {}).get("selected_model", "")])}
        if task == "classification":
            truth_s, pred_s = truth.astype(str), pred.astype(str)
            row.update(accuracy=accuracy_score(truth_s, pred_s), macro_f1=f1_score(truth_s, pred_s, average="macro"),
                       balanced_acc=balanced_accuracy_score(truth_s, pred_s))
        elif task == "regression":
            row.update(rmse=np.sqrt(mean_squared_error(truth, pred)), r2=r2_score(truth, pred))
            if truth.min() >= 0:
                row["rmsle"] = np.sqrt(mean_squared_error(np.log1p(truth), np.log1p(np.clip(pred, 0, None))))
        else:
            row.update(ari=adjusted_rand_score(truth, pred), nmi=normalized_mutual_info_score(truth, pred))
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print()
    print(df.round(4).to_string(index=False))
    df.round(5).to_csv(os.path.join(root, "benchmark results.csv"), index=False)


if __name__ == "__main__":
    main()
