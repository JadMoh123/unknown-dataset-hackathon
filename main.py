"""Entry point.

    python main.py DATASET_DIR [OUTPUT_DIR]
    python main.py --input DATASET_DIR --output OUTPUT_DIR
    python main.py --train train.csv --test test.csv --meta metadata.json --output OUTPUT_DIR
    python main.py --input FOLDER_OF_DATASETS --output OUTPUT_ROOT     (runs every dataset inside)

Writes OUTPUT_DIR/predictions.csv, OUTPUT_DIR/profile.json and, for
classification, OUTPUT_DIR/probabilities.csv.
"""
import time

_START = time.perf_counter()

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from automl import solve  # noqa: E402
from automl.data import is_dataset_folder  # noqa: E402
from automl.pipeline import DEFAULT_TIME_LIMIT  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Automatic ML for unknown tabular datasets.")
    ap.add_argument("dataset", nargs="?", help="dataset folder (or a folder of dataset folders)")
    ap.add_argument("output", nargs="?", help="output folder")
    ap.add_argument("--input", "--data", "-i", dest="input_dir")
    ap.add_argument("--output", "--out", "-o", dest="output_dir")
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--meta", "--metadata")
    ap.add_argument("--time-limit", type=float,
                    default=float(os.environ.get("TIME_LIMIT", DEFAULT_TIME_LIMIT)),
                    help="seconds allowed per dataset (default 60)")
    ap.add_argument("--pred-column", help="name of the prediction column in predictions.csv")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    data = a.input_dir or a.dataset
    out = a.output_dir or a.output
    if a.train:
        solve(None, out, a.time_limit, a.train, a.test, a.meta, a.pred_column, not a.quiet, start=_START)
        return 0
    if not data:
        ap.error("give a dataset folder or --train")

    if is_dataset_folder(data):
        solve(data, out, a.time_limit, meta_path=a.meta, pred_column=a.pred_column, verbose=not a.quiet,
              start=_START)
        return 0

    # A folder of dataset folders: run each with its own time budget.
    subs = [os.path.join(data, d) for d in sorted(os.listdir(data)) if os.path.isdir(os.path.join(data, d))
            and is_dataset_folder(os.path.join(data, d))]
    if not subs:
        ap.error(f"no data tables found under {data}")
    for d in subs:
        target = os.path.join(out or "outputs", os.path.basename(d))
        try:
            solve(d, target, a.time_limit, pred_column=a.pred_column, verbose=not a.quiet,
                  start=time.perf_counter())
        except Exception as exc:  # keep going with the other datasets
            print(f"[{os.path.basename(d)}] failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
