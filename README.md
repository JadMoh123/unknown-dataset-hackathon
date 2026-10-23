# Unknown Dataset Hackathon

**Course:** Introduction to Machine Learning
**Team:** Jad Moharam · Omar Tabsh · Mohannad Ghorayeb

One program that receives a table it has never seen, works out what it is looking at, and solves the
task that comes with it: **classification**, **regression** or **clustering**. It returns
predictions plus a structured data profile, within the 60-second limit, with no human in the loop.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py path/to/dataset_folder path/to/output_folder
```

Other ways to call the same entry point:

```bash
python main.py --input DATASET_DIR --output OUTPUT_DIR
python main.py --train train.csv --test test.csv --meta metadata.json --output OUTPUT_DIR
python main.py --input FOLDER_OF_DATASETS --output OUTPUT_ROOT      # runs each dataset in turn
python main.py DATASET_DIR OUTPUT_DIR --time-limit 60 --pred-column prediction
```

From Python: `from automl import solve; solve("DATASET_DIR", "OUTPUT_DIR")`.

### What it writes

| File | Content |
|---|---|
| `predictions.csv` | One row per row to predict, in input order: `<id column>, prediction`. The id column is the dataset's own id column when there is one, otherwise `id` = 0..n-1. Clustering writes the group number (0..k-1, largest group = 0). |
| `profile.json` | The structured data profile (see below) plus what the program decided and why. |
| `probabilities.csv` | Classification only: class probabilities per row. |

A valid `predictions.csv` (majority class / median / single cluster) is written **first**, before any
modelling, and then overwritten. A crash or a time-out can therefore never leave the grader with
no file.

---

## Approach

```
load folder ─► resolve task/target/k ─► clean & type columns ─► profile ─► encode two views ─► model ─► write
                                                                   │                             │
                                                     constant / duplicate / id      classification & regression:
                                                     missing / scale / imbalance    time-boxed CV search → ensemble → refit
                                                     relevance (permutation imp.)   clustering:
                                                                                    prune noise → pool → consensus pick
```

### 1. Loading (`automl/data.py`)
* Finds the tables and the metadata file (JSON, or YAML) in the dataset folder. It accepts
  `train/test` files, `X_train / y_train / X_test` splits, or a single `data.csv` for clustering. CSV delimiters
  are sniffed, and Parquet and Excel also work.
* Reads the task, target, id column, number of clusters, metric and feature types under many key
  spellings (`task`, `task_type`, `problem_type`; `target`, `label`, `target_column`; `n_clusters`, `k`,
  `num_groups`; `feature_types` as a map, a list of `{name, type}`, or `numeric_features` /
  `categorical_features` lists).
* Anything missing is inferred. The target is the column that is in train but not in test. Classification
  vs regression is decided from the target's values. With no target, the task is clustering, and if k is
  missing it is chosen by silhouette.

### 2. Cleaning and column analysis (`automl/prep.py`)
* Turns the usual missing spellings (`""`, `?`, `NA`, `null`, `-`, …) into real missing values, and
  converts numbers stored as text (e.g. Telco's `TotalCharges`).
* Types every column as numeric, categorical, datetime or free text. Declared types win.
  Datetimes become year / month / day / weekday / hour / elapsed-days features. Text becomes length
  and word count.
* Drops **constant** columns, **exact duplicate** columns and **identifier** columns (by name pattern
  plus near-unique values, or a running row counter).

### 3. Data profile (`automl/profiling.py`)
`profile.json` reports, for the dataset and for every column:
row and column counts; the type of every feature; missing values per column (train and test),
cells and rows with any missing; constant, near-constant, duplicate and identifier columns;
duplicate rows; strongly correlated pairs; scale spread (largest / smallest standard deviation);
numeric summaries (mean, std, quartiles, skew, outliers); top categories; the target's summary
(class counts, imbalance ratio, minority class, or skew for regression); **feature relevance**; and
whether the relationships look **nonlinear** (tree models beating linear ones in cross-validation).
It also records every preprocessing choice, every model tried with its CV score, the final
ensemble, and timings.

Feature relevance uses held-out **permutation importance** of a quick gradient-boosting model, with
mutual information reported alongside. We tried mutual information alone first, but its noise
floor was too high: pure-noise columns scored as high as weak real ones. Permutation importance
flags exactly the seven noise columns of the Friedman benchmark.

### 4. Two encodings of the same rows
* **Tree view:** raw numbers with NaN kept (gradient boosting routes missing values natively) and
  categoricals as integer codes passed as native categorical features.
* **Linear view:** median imputation plus missing-indicator columns, `log1p` for skewed
  non-negative features, standardisation (this handles differently scaled features), clipping
  of extreme values, and one-hot categoricals (frequent levels + "other" + "missing").

### 5. Classification and regression (`automl/supervised.py`)
* Candidates in priority order: HistGradientBoosting (plus a class-balanced version when classes are
  imbalanced); logistic / ridge regression; random forest; extra-trees; RBF SVM / SVR (up to 5k rows);
  k-NN. Together they cover linear, nonlinear, local and bagged behaviour.
* Each candidate gets **k-fold out-of-fold predictions**: 5 folds for small data, 3 for large.
  Classification folds are stratified. Selection runs on at most 25k rows, and the final model is
  refit on every row.
* A **feature-selection variant** of the best model (low-relevance features removed) competes too.
  It is kept only if cross-validation says it helps.
* **Greedy ensemble selection** (Caruana) on the out-of-fold predictions blends models only when
  the blend beats the best single model.
* The **decision rule is tuned**: a probability threshold for binary tasks, or class re-weighting
  for multiclass, chosen on out-of-fold predictions for the target metric. This is what handles
  imbalanced classes.
* Regression with a skewed non-negative target is also tried on a `log1p` scale. Predictions are
  clipped to a slightly widened training range.
* The metric comes from the metadata when given. Otherwise we use macro-F1 for classification and
  RMSE for regression (see *Assumptions*).

### 6. Clustering (`automl/clustering.py`)
1. **Prune noise features.** Columns that an initial k-means partition does not separate at all
   (variance explained < max(2 %, 10 % of the best column)) are dropped; this is repeated once.
2. **Pool of partitions:** k-means; Ward; and Gaussian mixtures with diagonal and full covariance,
   each both warm-started from k-means and best-of-five random starts. Mixtures fit on at most
   20k rows and then label every row.
3. **Consensus choice:** the partition that agrees most (mean adjusted Rand index) with the rest
   of the pool, with silhouette as tie-break.

We chose this rule by experiment on 19 labelled problems: 5 practice sets, 12 random synthetic sets
and 2 real ones. The consensus rule averaged **ARI 0.874**, silhouette alone 0.836, and the best
possible pick 0.914.

### 7. Time budget (`automl/budget.py`)
The clock starts when the interpreter starts. Modelling aims to finish by about 45 s, keeping 15 s
spare for slower grading CPUs. Before each fold the program projects the remaining cost and drops
candidates that would not fit, while keeping time in reserve for the final refit. If the refit of an
ensemble would not fit, the slowest members are removed first.

---

## Results on our practice datasets

The organisers' nine practice datasets are released at kickoff. Until then we built **14 stand-ins**
with `tools/make practice datasets.py`. They are modelled on the brief's Kaggle examples (Telco churn,
AI4I predictive maintenance, House Prices, Bike Sharing, Mall Customers) plus synthetic and real
sklearn datasets. Between them they contain every variation the brief lists: missing values,
differently scaled features, categoricals, constant and duplicate columns, irrelevant features,
imbalanced classes and nonlinear relationships. Scores are on a held-out test split
(`tools/benchmark.py`). Times are end-to-end process wall time on a 16-core laptop.

| Dataset | Task | Score | Plain baseline | Time |
|---|---|---|---|---|
| telco churn | classification | macro-F1 0.706 | 0.698 | 8.2 s |
| predictive maintenance (3 % failures) | classification | macro-F1 0.867 | 0.847 | 9.0 s |
| synthetic multiclass (ring class, noise cols) | classification | macro-F1 0.808 | 0.812 | 13.5 s |
| real breast cancer | classification | macro-F1 0.994 | 0.981 | 5.3 s |
| real wine classes | classification | macro-F1 1.000 | 0.945 | 6.2 s |
| house prices | regression | R² 0.929 | 0.869 | 7.1 s |
| bike sharing (datetime, RMSLE) | regression | R² 0.973 | −0.039 | 6.5 s |
| synthetic nonlinear (Friedman + noise) | regression | R² 0.925 | 0.913 | 6.9 s |
| real diabetes | regression | R² 0.509 | 0.397 | 5.6 s |
| mall customers | clustering | ARI 0.995 | 0.429 | 2.0 s |
| blobs with noise | clustering | ARI 1.000 | 0.997 | 2.2 s |
| anisotropic clusters | clustering | ARI 0.493 | 0.449 | 1.9 s |
| real iris | clustering | ARI 0.787 | 0.620 | 1.8 s |
| real wine | clustering | ARI 0.898 | 0.898 | 1.8 s |

The plain baseline (`tools/compare with baseline.py`) is default gradient boosting on
ordinal-encoded columns, or k-means on standardised numeric columns. Average gain:
**R² +0.30, ARI +0.16, macro-F1 +0.02.** The baseline's bike-sharing score shows why datetime
parsing matters. Mall shows why pruning noise features matters for clustering.

**Stress tests** (outside the repo): 100k-row classification and regression, a 60k-row clustering
table, a 300-column table, a 45-row table, a single-sample class, string labels, no metadata, `X/y`
file layout, YAML metadata with `;`-separated CSVs. All runs finished with status `ok`. With the
machine limited to **2 CPU cores**, the 100k-row tables took 31–34 s end to end, inside the
60 s limit with the full refit.

---

## Assumptions

The input/output contract, checker and scoring formulas are released at kickoff. We did not have
them while writing this, so we assumed the following. Each item is isolated in one place, so it
can be changed quickly once the contract is known.

1. **One dataset = one folder** holding the tables and a JSON/YAML metadata file
   (`automl/data.py`). Train/test are separate files, test has no target, and clustering gets a
   single table.
2. **Output** is `predictions.csv` with an id column and a `prediction` column, in test-row order.
   If the folder has a `sample_submission` file, its column names are used instead
   (`_write_predictions` in `automl/pipeline.py`). The prediction column name can also be set with
   `--pred-column`.
3. **Metric:** the metadata's metric is used when present. Otherwise we assume macro-F1 for
   classification (imbalance is explicitly in scope), RMSE for regression, and ARI for
   clustering. Supported names: accuracy, balanced accuracy, (macro / weighted / binary) F1, ROC-AUC,
   log-loss, RMSE, MAE, R², RMSLE.
4. **Clustering labels** are compared up to permutation (ARI/NMI-style), so group numbers only need to be
   consistent within a dataset.
5. **Time:** 60 s per dataset, one process per dataset, CPU only, wall-clock including Python
   start-up. `--time-limit` or the `TIME_LIMIT` environment variable changes it.
6. **Profile format:** the brief asks for a "structured data profile" without a schema, so
   `profile.json` reports everything it could reasonably check, under plain key names
   (`n_rows`, `feature_types`, `missing_values`, `constant_columns`, `duplicate_columns`,
   `identifier_columns`, `target_summary`, `low_relevance_features`, …). If the contract fixes a
   schema, `build_profile` in `automl/profiling.py` is the single place to map it.
7. Identifier-like columns (unique per row) carry no signal that transfers to new rows, so they
   are never used as features.
8. Rows whose target is missing are dropped from training.

---

## Reproduce

```bash
pip install -r requirements.txt
python "tools/make practice datasets.py"          # writes practice/datasets and practice/answers
python tools/benchmark.py                          # runs main.py per dataset, scores, times
python "tools/compare with baseline.py"            # plain-baseline comparison
```

Runs are deterministic: every random component uses a fixed seed.
Tested with Python 3.14 and the pinned versions in `requirements.txt`. The code avoids
version-specific APIs and should also run on scikit-learn ≥ 1.2 with pandas ≥ 2.0.

## Repository layout

```
main.py                    entry point (CLI)
automl/
  pipeline.py              end-to-end run, fallback output, writing files
  data.py                  dataset discovery, metadata parsing, task inference
  prep.py                  cleaning, type inference, column drops, tree/linear encodings
  profiling.py             profile.json, feature relevance
  supervised.py            classification/regression search, ensembling, decision tuning
  clustering.py            noise pruning, partition pool, consensus selection
  budget.py                time budget
tools/
  make practice datasets.py   practice data generator (with hidden answers)
  benchmark.py                end-to-end benchmark
  compare with baseline.py    baseline comparison
practice/                  generated practice data, answers and benchmark results
```

AI assistance was used while developing, as the rules allow. The final program runs
independently with only the pinned open-source libraries.
