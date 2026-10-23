"""Build practice datasets that mimic the hackathon's scope.

The organisers' nine practice datasets are released at kickoff; until then these
stand-ins exercise every variation the brief lists: missing values, differently
scaled features, categorical columns, constant/duplicate columns, irrelevant
features, imbalanced classes and nonlinear relationships. Hidden answers go to
a separate folder so the benchmark can score the program like the grader would.

    python "tools/make practice datasets.py" [OUTPUT_FOLDER]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn import datasets as skd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "practice"
DATA = os.path.join(ROOT, "datasets")
ANSWERS = os.path.join(ROOT, "answers")


def save_supervised(name, df, target, meta, test_frac=0.3, seed=0, id_col=None):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_frac))
    train, test = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    folder = os.path.join(DATA, name)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(ANSWERS, exist_ok=True)
    train.to_csv(os.path.join(folder, "train.csv"), index=False)
    test.drop(columns=[target]).to_csv(os.path.join(folder, "test.csv"), index=False)
    test[[target]].to_csv(os.path.join(ANSWERS, f"{name}.csv"), index=False)
    with open(os.path.join(folder, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def save_clustering(name, df, labels, meta):
    folder = os.path.join(DATA, name)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(ANSWERS, exist_ok=True)
    df.to_csv(os.path.join(folder, "data.csv"), index=False)
    pd.DataFrame({"label": labels}).to_csv(os.path.join(ANSWERS, f"{name}.csv"), index=False)
    with open(os.path.join(folder, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def holes(df, cols, frac, rng):
    for c in cols:
        df.loc[rng.rand(len(df)) < frac, c] = np.nan
    return df


# ----------------------------------------------------------------------------- classification

def churn(seed=1, n=5000):
    rng = np.random.RandomState(seed)
    tenure = rng.randint(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .21, .24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.34, .44, .22])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n)
    senior = rng.binomial(1, .16, n)
    monthly = np.round(20 + 50 * (internet == "Fiber optic") + 25 * (internet == "DSL") + rng.gamma(2, 8, n), 2)
    total = np.round(monthly * tenure + rng.normal(0, 20, n).clip(0), 2)
    logit = (-1.2 + 1.3 * (contract == "Month-to-month") - 0.9 * (contract == "Two year") - 0.045 * tenure
             + 0.9 * (internet == "Fiber optic") + 0.6 * (payment == "Electronic check") + 0.35 * senior
             + 0.012 * (monthly - 65))
    churned = rng.rand(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "customerID": [f"{rng.randint(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 5))}" for _ in range(n)],
        "gender": rng.choice(["Male", "Female"], n), "SeniorCitizen": senior,
        "Partner": rng.choice(["Yes", "No"], n), "tenure": tenure, "Contract": contract,
        "InternetService": internet, "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": [" " if t == 0 else f"{v}" for t, v in zip(tenure, total)],  # numbers stored as text
        "Country": "US",                      # constant column
        "MonthlyChargesCopy": monthly,        # duplicate column
        "noise_1": rng.normal(0, 1, n), "noise_2": rng.uniform(0, 1000, n),
        "Churn": np.where(churned, "Yes", "No"),
    })
    df = holes(df, ["Partner", "InternetService"], 0.04, rng)
    save_supervised("telco churn", df, "Churn",
                    {"task": "classification", "target": "Churn", "id_column": "customerID",
                     "feature_types": {"gender": "categorical", "SeniorCitizen": "categorical", "Partner": "categorical",
                                       "tenure": "numeric", "Contract": "categorical", "InternetService": "categorical",
                                       "PaymentMethod": "categorical", "MonthlyCharges": "numeric",
                                       "TotalCharges": "numeric", "Country": "categorical",
                                       "MonthlyChargesCopy": "numeric", "noise_1": "numeric", "noise_2": "numeric"}})


def maintenance(seed=2, n=8000):
    rng = np.random.RandomState(seed)
    kind = rng.choice(["L", "M", "H"], n, p=[.6, .3, .1])
    air = rng.normal(300, 2, n)
    proc = air + 10 + rng.normal(0, 1, n)
    rpm = rng.normal(1540, 180, n).clip(1170, 2890)
    torque = rng.normal(40, 10, n).clip(3, 77)
    wear = rng.randint(0, 254, n)
    power = torque * rpm * 2 * np.pi / 60
    fail = ((wear > 200 + 10 * (kind == "H")) & (rng.rand(n) < .5)) | ((proc - air < 8.6) & (rpm < 1380)) | \
           (power < 3500) | (power > 9000) | (torque * wear > 11000 + 1000 * (kind == "M")) | (rng.rand(n) < .001)
    df = pd.DataFrame({"UDI": np.arange(1, n + 1), "Product ID": [f"{k}{rng.randint(10000, 99999)}" for k in kind],
                       "Type": kind, "Air temperature [K]": air, "Process temperature [K]": proc,
                       "Rotational speed [rpm]": rpm.round(), "Torque [Nm]": torque, "Tool wear [min]": wear,
                       "sensor_offset": 0.0, "Machine failure": fail.astype(int)})
    df = holes(df, ["Torque [Nm]", "Air temperature [K]"], 0.03, rng)
    save_supervised("predictive maintenance", df, "Machine failure",
                    {"task_type": "binary classification", "target_column": "Machine failure",
                     "numeric_features": ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                                          "Torque [Nm]", "Tool wear [min]", "sensor_offset"],
                     "categorical_features": ["Type", "Product ID"], "id_columns": ["UDI"]})


def nonlinear_multiclass(seed=3, n=3000):
    rng = np.random.RandomState(seed)
    X, y = skd.make_classification(n, n_features=8, n_informative=5, n_redundant=1, n_classes=4,
                                   n_clusters_per_class=2, weights=[.5, .25, .15, .1], class_sep=1.0, random_state=seed)
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = np.where(r > 2.2, 3, y)                      # ring-shaped class: nonlinear
    scales = 10.0 ** rng.randint(-3, 5, X.shape[1])  # wildly different scales
    df = pd.DataFrame(X * scales, columns=[f"f{i}" for i in range(X.shape[1])])
    for i in range(6):
        df[f"irrelevant_{i}"] = rng.normal(0, 1, n)
    df["colour"] = rng.choice(["red", "green", "blue"], n)
    df["f0_dup"] = df["f0"]
    df = holes(df, ["f1", "f3", "colour"], 0.08, rng)
    df["species"] = np.array(["alpha", "beta", "gamma", "delta"])[y]
    save_supervised("synthetic multiclass", df, "species", {"problem_type": "multiclass_classification",
                                                            "label": "species", "metric": "macro_f1"})


# ----------------------------------------------------------------------------- regression

def houses(seed=4, n=2400):
    rng = np.random.RandomState(seed)
    hood = rng.choice([f"N{i:02d}" for i in range(25)], n)
    hood_val = dict(zip([f"N{i:02d}" for i in range(25)], rng.lognormal(0, .3, 25)))
    area = rng.lognormal(7.3, .35, n)
    lot = rng.lognormal(9.1, .5, n)
    qual = rng.randint(1, 11, n)
    year = rng.randint(1880, 2011, n)
    garage = rng.choice([0, 1, 2, 3], n, p=[.06, .27, .56, .11])
    cond = rng.choice(["Po", "Fa", "TA", "Gd", "Ex"], n, p=[.02, .08, .6, .25, .05])
    price = (np.exp(10.2 + 0.55 * np.log(area) * 1 + 0.09 * qual + 0.003 * (year - 1950) + 0.05 * garage
                    + 0.04 * np.log(lot)) * np.array([hood_val[h] for h in hood]) * rng.lognormal(0, .1, n))
    df = pd.DataFrame({"Id": np.arange(1, n + 1), "MSZoning": rng.choice(["RL", "RM", "FV", "RH"], n),
                       "Neighborhood": hood, "GrLivArea": area.round(), "LotArea": lot.round(),
                       "OverallQual": qual, "YearBuilt": year, "GarageCars": garage, "KitchenQual": cond,
                       "PoolQC": np.where(rng.rand(n) < .995, None, "Gd"), "Utilities": "AllPub",
                       "MiscVal": rng.choice([0, 0, 0, 400, 700], n), "SalePrice": price.round()})
    df = holes(df, ["LotArea", "GarageCars", "KitchenQual"], 0.05, rng)
    save_supervised("house prices", df, "SalePrice",
                    {"task": "regression", "target": "SalePrice", "id": "Id", "metric": "rmse"})


def bikes(seed=5, n=6000):
    rng = np.random.RandomState(seed)
    t = pd.Timestamp("2011-01-01") + pd.to_timedelta(np.sort(rng.choice(2 * 365 * 24, n, replace=False)), unit="h")
    hour, dow, month = t.hour, t.dayofweek, t.month
    temp = 15 + 10 * np.sin((month - 4) / 12 * 2 * np.pi) + rng.normal(0, 4, n)
    hum = rng.uniform(20, 100, n)
    wind = rng.gamma(2, 6, n)
    weather = rng.choice([1, 2, 3, 4], n, p=[.65, .26, .085, .005])
    workday = (dow < 5).astype(int)
    peak = workday * (np.exp(-((hour - 8) ** 2) / 2) * 3 + np.exp(-((hour - 17.5) ** 2) / 3) * 3.5) + \
        (1 - workday) * np.exp(-((hour - 14) ** 2) / 12) * 2
    rate = np.exp(1.5 + 1.3 * peak + 0.04 * temp - 0.006 * hum - 0.35 * (weather - 1) + 0.3 * (t.year - 2011)
                  + 0.8 * (hour >= 7) * (hour <= 21))
    count = rng.poisson(rate)
    df = pd.DataFrame({"datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "season": (month % 12) // 3 + 1,
                       "holiday": rng.binomial(1, .03, n), "workingday": workday, "weather": weather,
                       "temp": temp.round(2), "atemp": (temp + rng.normal(0, 1, n)).round(2),
                       "humidity": hum.round(), "windspeed": wind.round(3), "count": count})
    save_supervised("bike sharing", df, "count",
                    {"task": "regression", "target": "count", "evaluation_metric": "RMSLE",
                     "feature_types": {"datetime": "datetime", "season": "categorical", "holiday": "categorical",
                                       "workingday": "categorical", "weather": "categorical", "temp": "numeric",
                                       "atemp": "numeric", "humidity": "numeric", "windspeed": "numeric"}})


def friedman(seed=6, n=4000):
    rng = np.random.RandomState(seed)
    X, y = skd.make_friedman1(n, n_features=12, noise=1.0, random_state=seed)  # 7 irrelevant features
    df = pd.DataFrame(X * 10.0 ** rng.randint(-2, 4, 12), columns=[f"x{i}" for i in range(12)])
    df["grade"] = pd.cut(X[:, 3], 4, labels=["D", "C", "B", "A"]).astype(str)
    df["const"] = 7
    df = holes(df, ["x0", "x5", "grade"], 0.06, rng)
    df["y"] = y
    save_supervised("synthetic nonlinear regression", df, "y", {"task": "regression", "target": "y"})


# ----------------------------------------------------------------------------- clustering

def mall(seed=7, n=600):
    rng = np.random.RandomState(seed)
    centers = np.array([[25, 20], [25, 80], [55, 50], [88, 18], [88, 82]])
    lab = rng.choice(5, n, p=[.12, .12, .4, .18, .18])
    inc = centers[lab, 0] + rng.normal(0, 7, n)
    spend = centers[lab, 1] + rng.normal(0, 7, n)
    df = pd.DataFrame({"CustomerID": np.arange(1, n + 1), "Gender": rng.choice(["Male", "Female"], n),
                       "Age": rng.randint(18, 70, n), "Annual Income (k$)": inc.round(),
                       "Spending Score (1-100)": spend.clip(1, 100).round()})
    save_clustering("mall customers", df, lab, {"task": "clustering", "n_clusters": 5})


def blobs_noise(seed=8, n=3000):
    rng = np.random.RandomState(seed)
    X, lab = skd.make_blobs(n, n_features=5, centers=4, cluster_std=[1.0, 1.5, 0.7, 1.2], random_state=seed)
    df = pd.DataFrame(X * np.array([1, 100, 0.01, 10, 1000]), columns=[f"m{i}" for i in range(5)])
    for i in range(3):
        df[f"noise_{i}"] = rng.normal(0, 1, n)
    df["batch"] = "B1"
    df = holes(df, ["m1", "m3"], 0.05, rng)
    save_clustering("blobs with noise", df, lab, {"task_type": "clustering", "num_groups": 4,
                                                  "feature_types": {"batch": "categorical"}})


def elongated(seed=9, n=1500):
    rng = np.random.RandomState(seed)
    X, lab = skd.make_blobs(n, n_features=2, centers=3, random_state=170)
    X = X @ np.array([[0.6, -0.64], [-0.41, 0.85]])  # anisotropic clusters: k-means struggles
    df = pd.DataFrame(X, columns=["a", "b"])
    df["c"] = rng.normal(0, 0.2, n)
    save_clustering("anisotropic clusters", df, lab, {"task": "clustering", "k": 3})


# ----------------------------------------------------------------------------- real data shipped with sklearn

def real():
    bc = skd.load_breast_cancer(as_frame=True).frame
    save_supervised("real breast cancer", bc, "target", {"task": "classification", "target": "target"})
    wine = skd.load_wine(as_frame=True).frame
    save_supervised("real wine classes", wine, "target", {"task": "classification", "target": "target"})
    dia = skd.load_diabetes(as_frame=True, scaled=False).frame
    save_supervised("real diabetes", dia, "target", {"task": "regression", "target": "target"})
    iris = skd.load_iris(as_frame=True).frame
    save_clustering("real iris", iris.drop(columns=["target"]), iris["target"].values,
                    {"task": "clustering", "n_clusters": 3})
    w = skd.load_wine(as_frame=True).frame
    save_clustering("real wine", w.drop(columns=["target"]), w["target"].values,
                    {"task": "clustering", "n_clusters": 3})


if __name__ == "__main__":
    for fn in (churn, maintenance, nonlinear_multiclass, houses, bikes, friedman, mall, blobs_noise, elongated, real):
        fn()
    print(f"datasets written to {DATA}, answers to {ANSWERS}")
