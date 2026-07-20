"""
Put every 1-step eGFR model on the same 14k test split so we can compare fairly.

The TFT number was saved on its own, with no baseline next to it. This script
fills that gap: persistence, mean, median, and xgboost, all on the exact same
test.csv the TFT used, plus bootstrap confidence intervals and a per-stage
breakdown. Persistence is the number to beat here.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "egfr_next"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred):
    return {
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def boot_ci(y_true, y_pred, fn, n=1000, seed=42):
    # simple bootstrap over rows so we can say how shaky a number is
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        vals.append(fn(y_true[pick], y_pred[pick]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def per_stage(df, y_true, y_pred):
    out = {}
    if "ckd_stage" not in df.columns:
        return out
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for stage in sorted(df["ckd_stage"].unique()):
        mask = (df["ckd_stage"] == stage).to_numpy()
        if mask.sum() > 10:
            out[f"stage_{int(stage)}"] = round(rmse(y_true[mask], y_pred[mask]), 4)
    return out


def xgb_features(train_df):
    drop = ["subject_id", "time_idx", "charttime", TARGET, "gender"]
    cols = [c for c in train_df.columns if c not in drop]
    return cols


def fit_xgb(train_df, val_df, feature_cols):
    for d in (train_df, val_df):
        if "gender" in d.columns:
            d["is_female"] = (d["gender"] == "F").astype(int)
    cols = feature_cols + (["is_female"] if "is_female" in train_df.columns else [])
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        early_stopping_rounds=20,
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(
        train_df[cols].fillna(0), train_df[TARGET],
        eval_set=[(val_df[cols].fillna(0), val_df[TARGET])],
        verbose=False,
    )
    return model, cols


def run(train_path, val_path, test_path, tft_results, out_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    y = test_df[TARGET].to_numpy()
    n_test = len(test_df)
    n_patients = int(test_df["subject_id"].nunique())
    print(f"test set: {n_test} rows, {n_patients} patients")

    models = {}

    # mean and median predictors: the dumb floor
    mean_val = float(train_df[TARGET].mean())
    median_val = float(train_df[TARGET].median())
    models["mean"] = metrics(y, np.full(n_test, mean_val))
    models["median"] = metrics(y, np.full(n_test, median_val))

    # persistence: next eGFR = current eGFR. hard to beat on a 1-step task.
    pers = test_df["egfr"].to_numpy()
    models["persistence"] = metrics(y, pers)
    models["persistence"]["rmse_ci"] = boot_ci(y, pers, rmse)
    models["persistence"]["r2_ci"] = boot_ci(y, pers, lambda a, b: r2_score(a, b))

    # xgboost on the engineered features
    feats = xgb_features(train_df)
    model, used_cols = fit_xgb(train_df, val_df, feats)
    if "gender" in test_df.columns:
        test_df["is_female"] = (test_df["gender"] == "F").astype(int)
    xgb_pred = model.predict(test_df[used_cols].fillna(0))
    models["xgboost"] = metrics(y, xgb_pred)
    models["xgboost"]["rmse_ci"] = boot_ci(y, xgb_pred, rmse)
    models["xgboost"]["r2_ci"] = boot_ci(y, xgb_pred, lambda a, b: r2_score(a, b))
    models["xgboost"]["per_stage"] = per_stage(test_df, y, xgb_pred)

    # tft: read what was already saved. note it was scored on its own windowed
    # samples, so it isn't on the exact same rows. flag that so nobody over-reads
    # a tiny gap.
    if tft_results and os.path.exists(tft_results):
        with open(tft_results) as f:
            t = json.load(f)
        models["tft"] = {
            "rmse": round(float(t["rmse"]), 4),
            "mae": round(float(t.get("mae", float("nan"))), 4),
            "r2": round(float(t["r2"]), 4),
            "source": os.path.basename(tft_results),
            "note": "scored on tft windowed samples, not the identical row set",
        }

    ranking = sorted(
        [(m, v["rmse"]) for m, v in models.items()], key=lambda x: x[1]
    )
    pers_rmse = models["persistence"]["rmse"]
    beats = {m: bool(v["rmse"] < pers_rmse) for m, v in models.items() if m != "persistence"}
    headline = min(
        ((m, v["rmse"]) for m, v in models.items() if m not in ("mean", "median")),
        key=lambda x: x[1],
    )[0]

    results = {
        "cohort": {
            "n_patients": n_patients,
            "n_test_rows": n_test,
            "target": TARGET,
            "note": "persistence/mean/median/xgboost on the identical test.csv; tft from its own eval",
        },
        "models": models,
        "ranking_by_rmse": [m for m, _ in ranking],
        "headline_model": headline,
        "beats_persistence": beats,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")

    from src.paths import find_reports_dir
    _write_table(results, find_reports_dir())
    return results


def _write_table(results, reports_dir):
    reports_dir = os.path.abspath(reports_dir)
    os.makedirs(reports_dir, exist_ok=True)
    m = results["models"]
    lines = [
        "# 14k cohort: 1-step eGFR baselines",
        "",
        f"Test set: {results['cohort']['n_test_rows']} rows, "
        f"{results['cohort']['n_patients']} patients. Target: next-visit eGFR.",
        "",
        "| Model | RMSE | MAE | R2 |",
        "|:------|-----:|----:|---:|",
    ]
    for name in results["ranking_by_rmse"]:
        v = m[name]
        lines.append(f"| {name} | {v['rmse']} | {v.get('mae','-')} | {v.get('r2','-')} |")
    lines += [
        "",
        f"Headline: **{results['headline_model']}**. "
        "Persistence is the bar to beat on this task.",
        "",
        "Reading it plainly: next-visit eGFR is close to current eGFR, so a copy-last "
        "rule is already strong. This table is here for honesty, not to claim a win.",
    ]
    path = os.path.join(reports_dir, "results_14k_table.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {path}")


if __name__ == "__main__":
    from src.paths import find_processed, find_models_dir

    proc = find_processed()
    models = find_models_dir()
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=os.path.join(proc, "train.csv"))
    p.add_argument("--val", default=os.path.join(proc, "val.csv"))
    p.add_argument("--test", default=os.path.join(proc, "test.csv"))
    p.add_argument("--tft-results", default=os.path.join(models, "tft_results.json"))
    p.add_argument("--out", default=os.path.join(models, "results_14k.json"))
    args = p.parse_args()

    run(args.train, args.val, args.test, args.tft_results, args.out)
