"""
Look at how the error grows as we forecast further ahead.

Persistence is great at one step because eGFR barely moves visit to visit. But
ask it for eGFR several visits out and it falls apart, because it assumes nothing
ever changes. This script measures that: persistence vs xgboost at horizons of
1, 3, 6 and 12 visits, on the same patient splits.

(The TFT multi-horizon run belongs on a GPU; this CPU version covers the two
baselines that make the point.)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

DROP = {"subject_id", "time_idx", "charttime", "egfr_next", "gender"}


def add_horizon_target(df, h):
    df = df.sort_values(["subject_id", "time_idx"]).copy()
    df[f"egfr_h{h}"] = df.groupby("subject_id")["egfr"].shift(-h)
    return df


def features(df):
    cols = [c for c in df.columns
            if c not in DROP and not c.startswith("egfr_h")]
    if "gender" in df.columns:
        df["is_female"] = (df["gender"] == "F").astype(int)
        cols.append("is_female")
    return cols


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def run(proc_dir, out_path, fig_path, horizons=(1, 3, 6, 12)):
    train = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    val = pd.read_csv(os.path.join(proc_dir, "val.csv"))
    test = pd.read_csv(os.path.join(proc_dir, "test.csv"))

    out = {"horizons": {}}
    for h in horizons:
        tr = add_horizon_target(train, h).dropna(subset=[f"egfr_h{h}"])
        va = add_horizon_target(val, h).dropna(subset=[f"egfr_h{h}"])
        te = add_horizon_target(test, h).dropna(subset=[f"egfr_h{h}"])
        cols = features(tr)
        for d in (va, te):
            if "is_female" not in d.columns:
                d["is_female"] = (d["gender"] == "F").astype(int)

        y = te[f"egfr_h{h}"].to_numpy()

        # persistence: predict the value stays at today's eGFR
        pers = te["egfr"].to_numpy()

        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method="hist", early_stopping_rounds=20,
            eval_metric="rmse", random_state=42, missing=np.nan,
        )
        model.fit(tr[cols], tr[f"egfr_h{h}"],
                  eval_set=[(va[cols], va[f"egfr_h{h}"])], verbose=False)
        xgb_pred = model.predict(te[cols])

        out["horizons"][str(h)] = {
            "n_test": int(len(te)),
            "persistence": {"rmse": round(rmse(y, pers), 4),
                            "r2": round(float(r2_score(y, pers)), 4)},
            "xgboost": {"rmse": round(rmse(y, xgb_pred), 4),
                        "r2": round(float(r2_score(y, xgb_pred)), 4)},
        }
        print(f"h={h}: persistence rmse {out['horizons'][str(h)]['persistence']['rmse']}, "
              f"xgboost rmse {out['horizons'][str(h)]['xgboost']['rmse']}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    hs = list(horizons)
    pr = [out["horizons"][str(h)]["persistence"]["rmse"] for h in hs]
    xr = [out["horizons"][str(h)]["xgboost"]["rmse"] for h in hs]
    plt.figure(figsize=(5, 4))
    plt.plot(hs, pr, "o-", label="persistence")
    plt.plot(hs, xr, "s-", label="xgboost")
    plt.xlabel("horizon (visits ahead)")
    plt.ylabel("test RMSE")
    plt.title("error grows with horizon")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=130)
    plt.close()
    print(f"saved {out_path} and {fig_path}")
    return out


if __name__ == "__main__":
    from src.paths import find_processed, find_models_dir, find_reports_dir

    proc = find_processed()
    models = find_models_dir()
    reports = find_reports_dir()
    p = argparse.ArgumentParser()
    p.add_argument("--proc", default=proc)
    p.add_argument("--out", default=os.path.join(models, "multihorizon_results.json"))
    p.add_argument("--fig", default=os.path.join(reports, "figures", "F6_error_vs_horizon.png"))
    args = p.parse_args()
    run(args.proc, args.out, args.fig)
