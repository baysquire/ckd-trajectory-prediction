"""
Survival analysis: how long until a patient's CKD stage gets worse?

The event is a stage transition (e.g. 3->4 or 4->5). Time is measured
from the patient's first visit. Patients who never progress are censored
at their last observation.

We fit Cox PH and a random survival forest, compare them with C-index,
and draw a Kaplan-Meier curve by starting stage.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_survival_data(proc_dir):
    """Turn the longitudinal rows into one row per patient with time + event."""
    frames = []
    for split in ("train", "val", "test"):
        d = pd.read_csv(os.path.join(proc_dir, f"{split}.csv"))
        d["split"] = split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])

    rows = []
    for sid, g in df.groupby("subject_id"):
        g = g.sort_values("charttime")
        t0 = g["charttime"].iloc[0]
        start_stage = g["ckd_stage"].iloc[0]

        # find first time ckd_stage goes above the starting stage
        progressed = g[g["ckd_stage"] > start_stage]
        if len(progressed) > 0:
            event_time = progressed["charttime"].iloc[0]
            days = (event_time - t0).total_seconds() / 86400.0
            event = True
        else:
            # censored at last visit
            days = (g["charttime"].iloc[-1] - t0).total_seconds() / 86400.0
            event = False

        # skip patients with zero follow-up
        if days <= 0:
            continue

        # baseline features from first visit
        first = g.iloc[0]
        rows.append({
            "subject_id": sid,
            "split": g["split"].iloc[0],
            "time_days": days,
            "event": int(event),
            "start_stage": int(start_stage),
            "anchor_age": float(first["anchor_age"]),
            "is_female": int(first["gender"] == "F"),
            "has_diabetes": int(first["has_diabetes"]),
            "has_hypertension": int(first["has_hypertension"]),
            "has_heart_failure": int(first["has_heart_failure"]),
            "baseline_egfr": float(first["egfr"]),
            "baseline_creatinine": float(first["creatinine_mg_dl"]),
            "baseline_potassium": float(first.get("potassium", np.nan)),
            "baseline_hemoglobin": float(first.get("hemoglobin", np.nan)),
            "baseline_albumin": float(first.get("albumin", np.nan)),
            "n_visits": len(g),
        })

    return pd.DataFrame(rows)


def run(proc_dir, out_path, fig_path, table_path):
    surv_df = build_survival_data(proc_dir)
    print(f"survival cohort: {len(surv_df)} patients, "
          f"event rate {surv_df['event'].mean():.3f}")

    tr = surv_df[surv_df["split"] == "train"]
    te = surv_df[surv_df["split"] == "test"]

    feat_cols = [
        "anchor_age", "is_female", "has_diabetes", "has_hypertension",
        "has_heart_failure", "baseline_egfr", "baseline_creatinine",
        "baseline_potassium", "baseline_hemoglobin", "baseline_albumin",
        "n_visits", "start_stage",
    ]

    # sksurv can't handle NaN so fill with train medians.
    # better than filling with 0 (potassium=0 is dead, not missing).
    train_medians = tr[feat_cols].median()
    Xtr = tr[feat_cols].fillna(train_medians).to_numpy()
    Xte = te[feat_cols].fillna(train_medians).to_numpy()

    # scikit-survival needs a structured array for y
    from sksurv.util import Surv
    ytr = Surv.from_arrays(tr["event"].astype(bool), tr["time_days"])
    yte = Surv.from_arrays(te["event"].astype(bool), te["time_days"])

    results = {
        "n_train": len(tr),
        "n_test": len(te),
        "event_rate": round(float(te["event"].mean()), 4),
    }

    # cox proportional hazards
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    cox = CoxPHSurvivalAnalysis(alpha=0.1)
    cox.fit(Xtr, ytr)
    cox_c = cox.score(Xte, yte)
    results["cox_c_index"] = round(float(cox_c), 4)
    print(f"cox c-index: {cox_c:.4f}")

    # random survival forest
    from sksurv.ensemble import RandomSurvivalForest
    rsf = RandomSurvivalForest(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=1,
    )
    rsf.fit(Xtr, ytr)
    rsf_c = rsf.score(Xte, yte)
    results["rsf_c_index"] = round(float(rsf_c), 4)
    print(f"rsf c-index: {rsf_c:.4f}")

    # save results
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")

    # kaplan-meier figure by starting stage
    _km_figure(surv_df[surv_df["split"] == "test"], fig_path)

    # markdown table
    _write_table(results, table_path)

    return results


def _km_figure(te, fig_path):
    """KM curves split by starting CKD stage."""
    from sksurv.nonparametric import kaplan_meier_estimator

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.figure(figsize=(6, 4))

    for stage in sorted(te["start_stage"].unique()):
        sub = te[te["start_stage"] == stage]
        if len(sub) < 20:
            continue
        try:
            time, prob = kaplan_meier_estimator(
                sub["event"].astype(bool).to_numpy(),
                sub["time_days"].to_numpy(),
            )
            plt.step(time / 365.25, prob, where="post",
                     label=f"stage {stage} (n={len(sub)})")
        except Exception:
            continue

    plt.xlabel("years from first visit")
    plt.ylabel("probability of no stage progression")
    plt.title("kaplan-meier: time to CKD stage transition")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130)
    plt.close()
    print(f"saved {fig_path}")


def _write_table(results, table_path):
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    lines = [
        "# Survival Analysis: Time to CKD Stage Transition",
        "",
        f"Test patients: {results['n_test']}, event rate: {results['event_rate']}",
        "",
        "| Model | C-index |",
        "|:------|--------:|",
        f"| Cox PH | {results['cox_c_index']} |",
        f"| Random Survival Forest | {results['rsf_c_index']} |",
        "",
        "Event = first CKD stage transition (e.g. 3->4). Patients who never",
        "progress are censored at last observation.",
    ]
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {table_path}")


if __name__ == "__main__":
    from src.paths import find_processed, find_models_dir, find_reports_dir

    proc = find_processed()
    models = find_models_dir()
    reports = find_reports_dir()
    p = argparse.ArgumentParser()
    p.add_argument("--proc", default=proc)
    p.add_argument("--out", default=os.path.join(models, "survival_results.json"))
    p.add_argument("--fig", default=os.path.join(reports, "figures", "F8_survival_km.png"))
    p.add_argument("--table", default=os.path.join(reports, "survival_table.md"))
    args = p.parse_args()
    run(args.proc, args.out, args.fig, args.table)
