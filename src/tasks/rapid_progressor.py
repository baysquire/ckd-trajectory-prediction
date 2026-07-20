"""
Build the rapid-progressor task from the eGFR trajectories.

The idea a nephrologist actually cares about: is this patient losing kidney
function fast? We call it rapid if the eGFR slope over the follow-up year is
worse than -5 mL/min/1.73m2/year. Unlike next-visit eGFR, copy-last can't win
this, because the label is about a trend, not a single next value.

To keep it honest we split each patient's record in time:
  - baseline window: the first `baseline_days` of the record -> features only
  - outcome window:  the next `horizon_days` after that     -> label only
Nothing from the outcome window is allowed into the features.
"""
import argparse
import os

import numpy as np
import pandas as pd

BASELINE_LABS = [
    "creatinine_mg_dl", "potassium", "sodium", "bun",
    "hemoglobin", "albumin", "phosphorus",
]


def _ols_slope(times_years, values):
    # plain least-squares slope of egfr against time in years
    if len(values) < 2:
        return np.nan
    t = np.asarray(times_years, dtype=float)
    y = np.asarray(values, dtype=float)
    if np.ptp(t) == 0:
        return np.nan
    return float(np.polyfit(t, y, 1)[0])


def build_patient_row(g, baseline_days, horizon_days, min_points):
    g = g.sort_values("charttime")
    t0_start = g["charttime"].iloc[0]
    years = (g["charttime"] - t0_start).dt.total_seconds() / (365.25 * 24 * 3600)
    g = g.assign(_years=years.values)

    baseline_cut = baseline_days / 365.25
    horizon_cut = baseline_cut + horizon_days / 365.25

    base = g[g["_years"] <= baseline_cut]
    out = g[(g["_years"] > baseline_cut) & (g["_years"] <= horizon_cut)]

    if len(base) < 1 or len(out) < min_points:
        return None

    # label from the outcome window only
    slope = _ols_slope(out["_years"] - baseline_cut, out["egfr"])
    if np.isnan(slope):
        return None

    # features from the baseline window only
    row = {
        "subject_id": g["subject_id"].iloc[0],
        "anchor_age": float(base["anchor_age"].iloc[-1]),
        "is_female": int(base["gender"].iloc[-1] == "F"),
        "has_diabetes": int(base["has_diabetes"].iloc[-1]),
        "has_hypertension": int(base["has_hypertension"].iloc[-1]),
        "has_heart_failure": int(base["has_heart_failure"].iloc[-1]),
        "baseline_egfr": float(base["egfr"].iloc[-1]),
        "baseline_egfr_mean": float(base["egfr"].mean()),
        "baseline_egfr_std": float(base["egfr"].std(ddof=0)) if len(base) > 1 else 0.0,
        "baseline_slope": _ols_slope(base["_years"], base["egfr"]) if len(base) > 1 else 0.0,
        "n_baseline_visits": int(len(base)),
        "baseline_days_covered": float(base["_years"].iloc[-1] * 365.25),
        "ckd_stage": float(base["ckd_stage"].iloc[-1]),
        "slope": slope,
        "n_outcome_visits": int(len(out)),
    }
    for lab in BASELINE_LABS:
        if lab in base.columns:
            row[f"baseline_{lab}"] = float(base[lab].mean())
    row["baseline_slope"] = 0.0 if np.isnan(row["baseline_slope"]) else row["baseline_slope"]
    return row


def build(proc_dir, out_path, baseline_days=180, horizon_days=365,
          threshold=-5.0, min_points=3):
    frames = []
    for split in ("train", "val", "test"):
        d = pd.read_csv(os.path.join(proc_dir, f"{split}.csv"))
        d["split"] = split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])

    rows = []
    for sid, g in df.groupby("subject_id"):
        r = build_patient_row(g, baseline_days, horizon_days, min_points)
        if r is not None:
            r["split"] = g["split"].iloc[0]
            rows.append(r)

    out = pd.DataFrame(rows)
    out["rapid"] = (out["slope"] < threshold).astype(int)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"patients kept: {len(out)} of {df['subject_id'].nunique()}")
    print(f"rapid prevalence: {out['rapid'].mean():.3f}")
    for split in ("train", "val", "test"):
        s = out[out["split"] == split]
        print(f"  {split}: {len(s)} patients, rapid={s['rapid'].mean():.3f}")
    print(f"saved {out_path}")
    return out


if __name__ == "__main__":
    from src.paths import find_processed

    proc = find_processed()
    p = argparse.ArgumentParser()
    p.add_argument("--proc", default=proc)
    p.add_argument("--out", default=os.path.join(proc, "rapid_progressor.csv"))
    p.add_argument("--baseline-days", type=int, default=180)
    p.add_argument("--horizon-days", type=int, default=365)
    p.add_argument("--threshold", type=float, default=-5.0)
    p.add_argument("--min-points", type=int, default=3)
    args = p.parse_args()
    build(args.proc, args.out, args.baseline_days, args.horizon_days,
          args.threshold, args.min_points)
