"""
A few sanity checks so we can trust the numbers before we quote them anywhere.

Checks:
  - train/val/test really hold different patients (no id in two splits)
  - the target egfr_next is just next visit's egfr (so persistence is strong by
    construction, which is expected and worth stating out loud)
  - how correlated current egfr is with next egfr (the autocorrelation problem)
  - no obvious all-null feature columns

Writes a short markdown report so the findings live somewhere, not just stdout.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def load_splits(proc_dir):
    return (
        pd.read_csv(os.path.join(proc_dir, "train.csv")),
        pd.read_csv(os.path.join(proc_dir, "val.csv")),
        pd.read_csv(os.path.join(proc_dir, "test.csv")),
    )


def check_patient_overlap(train, val, test):
    a = set(train["subject_id"])
    b = set(val["subject_id"])
    c = set(test["subject_id"])
    return {
        "train_val_overlap": len(a & b),
        "train_test_overlap": len(a & c),
        "val_test_overlap": len(b & c),
        "clean": (len(a & b) == 0 and len(a & c) == 0 and len(b & c) == 0),
    }


def check_target(test):
    # rebuild what egfr_next should be and see if it matches the stored column
    d = test.sort_values(["subject_id", "time_idx"]).copy()
    rebuilt = d.groupby("subject_id")["egfr"].shift(-1)
    both = d["egfr_next"].notna() & rebuilt.notna()
    match = np.allclose(d.loc[both, "egfr_next"], rebuilt[both], atol=1e-6)
    corr = float(d.loc[both, "egfr"].corr(d.loc[both, "egfr_next"]))
    return {"egfr_next_is_shifted_egfr": bool(match), "corr_egfr_vs_next": round(corr, 4)}


def check_null_columns(train):
    nulls = {c: float(train[c].isna().mean()) for c in train.columns}
    all_null = [c for c, v in nulls.items() if v == 1.0]
    return {"all_null_columns": all_null}


def run(proc_dir, out_path):
    train, val, test = load_splits(proc_dir)
    report = {
        "patient_overlap": check_patient_overlap(train, val, test),
        "target": check_target(test),
        "null_columns": check_null_columns(train),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [
        "# Leakage audit",
        "",
        "## Patient split",
        f"- train/val overlap: {report['patient_overlap']['train_val_overlap']}",
        f"- train/test overlap: {report['patient_overlap']['train_test_overlap']}",
        f"- val/test overlap: {report['patient_overlap']['val_test_overlap']}",
        f"- clean: {report['patient_overlap']['clean']}",
        "",
        "## Target",
        f"- egfr_next is next visit's egfr: {report['target']['egfr_next_is_shifted_egfr']}",
        f"- corr(current egfr, next egfr): {report['target']['corr_egfr_vs_next']}",
        "",
        "This high correlation is why copy-last (persistence) is hard to beat on the",
        "1-step task. It's expected, and it's the reason we move the headline to the",
        "rapid-progressor and multi-horizon tasks.",
        "",
        "## Null columns",
        f"- all-null columns: {report['null_columns']['all_null_columns'] or 'none'}",
        "",
        "## Imputation note",
        "Missing labs are now forward-filled only (see preprocess.handle_missing).",
        "The old code also back-filled, which pulled future values into the past.",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(out_path.replace(".md", ".json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved {out_path}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    from src.paths import find_processed, find_reports_dir

    proc = find_processed()
    reports = find_reports_dir()
    p = argparse.ArgumentParser()
    p.add_argument("--proc", default=proc)
    p.add_argument("--out", default=os.path.join(reports, "leakage_audit.md"))
    args = p.parse_args()
    run(args.proc, args.out)
