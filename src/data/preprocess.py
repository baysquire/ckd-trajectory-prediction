"""
Preprocessing pipeline for Paper 1.

Takes raw MIMIC-IV CKD cohort CSV (from the SQL extraction) and produces
train/val/test splits ready for the evaluation scripts.

Steps:
  1. Remove lab values outside reasonable clinical ranges
  2. Forward-fill missing labs within each patient (no back-fill)
  3. Fill any remaining gaps with the column median
  4. Drop patients with fewer than 3 visits
  5. Compute derived features (eGFR, CKD stage, slopes, deltas)
  6. Filter to CKD stage 3-5 (median eGFR < 60)
  7. Split 70/15/15 by patient ID (seed=42)
"""
import json
import os

import numpy as np
import pandas as pd

# keep the import flexible so the script works when run as a module
# or standalone
try:
    from src.features.build_features import build_all_features
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from features.build_features import build_all_features


# reasonable clinical ranges for outlier filtering
LAB_RANGES = {
    'creatinine_mg_dl': (0.3, 30),
    'potassium': (2.0, 8.0),
    'sodium': (110, 165),
    'bun': (2, 150),
    'hemoglobin': (3, 20),
    'albumin': (1.0, 6.0),
    'phosphorus': (1.0, 12),
    'calcium': (5.0, 15),
    'bicarbonate': (5, 45),
}


def remove_outliers(df):
    """Drop rows where a lab value falls outside reasonable clinical range.

    NaN values are kept -- they are not outliers, just missing.
    """
    for col, (lo, hi) in LAB_RANGES.items():
        if col in df.columns:
            before = len(df)
            keep = df[col].isna() | ((df[col] >= lo) & (df[col] <= hi))
            df = df[keep]
            dropped = before - len(df)
            if dropped > 0:
                print(f"  dropped {dropped} rows for {col} outside [{lo}, {hi}]")
    return df


def handle_missing(df):
    """Forward-fill within each patient, then fill remaining with column median.

    Back-fill is deliberately avoided: it would drag future lab values
    into earlier visits, leaking the answer when we forecast forward.
    """
    lab_cols = [c for c in LAB_RANGES.keys() if c in df.columns]

    print("  missingness before imputation:")
    for col in lab_cols:
        miss_pct = df[col].isna().mean() * 100
        print(f"    {col}: {miss_pct:.1f}%")

    df = df.sort_values(['subject_id', 'time_idx'])

    for col in lab_cols:
        df[col] = df.groupby('subject_id')[col].transform(lambda x: x.ffill())

    for col in lab_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    print("  missingness after imputation:")
    for col in lab_cols:
        miss_pct = df[col].isna().mean() * 100
        print(f"    {col}: {miss_pct:.1f}%")

    return df


def enforce_types(df):
    """Ensure categoricals are strings and binary flags are ints."""
    cat_cols = ['gender']
    binary_cols = ['has_diabetes', 'has_hypertension', 'has_heart_failure']

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


def filter_short_patients(df, min_visits=3):
    """Remove patients with fewer than min_visits encounters."""
    counts = df.groupby('subject_id').size()
    valid = counts[counts >= min_visits].index
    dropped = df['subject_id'].nunique() - len(valid)
    df = df[df['subject_id'].isin(valid)]
    if dropped > 0:
        print(f"  removed {dropped} patients with fewer than {min_visits} visits")
    return df


def filter_ckd_stage(df, max_egfr=60):
    """Keep patients whose median eGFR is below max_egfr (CKD stage 3-5)."""
    if 'egfr' not in df.columns:
        return df
    median_egfr = df.groupby('subject_id')['egfr'].median()
    keep = median_egfr[median_egfr < max_egfr].index
    dropped = df['subject_id'].nunique() - len(keep)
    df = df[df['subject_id'].isin(keep)]
    if dropped > 0:
        print(f"  removed {dropped} patients with median egfr >= {max_egfr} (stage 1-2)")
    return df


def write_cohort_summary(df, output_dir):
    """Write a JSON summary of cohort characteristics."""
    stage_counts = {}
    if 'ckd_stage' in df.columns:
        per_patient_stage = df.groupby('subject_id')['ckd_stage'].median().round()
        stage_counts = {int(k): int(v)
                        for k, v in per_patient_stage.value_counts().items()}

    visits = df.groupby('subject_id').size()
    summary = {
        'n_patients': int(df['subject_id'].nunique()),
        'n_rows': int(len(df)),
        'median_visits_per_patient': float(visits.median()),
        'stage_distribution': stage_counts,
    }
    if 'egfr' in df.columns:
        summary['egfr_median'] = round(float(df['egfr'].median()), 2)

    with open(os.path.join(output_dir, 'cohort_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  cohort: {summary['n_patients']} patients, {summary['n_rows']} rows")


def split_by_patient(df, train_frac=0.7, val_frac=0.15):
    """70/15/15 patient-level split with fixed seed for reproducibility."""
    patient_ids = df['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    train_df = df[df['subject_id'].isin(train_ids)]
    val_df = df[df['subject_id'].isin(val_ids)]
    test_df = df[df['subject_id'].isin(test_ids)]

    print(f"  split: {len(train_ids)} train, {len(val_ids)} val, "
          f"{len(test_ids)} test patients")
    return train_df, val_df, test_df


def run_preprocessing(input_path, output_dir):
    """Full pipeline: load -> clean -> features -> split -> save."""
    print(f"loading data from {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')

    is_mimic = 'charttime' in df.columns
    if is_mimic:
        df['charttime'] = pd.to_datetime(df['charttime'])
        df = df.sort_values(['subject_id', 'charttime'])
        df['time_idx'] = df.groupby('subject_id').cumcount()
        print("  detected MIMIC data. generated time_idx from charttime.")

    print(f"  loaded {len(df)} rows, {df['subject_id'].nunique()} patients")

    print("removing outliers...")
    df = remove_outliers(df)

    print("handling missing values...")
    df = handle_missing(df)

    print("enforcing data types...")
    df = enforce_types(df)

    print("filtering short patient histories...")
    df = filter_short_patients(df, min_visits=3)

    os.makedirs(output_dir, exist_ok=True)

    print("building features...")
    df = build_all_features(df)

    if is_mimic:
        print("filtering to CKD stage 3-5...")
        df = filter_ckd_stage(df)
        write_cohort_summary(df, output_dir)

    print("splitting by patient...")
    train_df, val_df, test_df = split_by_patient(df)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"done. saved to {output_dir}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    input_path = os.path.join(project_root, 'data', 'raw', 'mimic_ckd_cohort.csv')
    output_dir = os.path.join(project_root, 'data', 'processed')
    run_preprocessing(input_path, output_dir)
