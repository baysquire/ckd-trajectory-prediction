import pandas as pd
import numpy as np
import os

# build_features lives one folder over. keep the import lazy-friendly.
try:
    from features.build_features import build_all_features
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
    # drop values outside reasonable clinical ranges, but keep missing
    # values so imputation can fill them later (a NaN isn't an outlier)
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
    # for labs: forward fill within each patient, then fill remaining with median
    lab_cols = [c for c in LAB_RANGES.keys() if c in df.columns]

    print("  missingness before imputation:")
    for col in lab_cols:
        miss_pct = df[col].isna().mean() * 100
        print(f"    {col}: {miss_pct:.1f}%")

    df = df.sort_values(['subject_id', 'time_idx'])

    for col in lab_cols:
        df[col] = df.groupby('subject_id')[col].transform(
            lambda x: x.ffill().bfill()
        )

    # anything still missing gets the column median
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
    # make sure categoricals are strings and numerics are floats
    cat_cols = ['gender']
    binary_cols = ['diabetes', 'hypertension', 'heart_failure']
    med_cols = [c for c in df.columns if c.startswith('med_')]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in binary_cols + med_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df

def filter_short_patients(df, min_visits=3):
    # need enough visits per patient for the time series model
    counts = df.groupby('subject_id').size()
    valid = counts[counts >= min_visits].index
    dropped = df['subject_id'].nunique() - len(valid)
    df = df[df['subject_id'].isin(valid)]
    if dropped > 0:
        print(f"  removed {dropped} patients with fewer than {min_visits} visits")
    return df

def filter_ckd_stage(df, max_egfr=60):
    # the diagnosis codes pull in some milder patients, but the study is about
    # ckd stage 3-5. keep patients whose typical egfr sits below 60.
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
    # small json so we can quote real cohort numbers in the paper
    import json
    stage_counts = {}
    if 'ckd_stage' in df.columns:
        per_patient_stage = df.groupby('subject_id')['ckd_stage'].median().round()
        stage_counts = {int(k): int(v) for k, v in per_patient_stage.value_counts().items()}

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
    # split by patient id so no data leakage between sets
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

    print(f"  split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test patients")
    return train_df, val_df, test_df

def run_preprocessing(input_path, output_dir):
    print(f"loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    is_mimic = 'charttime' in df.columns
    if is_mimic:
        df['charttime'] = pd.to_datetime(df['charttime'])
        df = df.sort_values(['subject_id', 'charttime'])
        df['time_idx'] = df.groupby('subject_id').cumcount()
        print("  detected real mimic data. generated time_idx from charttime.")
        
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
    prefix = 'mimic_cohort' if is_mimic else 'synthetic_cohort'

    # save the cleaned set before features, in case a script wants raw labs
    df.to_csv(os.path.join(output_dir, f'{prefix}_clean.csv'), index=False)

    # build features here so the train/val/test splits actually carry
    # the derived columns and the egfr_next target. otherwise the model
    # scripts get splits with no target and fall over.
    print("building features...")
    df = build_all_features(df)

    # real cohort only: trim to ckd stage 3-5 and note the numbers
    if is_mimic:
        print("filtering to ckd stage 3-5...")
        df = filter_ckd_stage(df)
        write_cohort_summary(df, output_dir)

    df.to_csv(os.path.join(output_dir, f'{prefix}_features.csv'), index=False)

    print("splitting by patient...")
    train_df, val_df, test_df = split_by_patient(df)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"done. saved to {output_dir}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    # default to synthetic if not provided, but favor mimic if it exists
    mimic_path = os.path.join(project_root, 'data', 'raw', 'mimic_ckd_cohort.csv')
    synth_path = os.path.join(project_root, 'data', 'processed', 'synthetic_cohort.csv')
    
    input_path = mimic_path if os.path.exists(mimic_path) else synth_path
    output_dir = os.path.join(project_root, 'data', 'processed')
    run_preprocessing(input_path, output_dir)
