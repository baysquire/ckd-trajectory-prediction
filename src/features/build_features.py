import pandas as pd
import numpy as np

def calc_egfr_2021(cr, age, is_female):
    # 2021 ckd-epi equation, no race variable
    kappa = np.where(is_female, 0.7, 0.9)
    alpha = np.where(is_female, -0.241, -0.302)

    min_val = np.minimum(cr / kappa, 1)
    max_val = np.maximum(cr / kappa, 1)

    egfr = 142 * (min_val ** alpha) * (max_val ** -1.200) * (0.9938 ** age)
    egfr = np.where(is_female, egfr * 1.012, egfr)

    return egfr

def add_egfr(df):
    if 'creatinine_mg_dl' not in df.columns or 'anchor_age' not in df.columns:
        return df
    is_f = df['gender'] == 'F'
    df['egfr'] = calc_egfr_2021(df['creatinine_mg_dl'], df['anchor_age'], is_f)
    df['egfr'] = df['egfr'].round(2)
    return df

def add_ckd_stage(df):
    # map egfr to stage
    if 'egfr' not in df.columns:
        return df
    conditions = [
        df['egfr'] >= 90,
        df['egfr'] >= 60,
        df['egfr'] >= 30,
        df['egfr'] >= 15,
        df['egfr'] < 15,
    ]
    stages = [1, 2, 3, 4, 5]
    df['ckd_stage'] = np.select(conditions, stages, default=0)
    return df

def add_egfr_target(df):
    # target: next egfr
    if 'egfr' not in df.columns or 'subject_id' not in df.columns:
        return df
    df = df.sort_values(['subject_id', 'time_idx'])
    df['egfr_next'] = df.groupby('subject_id')['egfr'].shift(-1)
    df = df.dropna(subset=['egfr_next'])
    return df

def add_egfr_slope(df, window=3):
    # egfr change rate
    if 'egfr' not in df.columns:
        return df
    df = df.sort_values(['subject_id', 'time_idx'])

    def slope_func(x):
        if len(x) < 2:
            return 0.0
        # diff between last and first
        return (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1)

    df['egfr_slope'] = df.groupby('subject_id')['egfr'].transform(
        lambda x: x.rolling(window=window, min_periods=2).apply(slope_func, raw=False)
    )
    df['egfr_slope'] = df['egfr_slope'].fillna(0).round(3)
    return df

def add_lab_deltas(df):
    # diff from last visit
    df = df.sort_values(['subject_id', 'time_idx'])
    delta_cols = ['creatinine_mg_dl', 'potassium', 'hemoglobin', 'egfr']

    for col in delta_cols:
        if col in df.columns:
            df[f'{col}_delta'] = df.groupby('subject_id')[col].diff()
            df[f'{col}_delta'] = df[f'{col}_delta'].fillna(0).round(3)

    return df

def add_rolling_stats(df, window=3):
    # rolling mean/std
    if 'egfr' not in df.columns:
        return df
    df = df.sort_values(['subject_id', 'time_idx'])

    df['egfr_rolling_mean'] = df.groupby('subject_id')['egfr'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    ).round(2)

    df['egfr_rolling_std'] = df.groupby('subject_id')['egfr'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    ).fillna(0).round(2)

    return df

def add_med_count(df):
    # count active meds
    med_cols = [c for c in df.columns if c.startswith('med_')]
    if med_cols:
        df['med_count'] = df[med_cols].sum(axis=1)
    return df

def add_comorbidity_features(df):
    # comorbidity interactions
    if 'has_diabetes' in df.columns and 'has_hypertension' in df.columns:
        df['diabetes_x_htn'] = df['has_diabetes'] * df['has_hypertension']
    if 'has_diabetes' in df.columns and 'has_heart_failure' in df.columns:
        df['diabetes_x_hf'] = df['has_diabetes'] * df['has_heart_failure']
    return df

def add_time_since_first(df):
    # visits since first record
    df = df.sort_values(['subject_id', 'time_idx'])
    df['visits_elapsed'] = df.groupby('subject_id').cumcount()
    return df

def build_all_features(df):
    # run everything in order
    print("computing egfr (ckd-epi 2021)...")
    df = add_egfr(df)

    print("adding ckd stage...")
    df = add_ckd_stage(df)

    print("computing egfr slope...")
    df = add_egfr_slope(df)

    print("computing lab deltas...")
    df = add_lab_deltas(df)

    print("computing rolling stats...")
    df = add_rolling_stats(df)

    print("counting medications...")
    df = add_med_count(df)
    
    print("adding comorbidity interactions...")
    df = add_comorbidity_features(df)

    print("adding time features...")
    df = add_time_since_first(df)

    print("adding prediction target (egfr_next)...")
    df = add_egfr_target(df)

    return df

if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    mimic_path = os.path.join(project_root, 'data', 'processed', 'mimic_cohort_clean.csv')
    synth_path = os.path.join(project_root, 'data', 'processed', 'synthetic_cohort_clean.csv')
    raw_synth_path = os.path.join(project_root, 'data', 'processed', 'synthetic_cohort.csv')
    
    if os.path.exists(mimic_path):
        input_path = mimic_path
        prefix = 'mimic'
    elif os.path.exists(synth_path):
        input_path = synth_path
        prefix = 'synthetic'
    else:
        input_path = raw_synth_path
        prefix = 'synthetic'

    data = pd.read_csv(input_path)
    data = build_all_features(data)

    output_path = os.path.join(project_root, 'data', 'processed', f'{prefix}_cohort_features.csv')
    data.to_csv(output_path, index=False)
    print(f"saved {len(data)} rows with {len(data.columns)} features to {output_path}")
