"""
Feature engineering for Paper 1.

All derived columns used by the evaluation scripts are computed here:
  - eGFR via the race-free CKD-EPI 2021 creatinine equation
  - CKD stage (1-5) from eGFR
  - eGFR slope over a rolling window
  - Visit-to-visit deltas for key labs
  - Rolling mean and standard deviation of eGFR
  - Comorbidity interaction terms
  - Time-since-first-visit counter
  - Prediction target: next visit's eGFR (egfr_next)
"""
import numpy as np
import pandas as pd


def calc_egfr_2021(cr, age, is_female):
    """Race-free CKD-EPI 2021 creatinine equation (Inker et al., NEJM 2021).

    Parameters
    ----------
    cr : array-like
        Serum creatinine in mg/dL.
    age : array-like
        Patient age in years.
    is_female : array-like of bool
        True for female patients.

    Returns
    -------
    numpy array of eGFR values in mL/min/1.73 m^2.
    """
    kappa = np.where(is_female, 0.7, 0.9)
    alpha = np.where(is_female, -0.241, -0.302)

    min_val = np.minimum(cr / kappa, 1)
    max_val = np.maximum(cr / kappa, 1)

    egfr = 142 * (min_val ** alpha) * (max_val ** -1.200) * (0.9938 ** age)
    egfr = np.where(is_female, egfr * 1.012, egfr)

    return egfr


def add_egfr(df):
    """Compute eGFR from creatinine, age, and sex."""
    if 'creatinine_mg_dl' not in df.columns or 'anchor_age' not in df.columns:
        return df
    is_f = df['gender'] == 'F'
    df['egfr'] = calc_egfr_2021(df['creatinine_mg_dl'], df['anchor_age'], is_f)
    df['egfr'] = df['egfr'].round(2)
    return df


def add_ckd_stage(df):
    """Map eGFR to CKD stage 1-5."""
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
    """Create the one-step prediction target: next visit's eGFR."""
    if 'egfr' not in df.columns or 'subject_id' not in df.columns:
        return df
    df = df.sort_values(['subject_id', 'time_idx'])
    df['egfr_next'] = df.groupby('subject_id')['egfr'].shift(-1)
    df = df.dropna(subset=['egfr_next'])
    return df


def add_egfr_slope(df, window=3):
    """Rolling eGFR slope over the last `window` visits."""
    if 'egfr' not in df.columns:
        return df
    df = df.sort_values(['subject_id', 'time_idx'])

    def slope_func(x):
        if len(x) < 2:
            return 0.0
        return (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1)

    df['egfr_slope'] = df.groupby('subject_id')['egfr'].transform(
        lambda x: x.rolling(window=window, min_periods=2).apply(
            slope_func, raw=False)
    )
    df['egfr_slope'] = df['egfr_slope'].fillna(0).round(3)
    return df


def add_lab_deltas(df):
    """Visit-to-visit change for key labs."""
    df = df.sort_values(['subject_id', 'time_idx'])
    delta_cols = ['creatinine_mg_dl', 'potassium', 'hemoglobin', 'egfr']

    for col in delta_cols:
        if col in df.columns:
            df[f'{col}_delta'] = df.groupby('subject_id')[col].diff()
            df[f'{col}_delta'] = df[f'{col}_delta'].fillna(0).round(3)

    return df


def add_rolling_stats(df, window=3):
    """Rolling mean and std of eGFR over last few visits."""
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


def add_comorbidity_features(df):
    """Interaction terms between comorbidities."""
    if 'has_diabetes' in df.columns and 'has_hypertension' in df.columns:
        df['diabetes_x_htn'] = df['has_diabetes'] * df['has_hypertension']
    if 'has_diabetes' in df.columns and 'has_heart_failure' in df.columns:
        df['diabetes_x_hf'] = df['has_diabetes'] * df['has_heart_failure']
    return df


def add_time_since_first(df):
    """Visit counter from each patient's first record."""
    df = df.sort_values(['subject_id', 'time_idx'])
    df['visits_elapsed'] = df.groupby('subject_id').cumcount()
    return df


def build_all_features(df):
    """Run the full feature engineering pipeline in order."""
    print("computing eGFR (CKD-EPI 2021)...")
    df = add_egfr(df)

    print("adding CKD stage...")
    df = add_ckd_stage(df)

    print("computing eGFR slope...")
    df = add_egfr_slope(df)

    print("computing lab deltas...")
    df = add_lab_deltas(df)

    print("computing rolling stats...")
    df = add_rolling_stats(df)

    print("adding comorbidity interactions...")
    df = add_comorbidity_features(df)

    print("adding time features...")
    df = add_time_since_first(df)

    print("adding prediction target (egfr_next)...")
    df = add_egfr_target(df)

    return df
