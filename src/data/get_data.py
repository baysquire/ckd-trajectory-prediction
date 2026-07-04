import os
import pandas as pd
import numpy as np
from google.cloud import bigquery

from src.drug_formulary import DRUG_NAMES, med_column, normalize_mimic_drug


def get_cohort(project_id=None):
    """pull ckd cohort from bigquery."""

    curr_dir = os.path.dirname(__file__)
    sql_file = os.path.join(curr_dir, 'sql', 'extract_ckd_cohort.sql')

    with open(sql_file, 'r') as f:
        query = f.read()

    if not project_id:
        project_id = 'nephroai-mimic-extract'

    print(f"connecting to bigquery ({project_id})...")
    client = bigquery.Client(project=project_id)

    print("extracting ckd cohort...")
    df = client.query(query).to_dataframe()
    print(f"  got {len(df)} rows, {df['subject_id'].nunique()} patients")

    print(f"\n  creatinine range: {df['creatinine_mg_dl'].min():.1f} - {df['creatinine_mg_dl'].max():.1f}")
    print(f"  potassium range:  {df['potassium'].min():.1f} - {df['potassium'].max():.1f}")
    print(f"  age range:        {df['anchor_age'].min()} - {df['anchor_age'].max()}")
    print(f"  gender split:     {df.groupby('subject_id')['gender'].first().value_counts().to_dict()}")

    per_patient = df.groupby('subject_id').first()
    print(f"  diabetes:      {per_patient['has_diabetes'].sum()}")
    print(f"  hypertension:  {per_patient['has_hypertension'].sum()}")
    print(f"  heart failure: {per_patient['has_heart_failure'].sum()}")

    print("\n  missingness:")
    for col in ['creatinine_mg_dl', 'potassium', 'sodium', 'bun', 'hemoglobin', 'albumin', 'phosphorus']:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            print(f"    {col}: {pct:.1f}%")

    visits = df.groupby('subject_id').size()
    print(f"\n  visits per patient: median={visits.median():.0f}, min={visits.min()}, max={visits.max()}")

    out_path = os.path.abspath(os.path.join(curr_dir, '../../data/raw/mimic_ckd_cohort.csv'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved to {out_path}")

    return df


def get_medications(project_id=None):
    """pull meds."""

    curr_dir = os.path.dirname(__file__)
    sql_file = os.path.join(curr_dir, 'sql', 'extract_medications.sql')

    with open(sql_file, 'r') as f:
        query = f.read()

    if not project_id:
        project_id = 'nephroai-mimic-extract'

    client = bigquery.Client(project=project_id)
    print("extracting medication history...")
    meds_df = client.query(query).to_dataframe()
    print(f"  got {len(meds_df)} prescription records, {meds_df['subject_id'].nunique()} patients")

    out_path = os.path.abspath(os.path.join(curr_dir, '../../data/raw/mimic_medications.csv'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meds_df.to_csv(out_path, index=False)
    print(f"saved to {out_path}")

    return meds_df


def merge_meds_with_cohort(cohort_df, meds_df):
    """align meds with lab visits."""

    cohort_df = cohort_df.copy()
    for drug in DRUG_NAMES:
        cohort_df[med_column(drug)] = 0

    if meds_df is None or meds_df.empty:
        return cohort_df

    meds_df = meds_df.copy()
    meds_df['canonical_drug'] = meds_df['medication'].apply(normalize_mimic_drug)
    meds_df = meds_df.dropna(subset=['canonical_drug'])
    meds_df['starttime'] = pd.to_datetime(meds_df['starttime'])
    if 'stoptime' in meds_df.columns:
        meds_df['stoptime'] = pd.to_datetime(meds_df['stoptime'])
    else:
        meds_df['stoptime'] = pd.NaT

    cohort_df['charttime'] = pd.to_datetime(cohort_df['charttime'])

    # match intervals
    for pid, pat_labs in cohort_df.groupby('subject_id'):
        pat_meds = meds_df[meds_df['subject_id'] == pid]
        if pat_meds.empty:
            continue

        idx = pat_labs.index
        visit_times = pat_labs['charttime'].values

        for drug in DRUG_NAMES:
            col = med_column(drug)
            drug_rows = pat_meds[pat_meds['canonical_drug'] == drug]
            if drug_rows.empty:
                continue

            active = np.zeros(len(idx), dtype=np.int8)
            for _, rx in drug_rows.iterrows():
                start = rx['starttime']
                stop = rx['stoptime']
                if pd.isna(stop):
                    # default to 30 days
                    stop = start + pd.Timedelta(days=30)
                mask = (visit_times >= np.datetime64(start)) & (visit_times <= np.datetime64(stop))
                active |= mask.astype(np.int8)

            cohort_df.loc[idx, col] = active

    return cohort_df


if __name__ == '__main__':
    cohort = get_cohort()
    meds = get_medications()
    merged = merge_meds_with_cohort(cohort, meds)

    out_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../../data/raw/mimic_ckd_cohort.csv'))
    merged.to_csv(out_path, index=False)
    print(f"\nfinal merged dataset: {len(merged)} rows, {merged['subject_id'].nunique()} patients")
    med_cols = [c for c in merged.columns if c.startswith('med_')]
    print(f"med columns: {len(med_cols)}")
