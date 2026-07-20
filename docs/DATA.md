# Data

## Prerequisites

You need credentialed [PhysioNet](https://physionet.org/) access for MIMIC-IV.

## Option A: Full pipeline (from raw MIMIC-IV)

1. Run the cohort extraction SQL against MIMIC-IV BigQuery:
   ```
   src/data/sql/extract_ckd_cohort.sql
   ```
2. Save the output as `data/raw/mimic_ckd_cohort.csv`.
3. Run preprocessing:
   ```bash
   python -m src.data.preprocess
   ```
   This produces `data/processed/train.csv`, `val.csv`, and `test.csv`.

## Option B: Pre-processed CSVs

If you already have the processed files, place them in:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Or set the environment variable:

```text
NEPHRO_DATA_PATH=/path/to/processed
```

## Column Dictionary

### Raw columns (from SQL extraction)

| Column | Type | Unit | Source Table |
|:-------|:-----|:-----|:------------|
| `subject_id` | int | — | `patients` |
| `gender` | str | M/F | `patients` |
| `anchor_age` | int | years | `patients` |
| `has_diabetes` | int (0/1) | — | `diagnoses_icd` |
| `has_hypertension` | int (0/1) | — | `diagnoses_icd` |
| `has_heart_failure` | int (0/1) | — | `diagnoses_icd` |
| `charttime` | datetime | — | `labevents` |
| `creatinine_mg_dl` | float | mg/dL | `labevents` (50912, 52546) |
| `potassium` | float | mEq/L | `labevents` (50971, 52610) |
| `sodium` | float | mEq/L | `labevents` (50983, 52623) |
| `bun` | float | mg/dL | `labevents` (51006) |
| `hemoglobin` | float | g/dL | `labevents` (50811, 51222) |
| `albumin` | float | g/dL | `labevents` (50862) |
| `phosphorus` | float | mg/dL | `labevents` (50970) |

### Derived columns (from preprocessing)

| Column | Type | Unit | Source |
|:-------|:-----|:-----|:-------|
| `time_idx` | int | — | cumulative visit count per patient |
| `egfr` | float | mL/min/1.73m² | CKD-EPI 2021 from creatinine, age, sex |
| `egfr_next` | float | mL/min/1.73m² | next visit's eGFR (prediction target) |
| `ckd_stage` | int (1-5) | — | from eGFR thresholds (90/60/30/15) |
| `egfr_slope` | float | per visit | rolling 3-visit eGFR slope |
| `creatinine_mg_dl_delta` | float | mg/dL | visit-to-visit creatinine change |
| `potassium_delta` | float | mEq/L | visit-to-visit potassium change |
| `hemoglobin_delta` | float | g/dL | visit-to-visit hemoglobin change |
| `egfr_delta` | float | mL/min/1.73m² | visit-to-visit eGFR change |
| `egfr_rolling_mean` | float | mL/min/1.73m² | 3-visit rolling mean eGFR |
| `egfr_rolling_std` | float | mL/min/1.73m² | 3-visit rolling std eGFR |
| `diabetes_x_htn` | int (0/1) | — | diabetes × hypertension interaction |
| `diabetes_x_hf` | int (0/1) | — | diabetes × heart failure interaction |
| `visits_elapsed` | int | — | visits since patient's first record |
