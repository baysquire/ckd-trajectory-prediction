# CKD Progression Baselines (Paper 1)

Code for: *Baseline Models for CKD Progression Prediction on MIMIC-IV*.
Preprint available at: [https://doi.org/10.5281/zenodo.21463230](https://doi.org/10.5281/zenodo.21463230)

Three tasks evaluated with honest baselines:
- Rapid-progressor identification
- Multi-horizon eGFR forecasting
- Time-to-stage-transition survival analysis

## Results (frozen)

Rapid-progressor (test n=1,360): logistic AUROC 0.594 vs KFRE-proxy 0.415
and prior-slope 0.492.

1-step eGFR: XGBoost RMSE 7.42, persistence 7.99, TFT 8.94.

See `docs/PAPER1_NUMBER_FREEZE.md` for the full table.

## Requirements

- Python >= 3.10
- Dependencies: `pip install -r requirements.txt`

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

MIMIC-IV is not redistributed. After obtaining PhysioNet access:

**Option A** (full pipeline): Run the cohort extraction SQL
(`src/data/sql/extract_ckd_cohort.sql`) against MIMIC-IV BigQuery,
save the output as `data/raw/mimic_ckd_cohort.csv`, then run:

```bash
python -m src.data.preprocess
```

**Option B** (pre-processed): Place `train.csv`, `val.csv`, `test.csv`
in `data/processed/`, or set `NEPHRO_DATA_PATH` to that folder.

See `docs/DATA.md` for the full column dictionary.

## Run

```bash
python scripts/reproduce_paper1.py
```

Outputs go under `results/`.

## Repository structure

```
src/
  data/
    sql/extract_ckd_cohort.sql   # Cohort extraction query
    preprocess.py                 # Full preprocessing pipeline
  features/
    build_features.py             # eGFR, slopes, deltas, staging
  eval/
    leakage_audit.py              # Patient overlap + target checks
    run_baselines_14k.py          # 1-step eGFR baselines
    multihorizon.py               # Multi-horizon forecasting
  tasks/
    rapid_progressor.py           # Landmark labelling
    train_rapid_progressor.py     # Classification models + baselines
    survival.py                   # Cox PH + random survival forest
scripts/
  reproduce_paper1.py             # Run all steps in order
docs/
  DATA.md                         # Column dictionary
  PAPER1_NUMBER_FREEZE.md         # Frozen metrics
  PAPER1_SCOPE.md                 # What is / isn't in this repo
```

## License

See LICENSE.
