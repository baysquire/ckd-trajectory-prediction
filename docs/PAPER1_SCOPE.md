# What's in this repo

Paper 1 cohort extraction, preprocessing, and evaluation code:

- Cohort extraction SQL (MIMIC-IV BigQuery)
- Preprocessing pipeline (outlier removal, imputation, feature engineering, splits)
- Leakage audit
- 1-step eGFR baselines
- Rapid-progressor classification
- Multi-horizon eGFR forecasting
- Survival analysis (Cox PH, random survival forest)

Not included:
- TFT training code (TFT results are reported as supplementary context)
- GNN/RL experiments
- Synthetic data generation

To reproduce:

    python -m src.data.preprocess          # if starting from raw MIMIC extract
    python scripts/reproduce_paper1.py     # runs all evaluation steps
