# CKD progression baselines (Paper 1)

Code for the MIMIC-IV CKD progression paper:
rapid-progressor ID, multi-horizon eGFR, and survival.

This repo is only the Paper 1 evals. It does not include the other
experimental models from the larger research folder.

## Results (frozen)

Rapid-progressor (test n=1360): logistic AUROC 0.594 vs KFRE-proxy 0.415
and prior-slope 0.492.

1-step eGFR: XGBoost RMSE 7.42, persistence 7.99, TFT 8.94.

See `docs/PAPER1_NUMBER_FREEZE.md` for the full table.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

MIMIC-IV is not in this repo. After you have PhysioNet access, put
processed CSVs here:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Optional: set `NEPHRO_DATA_PATH` to that folder.

## Run

```bash
python scripts/reproduce_paper1.py
```

Outputs go under `results/`.

## License

See LICENSE.
