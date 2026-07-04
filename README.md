# Predicting Chronic Kidney Disease Trajectories using Temporal Fusion Transformers

This repository contains the official implementation of our paper on forecasting eGFR trajectories for Chronic Kidney Disease (CKD) patients using Temporal Fusion Transformers (TFT) on the MIMIC-IV dataset.

## Overview
Accurate prediction of CKD progression is critical for timely clinical intervention. We utilize a TFT architecture to handle both static comorbidities and time-varying clinical signals (laboratory results, medication histories) to forecast future eGFR values.

## Repository Structure
- `src/data/`: Scripts and BigQuery SQL to extract the CKD cohort and medication history from MIMIC-IV.
- `src/features/`: Feature engineering pipeline (calculating eGFR using CKD-EPI 2021, rolling stats, lab deltas).
- `src/models/`: The PyTorch Forecasting implementation of the TFT model, including training loops and persistence baselines.

## Requirements
To install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Extraction
You must have credentialed access to MIMIC-IV on PhysioNet. Authenticate your Google Cloud SDK and run:
```bash
python src/data/get_data.py
```

### 2. Feature Engineering
Process the raw extraction into model-ready features:
```bash
python src/features/build_features.py
```

### 3. Model Training
Train the Temporal Fusion Transformer and evaluate against the persistence baseline:
```bash
python src/models/train_tft.py
```

## License
This code is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. It is freely available for academic research and peer review, but commercial use is strictly prohibited. See `LICENSE` for details.
