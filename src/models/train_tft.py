import pandas as pd
import numpy as np
import os
import pickle
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import RMSE, MAE
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.data.encoders import NaNLabelEncoder
    has_ptf = True
except ImportError:
    has_ptf = False
    print("pytorch_forecasting not installed.")


def get_feature_columns(df):
    """get available feature columns."""

    static_cats = [c for c in ['gender'] if c in df.columns]
    static_reals = [c for c in ['anchor_age'] if c in df.columns]

    # static comorbidities
    comorbidity_cols = [c for c in ['diabetes', 'hypertension', 'heart_failure'] if c in df.columns]
    for col in comorbidity_cols:
        df[col] = df[col].astype(str)
    static_cats += comorbidity_cols

    # meds are known at each visit (doctor writes the prescription)
    med_cols = [c for c in df.columns if c.startswith('med_')]

    # time-varying knowns. avoid raw time_idx to prevent leakage.
    time_varying_known = med_cols.copy()
    for col in ['visits_elapsed', 'med_count']:
        if col in df.columns:
            time_varying_known.append(col)

    # unknown labs
    lab_cols = [c for c in [
        'egfr', 'creatinine_mg_dl', 'potassium', 'sodium', 'bun',
        'hemoglobin', 'albumin', 'phosphorus', 'calcium', 'bicarbonate'
    ] if c in df.columns]

    # derived features from build_features.py
    derived_cols = [c for c in [
        'egfr_slope', 'creatinine_mg_dl_delta', 'potassium_delta',
        'hemoglobin_delta', 'egfr_delta', 'egfr_rolling_mean', 'egfr_rolling_std'
    ] if c in df.columns]

    time_varying_unknown = lab_cols + derived_cols

    return static_cats, static_reals, time_varying_known, time_varying_unknown


def build_dataset(df, max_encoder_length, max_prediction_length,
                  static_cats, static_reals, time_varying_known, time_varying_unknown):
    """build the pytorch forecasting dataset"""

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="egfr_next",
        group_ids=["subject_id"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_cats,
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        # handle unseen patients gracefully.
        categorical_encoders={"subject_id": NaNLabelEncoder(add_nan=True)},
        target_normalizer=GroupNormalizer(groups=[]),
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


def _subsample_patients(df, n_patients, seed=42):
    """keep only n_patients (helps when training on cpu)."""
    ids = df['subject_id'].unique()
    if n_patients is None or n_patients >= len(ids):
        return df
    rng = np.random.default_rng(seed)
    keep = rng.choice(ids, size=n_patients, replace=False)
    return df[df['subject_id'].isin(keep)]


def train_tft_model(train_path, val_path, test_path=None,
                    max_patients=None, max_epochs=200, max_encoder_length=6):
    if not has_ptf:
        return

    for p in [train_path, val_path]:
        if not os.path.exists(p):
            print(f"missing: {p}")
            return

    print("loading train and val data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # optional subsample so this can finish on a cpu
    if max_patients:
        train_df = _subsample_patients(train_df, max_patients)
        val_df = _subsample_patients(val_df, max(1, max_patients // 5))
        print(f"subsampled to {train_df['subject_id'].nunique()} train / "
              f"{val_df['subject_id'].nunique()} val patients")

    required = ['time_idx', 'egfr_next', 'subject_id']
    for col in required:
        if col not in train_df.columns:
            print(f"missing column: {col}")
            return

    # tft needs string group ids
    train_df['subject_id'] = train_df['subject_id'].astype(str)
    val_df['subject_id'] = val_df['subject_id'].astype(str)

    max_prediction_length = 1

    # keep valid series length
    min_needed = max_prediction_length + 1
    for label, frame in [('train', train_df), ('val', val_df)]:
        counts = frame.groupby("subject_id").size()
        valid = counts[counts >= min_needed].index
        if label == 'train':
            train_df = train_df[train_df["subject_id"].isin(valid)]
        else:
            val_df = val_df[val_df["subject_id"].isin(valid)]
        print(f"  {label}: kept {len(valid)} patients")

    if train_df.empty:
        print("not enough data for training.")
        return

    # columns
    static_cats, static_reals, tv_known, tv_unknown = get_feature_columns(train_df)
    # apply same str conversion to val
    for col in [c for c in ['diabetes', 'hypertension', 'heart_failure'] if c in val_df.columns]:
        val_df[col] = val_df[col].astype(str)

    print(f"features: {len(static_cats)} static cats, {len(static_reals)} static reals, "
          f"{len(tv_known)} known, {len(tv_unknown)} unknown")

    # build train dataset
    training = build_dataset(
        train_df, max_encoder_length, max_prediction_length,
        static_cats, static_reals, tv_known, tv_unknown
    )

    # build val dataset from the same encoding (important!)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

    # dataloaders (bigger batches keep cpu epochs manageable)
    train_dataloader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=256, num_workers=0)

    print("building tft model...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=16,
        output_size=1,
        loss=RMSE(),
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )

    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10,
        verbose=False, mode="min"
    )
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(os.path.dirname(__file__), '../../models'),
        filename='tft_best',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    # setup logger
    try:
        from lightning.pytorch.loggers import TensorBoardLogger
        run_logger = TensorBoardLogger("tb_logs", name="tft_ckd")
    except (ImportError, ModuleNotFoundError):
        from lightning.pytorch.loggers import CSVLogger
        run_logger = CSVLogger("tb_logs", name="tft_ckd")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        enable_model_summary=True,
        logger=run_logger
    )

    print("training (separate train/val splits)...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    print(f"best val loss: {checkpoint.best_model_score:.4f}")

    # load best model
    tft_best = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)

    # --- test evaluation ---
    if test_path and os.path.exists(test_path):
        print("\nevaluating on held-out test set...")
        test_df = pd.read_csv(test_path)
        if max_patients:
            test_df = _subsample_patients(test_df, max(1, max_patients // 5))
        test_df['subject_id'] = test_df['subject_id'].astype(str)
        for col in [c for c in ['diabetes', 'hypertension', 'heart_failure'] if c in test_df.columns]:
            test_df[col] = test_df[col].astype(str)

        # same relaxed rule as train/val so the test set isn't emptied out
        counts = test_df.groupby("subject_id").size()
        valid = counts[counts >= min_needed].index
        test_df = test_df[test_df["subject_id"].isin(valid)]
        print(f"  test: kept {len(valid)} patients")

        if not test_df.empty:
            test_dataset = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
            test_dataloader = test_dataset.to_dataloader(train=False, batch_size=32, num_workers=0)

            # get predictions
            predictions = tft_best.predict(test_dataloader, return_y=True)
            pred_vals = predictions.output
            actual_vals = predictions.y[0] if isinstance(predictions.y, tuple) else predictions.y

            # metrics
            rmse = torch.sqrt(torch.mean((pred_vals - actual_vals) ** 2)).item()
            mae = torch.mean(torch.abs(pred_vals - actual_vals)).item()

            # r-squared
            ss_res = torch.sum((actual_vals - pred_vals) ** 2)
            ss_tot = torch.sum((actual_vals - actual_vals.mean()) ** 2)
            r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

            # persistence baseline (predict next value is same as previous)
            pers_preds = []
            pers_actuals = []
            for x, y in test_dataloader:
                actual = y[0] if isinstance(y, tuple) else y
                # encoder_target is the history of the target variable
                last_val = x['encoder_target'][:, -1]
                pers_preds.append(last_val)
                pers_actuals.append(actual.squeeze())
            
            pers_preds = torch.cat(pers_preds)
            pers_actuals = torch.cat(pers_actuals)
            pers_rmse = torch.sqrt(torch.mean((pers_preds - pers_actuals) ** 2)).item()

            print(f"test RMSE: {rmse:.4f}")
            print(f"test MAE:  {mae:.4f}")
            print(f"test R²:   {r2:.4f}")
            print(f"persistence RMSE: {pers_rmse:.4f}")
            
            if rmse > pers_rmse:
                print("note: model did not beat persistence. we need more features or longer training.")

            # save results
            import json
            results = {'rmse': rmse, 'mae': mae, 'r2': r2}
            results_path = os.path.join(os.path.dirname(__file__), '../../models/tft_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"saved test metrics to {results_path}")
        else:
            print("not enough test patients with long enough histories")

    # --- extract z_P embeddings ---
    print("\nextracting patient embeddings (z_P)...")
    tft_best.eval()
    embed_dict = {}

    # grab lstm output via hook
    captured = {}

    def _grab_encoder(module, inputs, output):
        captured['enc'] = output[0] if isinstance(output, tuple) else output

    handle = tft_best.lstm_encoder.register_forward_hook(_grab_encoder)

    dataloaders_to_extract = [train_dataloader, val_dataloader]
    if test_path and os.path.exists(test_path) and 'test_dataloader' in locals():
        dataloaders_to_extract.append(test_dataloader)

    with torch.no_grad():
        for dl in dataloaders_to_extract:
            for x, y in dl:
                tft_best(x)
                enc = captured['enc']  # [batch, seq_len, hidden]
                z_P = enc.mean(dim=1)  # [batch, hidden_size=32]

                # map back to real patient ids using the categorical encoder
                ids = x['groups']  # [batch, 1]
                flat_ids = ids.squeeze(-1) if ids.dim() > 1 else ids
                subject_ids = training.categorical_encoders["subject_id"].inverse_transform(flat_ids.cpu().numpy())

                for i, pid in enumerate(subject_ids):
                    # overwriting multiple visits is fine, last one represents them
                    embed_dict[str(pid)] = z_P[i].cpu().numpy()

    handle.remove()

    embed_path = os.path.join(os.path.dirname(__file__), '../../data/processed/patient_embeddings.pkl')
    os.makedirs(os.path.dirname(embed_path), exist_ok=True)
    with open(embed_path, 'wb') as f:
        pickle.dump(embed_dict, f)

    pass


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    project_root = os.path.join(base_dir, '../..')

    # prefer real data, fall back to synthetic
    mimic_features = os.path.join(project_root, 'data/processed/mimic_cohort_features.csv')
    synth_features = os.path.join(project_root, 'data/processed/synthetic_cohort_features.csv')

    if os.path.exists(mimic_features):
        # use preprocessed splits
        train_path = os.path.join(project_root, 'data/processed/train.csv')
        val_path = os.path.join(project_root, 'data/processed/val.csv')
        test_path = os.path.join(project_root, 'data/processed/test.csv')
    else:
        # synthetic: we need to split ourselves since the features file is one csv
        # for now just use the same file for both (will be fixed with real data)
        train_path = synth_features
        val_path = synth_features
        test_path = None
        print("NOTE: using synthetic data. train/val is the same file until real data arrives.")

    # optional env overrides
    max_patients = os.environ.get('NEPHRO_MAX_PATIENTS')
    max_patients = int(max_patients) if max_patients else None
    max_epochs = int(os.environ.get('NEPHRO_MAX_EPOCHS', 200))
    max_encoder_length = int(os.environ.get('NEPHRO_ENCODER_LEN', 6))

    train_tft_model(train_path, val_path, test_path,
                    max_patients=max_patients, max_epochs=max_epochs,
                    max_encoder_length=max_encoder_length)
