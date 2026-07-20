"""
Train and score the rapid-progressor classifier and, just as important, the
baselines it has to beat.

Baselines:
  - prior-slope rule: rank risk by how fast eGFR was already dropping at baseline
  - KFRE proxy: the kidney failure risk equation (Tangri), ACR fixed since MIMIC
    doesn't give us albuminuria here. It targets kidney failure, not slope, so
    it's a fair-but-imperfect yardstick; we say so.
  - logistic regression on the baseline features

Models: xgboost and lightgbm.

We report AUROC, AUPRC, Brier and calibration, break it down by subgroup, and
bootstrap the gap between our best model and the baselines so a reviewer can see
whether the win is real or noise.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

DROP = {"subject_id", "split", "slope", "rapid", "n_outcome_visits"}


def feature_cols(df):
    return [c for c in df.columns if c not in DROP]


def kfre_proxy_score(df, ln_acr=np.log(30.0)):
    # 4-variable KFRE linear predictor (Tangri et al. 2011). we hold ACR fixed
    # because we don't have albuminuria in this extract, so this mostly ranks by
    # age, sex and eGFR. good enough as a baseline, and we flag the caveat.
    age = df["anchor_age"].to_numpy()
    male = (1 - df["is_female"]).to_numpy()
    egfr = df["baseline_egfr"].to_numpy()
    lp = (-0.2201 * (age / 10 - 7.036)
          + 0.2467 * (male - 0.5642)
          - 0.5567 * (egfr / 5 - 7.222)
          + 0.4510 * (ln_acr - 5.137))
    return lp  # higher = higher predicted failure risk


def bootstrap_auc_gap(y, score_a, score_b, n=1000, seed=42):
    # paired bootstrap: is model A's AUROC actually above model B's?
    y = np.asarray(y)
    a = np.asarray(score_a)
    b = np.asarray(score_b)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    gaps = []
    for _ in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[pick])) < 2:
            continue
        gaps.append(roc_auc_score(y[pick], a[pick]) - roc_auc_score(y[pick], b[pick]))
    gaps = np.array(gaps)
    return {
        "auc_gap": round(float(gaps.mean()), 4),
        "gap_ci": [round(float(np.percentile(gaps, 2.5)), 4),
                   round(float(np.percentile(gaps, 97.5)), 4)],
        "p_gap_gt_0": round(float((gaps > 0).mean()), 4),
    }


def auc_ci(y, score, n=1000, seed=42):
    y = np.asarray(y)
    s = np.asarray(score)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[pick])) < 2:
            continue
        vals.append(roc_auc_score(y[pick], s[pick]))
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def score_block(y, score, positive_scores_are_prob=False):
    out = {
        "auroc": round(float(roc_auc_score(y, score)), 4),
        "auprc": round(float(average_precision_score(y, score)), 4),
        "auroc_ci": auc_ci(y, score),
    }
    if positive_scores_are_prob:
        out["brier"] = round(float(brier_score_loss(y, score)), 4)
    return out


def subgroup(y, score, df, col):
    out = {}
    for v in sorted(df[col].unique()):
        m = (df[col] == v).to_numpy()
        if m.sum() > 30 and len(np.unique(np.asarray(y)[m])) == 2:
            out[f"{col}={v}"] = round(float(roc_auc_score(np.asarray(y)[m], np.asarray(score)[m])), 4)
    return out


def run(data_path, out_dir, fig_dir):
    df = pd.read_csv(data_path)
    cols = feature_cols(df)
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]

    # logistic regression can't handle NaN, so fill with train medians.
    # tree models handle NaN on their own, so leave those alone.
    train_medians = tr[cols].median()
    Xtr_imputed = tr[cols].fillna(train_medians)
    Xva_imputed = va[cols].fillna(train_medians)
    Xte_imputed = te[cols].fillna(train_medians)

    Xtr_raw, Xva_raw, Xte_raw = tr[cols], va[cols], te[cols]
    ytr = tr["rapid"].to_numpy()
    yva = va["rapid"].to_numpy()
    yte = te["rapid"].to_numpy()

    scores = {}       # name -> test scores (prob or rank)
    val_scores = {}   # name -> validation scores (for model selection)
    results = {"n": {"train": len(tr), "val": len(va), "test": len(te)},
               "prevalence": round(float(te["rapid"].mean()), 4),
               "models": {}}

    # baseline 1: prior slope. more negative slope -> higher risk, so negate.
    scores["prior_slope"] = (-te["baseline_slope"]).to_numpy()
    results["models"]["prior_slope"] = score_block(yte, scores["prior_slope"])

    # baseline 2: KFRE proxy
    scores["kfre_proxy"] = kfre_proxy_score(te)
    results["models"]["kfre_proxy"] = score_block(yte, scores["kfre_proxy"])
    results["models"]["kfre_proxy"]["note"] = "ACR fixed; targets kidney failure not slope"

    # baseline 3: logistic regression (uses median-imputed data)
    scaler = StandardScaler().fit(Xtr_imputed)
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(scaler.transform(Xtr_imputed), ytr)
    val_scores["logistic"] = logit.predict_proba(scaler.transform(Xva_imputed))[:, 1]
    scores["logistic"] = logit.predict_proba(scaler.transform(Xte_imputed))[:, 1]
    results["models"]["logistic"] = score_block(yte, scores["logistic"], True)

    # model: xgboost (handles NaN natively)
    xgbc = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=30, random_state=42, missing=np.nan,
    )
    xgbc.fit(Xtr_raw, ytr, eval_set=[(Xva_raw, yva)], verbose=False)
    val_scores["xgboost"] = xgbc.predict_proba(Xva_raw)[:, 1]
    scores["xgboost"] = xgbc.predict_proba(Xte_raw)[:, 1]
    results["models"]["xgboost"] = score_block(yte, scores["xgboost"], True)

    # model: lightgbm (handles NaN natively)
    lgbc = lgb.LGBMClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, verbose=-1,
    )
    lgbc.fit(Xtr_raw, ytr, eval_set=[(Xva_raw, yva)],
             callbacks=[lgb.early_stopping(30, verbose=False)])
    val_scores["lightgbm"] = lgbc.predict_proba(Xva_raw)[:, 1]
    scores["lightgbm"] = lgbc.predict_proba(Xte_raw)[:, 1]
    results["models"]["lightgbm"] = score_block(yte, scores["lightgbm"], True)

    # pick best model on validation, not test, to keep test set honest.
    best = max(("xgboost", "lightgbm", "logistic"),
               key=lambda m: roc_auc_score(yva, val_scores[m]))
    results["best_model"] = best
    results["best_model_val_auroc"] = round(float(roc_auc_score(yva, val_scores[best])), 4)
    results["vs_kfre"] = bootstrap_auc_gap(yte, scores[best], scores["kfre_proxy"])
    results["vs_prior_slope"] = bootstrap_auc_gap(yte, scores[best], scores["prior_slope"])

    # subgroups for the best model
    results["subgroups"] = {}
    for col in ("ckd_stage", "has_diabetes", "is_female"):
        if col in te.columns:
            results["subgroups"].update(subgroup(yte, scores[best], te, col))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rapid_progressor_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _figures(yte, scores, best, xgbc, Xte, fig_dir)
    _table(results, fig_dir)
    print(json.dumps(results, indent=2))
    return results


def _figures(y, scores, best, xgb_model, Xte, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # ROC: best model vs the two baselines
    plt.figure(figsize=(5, 5))
    for name in (best, "kfre_proxy", "prior_slope"):
        fpr, tpr, _ = roc_curve(y, scores[name])
        plt.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y, scores[name]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("rapid progressor: ROC")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "F1_roc.png"), dpi=130)
    plt.close()

    # precision-recall for the best model (minority class matters)
    plt.figure(figsize=(5, 5))
    prec, rec, _ = precision_recall_curve(y, scores[best])
    plt.plot(rec, prec, label=f"{best} (AUPRC {average_precision_score(y, scores[best]):.3f})")
    plt.axhline(np.mean(y), ls="--", c="k", lw=0.8, label=f"prevalence {np.mean(y):.2f}")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("rapid progressor: precision-recall")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "F2_pr.png"), dpi=130)
    plt.close()

    # calibration
    plt.figure(figsize=(5, 5))
    frac_pos, mean_pred = calibration_curve(y, scores[best], n_bins=10)
    plt.plot(mean_pred, frac_pos, "o-", label=best)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("predicted risk")
    plt.ylabel("observed rate")
    plt.title("rapid progressor: calibration")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "F3_calibration.png"), dpi=130)
    plt.close()

    # shap summary for the tree model, if it's the best one
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer.shap_values(Xte)
        shap.summary_plot(sv, Xte, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "F4_shap.png"), dpi=130)
        plt.close()
    except Exception as e:
        print(f"skipped shap plot: {e}")


def _table(results, fig_dir):
    m = results["models"]
    lines = [
        "# Rapid-progressor results",
        "",
        f"Test patients: {results['n']['test']}, rapid prevalence {results['prevalence']}.",
        "",
        "| Model | AUROC | AUPRC |",
        "|:------|------:|------:|",
    ]
    order = sorted(m, key=lambda k: m[k]["auroc"], reverse=True)
    for k in order:
        lines.append(f"| {k} | {m[k]['auroc']} | {m[k]['auprc']} |")
    best = results["best_model"]
    lines += [
        "",
        f"Best model: **{best}**.",
        f"- vs KFRE proxy: AUROC gap {results['vs_kfre']['auc_gap']} "
        f"(95% CI {results['vs_kfre']['gap_ci']}).",
        f"- vs prior-slope rule: AUROC gap {results['vs_prior_slope']['auc_gap']} "
        f"(95% CI {results['vs_prior_slope']['gap_ci']}).",
        "",
        "The prior-slope rule is the honest floor: a clinician with a ruler. If we",
        "can't clear it, the model isn't adding much.",
    ]
    with open(os.path.join(fig_dir, "..", "rapid_progressor_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    from src.paths import find_processed, find_models_dir, find_reports_dir

    proc = find_processed()
    models = find_models_dir()
    reports = find_reports_dir()
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(proc, "rapid_progressor.csv"))
    p.add_argument("--out", default=models)
    p.add_argument("--figs", default=os.path.join(reports, "figures"))
    args = p.parse_args()
    run(args.data, args.out, args.figs)
