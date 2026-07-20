# Paper 1 — Frozen Numbers

## Headline abstract numbers

- Cohort: 14,229 patients; 902,352 rows; median visits 40; median eGFR 36.57
- Rapid-progressor test n=1,360; prevalence 0.3868
- Best: **logistic AUROC 0.5943** [0.5608, 0.6247]; AUPRC 0.4737
- KFRE-proxy AUROC 0.4149; prior-slope 0.4916
- Gaps: vs KFRE +0.1788 [0.1192, 0.2346]; vs prior-slope +0.1019 [0.0583, 0.1456]
- 1-step: XGB RMSE 7.4231 / R² 0.8809; persistence 7.9862 / 0.8622; TFT 8.9388 / 0.8248
- Multi-horizon XGB RMSE: h1 7.4334, h3 10.3005, h6 12.3968, h12 14.3911
- Survival: RSF C-index 0.7329; Cox 0.7189; event rate 0.6684; test n=2,135
- Leakage: patient overlap 0; corr(egfr, egfr_next)=0.9313
