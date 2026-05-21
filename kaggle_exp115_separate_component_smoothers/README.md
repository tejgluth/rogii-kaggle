# ROGII clean separate component smoothers submission

Kaggle-ready notebook for exp115, the current best clean candidate in this repo.

Required Kaggle inputs:

- Competition: `rogii-wellbore-geology-prediction`
- Dataset: `ravaghi/wellbore-geology-prediction-artifacts`

The notebook:

- reads the artifact `train.csv` and seven cached OOF member predictions;
- excludes every `sample_submission.csv` well from residual training and selection;
- rebuilds the equal-artifact base, `gbr_d4` well residual, and compact row ridge residual;
- selects row alpha, separate base/row smoothers, and residual weights only on non-heldout OOF predictions;
- applies the selected smoothers to the heldout/sample rows and writes `/kaggle/working/submission.csv`.

Local clean-holdout audit: RMSE `6.7761` / MSE `45.9151`. The exp114 audit was RMSE `6.7768`; the raw exp109 audit was RMSE `6.8267`. Heldout labels were not used for fitting, model selection, weight selection, or postprocess selection.
