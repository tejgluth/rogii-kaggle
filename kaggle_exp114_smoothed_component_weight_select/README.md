# ROGII clean smoothed component weight submission

Kaggle-ready notebook for exp114, the current best clean candidate in this repo.

Required Kaggle inputs:

- Competition: `rogii-wellbore-geology-prediction`
- Dataset: `ravaghi/wellbore-geology-prediction-artifacts`

The notebook:

- reads the artifact `train.csv` and seven cached OOF member predictions;
- excludes every `sample_submission.csv` well from residual training and selection;
- rebuilds the equal-artifact base, `gbr_d4` well residual, and compact row ridge residual;
- selects row alpha, per-well smoother, and residual weights only on non-heldout OOF predictions;
- smooths the base, well-residual, and row-residual components separately before combining them;
- writes `/kaggle/working/submission.csv`.

Local clean-holdout audit: RMSE `6.7768` / MSE `45.9252`. The exp112 audit was RMSE `6.7803`; the raw exp109 audit was RMSE `6.8267`. Heldout labels were not used for fitting, model selection, weight selection, or postprocess selection.
