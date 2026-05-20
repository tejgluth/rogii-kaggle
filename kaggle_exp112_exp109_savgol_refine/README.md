# ROGII clean exp109 Savgol refine submission

Kaggle-ready notebook for exp112, the current best clean candidate in this repo.

Required Kaggle inputs:

- Competition: `rogii-wellbore-geology-prediction`
- Dataset: `ravaghi/wellbore-geology-prediction-artifacts`

The notebook:

- reads the artifact `train.csv` and seven cached OOF member predictions;
- excludes every `sample_submission.csv` well from residual training and selection;
- rebuilds the exp109 equal-artifact base, `gbr_d4` well residual, and compact row ridge residual;
- applies the exp112 per-well Savitzky-Golay smoother selected only on non-heldout OOF predictions;
- writes `/kaggle/working/submission.csv`.

Local clean-holdout audit: RMSE `6.7803` / MSE `45.9719`. The raw exp109 audit was RMSE `6.8267`. The heldout labels were not used to select the smoother.
