# ROGII Clean Artifact Well GBR Submission

Kaggle-ready notebook for the current best clean submission route that can be
reproduced using only attached Kaggle inputs:

- Competition input: `rogii-wellbore-geology-prediction`
- Dataset input: `ravaghi/wellbore-geology-prediction-artifacts`

The notebook builds an artifact-only system:

- reads artifact `train.csv` features, labels, and seven cached OOF delta
  predictions;
- excludes every `sample_submission.csv` row from residual training and CV;
- averages the seven artifact OOF predictions as the base TVT prediction;
- learns a per-well residual offset from non-sample wells using well-level
  feature summaries;
- selects a compact residual model grid and shrink weight by 5-fold CV on
  non-sample wells;
- writes `/kaggle/working/submission.csv`.

Local clean-holdout audit for this input-only variant was RMSE `6.8816` / MSE
`47.3569`, improving exp099's RMSE `7.0836` and the original artifact-only OOF
baseline RMSE `7.1336`.
