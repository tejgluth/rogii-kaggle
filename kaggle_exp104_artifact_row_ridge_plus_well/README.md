# ROGII Clean Artifact Row Ridge + Well Submission

Kaggle-ready notebook for the current best clean submission route that can be
reproduced using only attached Kaggle inputs:

- Competition input: `rogii-wellbore-geology-prediction`
- Dataset input: `ravaghi/wellbore-geology-prediction-artifacts`

The notebook builds an artifact-only system:

- reads artifact `train.csv` features, labels, and seven cached OOF delta
  predictions;
- excludes every `sample_submission.csv` row from residual training and CV;
- averages the seven artifact OOF predictions as the base TVT prediction;
- learns the exp100 per-well `gbr_d4` residual correction from non-sample wells;
- trains a conservative row-level ridge residual with GroupKFold over
  non-sample wells;
- selects only the well and row residual weights by non-sample OOF RMSE;
- writes `/kaggle/working/submission.csv`.

Local clean-holdout audit for this input-only variant was RMSE `6.8469` / MSE
`46.8801`, improving exp100's RMSE `6.8816`.
