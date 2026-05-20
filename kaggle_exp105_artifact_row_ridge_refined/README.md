# ROGII Clean Artifact Row Ridge Refined Submission

Kaggle-ready notebook for the current best clean submission route that can be
reproduced using only attached Kaggle inputs:

- Competition input: `rogii-wellbore-geology-prediction`
- Dataset input: `ravaghi/wellbore-geology-prediction-artifacts`

The notebook builds an artifact-only system:

- reads artifact `train.csv` features, labels, and seven cached OOF delta
  predictions;
- excludes every `sample_submission.csv` well from residual training and CV;
- averages the seven artifact OOF predictions as the base TVT prediction;
- learns the exp100 per-well `gbr_d4` residual correction from non-sample wells;
- trains a conservative row-level ridge residual with GroupKFold over
  non-sample wells;
- refines alpha and residual weights using non-sample OOF rows only;
- writes `/kaggle/working/submission.csv`.

Local clean-holdout audit for this input-only variant was RMSE `6.8401` / MSE
`46.7872`, improving exp104's RMSE `6.8469` and exp100's RMSE `6.8816`.
