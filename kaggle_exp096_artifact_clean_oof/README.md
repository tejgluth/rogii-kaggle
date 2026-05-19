# ROGII Clean Artifact OOF Submission

Kaggle-ready notebook for the best clean submission route that can be reproduced
using only attached Kaggle inputs:

- Competition input: `rogii-wellbore-geology-prediction`
- Dataset input: `ravaghi/wellbore-geology-prediction-artifacts`

The notebook builds an all-artifact OOF ensemble:

- reads `data/train.csv` from the artifact dataset for `id` and `last_known_tvt`;
- reads the seven cached artifact OOF delta predictions;
- averages absolute predictions from all seven artifact models;
- merges only the requested `sample_submission.csv` ids;
- writes `/kaggle/working/submission.csv`.

It intentionally does not read the artifact `target` column and does not use
overlapping train labels. Local clean-holdout audit for this input-only variant
was RMSE `7.1336` / MSE `50.8886`.

Note: local `exp095` was slightly better because it also used a local exp028 OOF
component that is not available from the public artifact dataset. This notebook
keeps the input contract strict.
