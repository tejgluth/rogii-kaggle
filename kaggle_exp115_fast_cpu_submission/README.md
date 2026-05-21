# ROGII exp115 fast CPU submission

Competition-ready Kaggle notebook for the current best clean model, exp115.

Required Kaggle inputs:

- Competition: `rogii-wellbore-geology-prediction`
- Dataset: `ravaghi/wellbore-geology-prediction-artifacts`

This is the optimized scoring notebook. It uses the exp115 parameters that were
selected cleanly on non-heldout OOF rows, then trains only the final residual
models needed for the competition sample rows. It does not rerun the
alpha/smoother/weight search.

Notebook type: CPU. The model uses pandas, scikit-learn Ridge, and
scikit-learn GradientBoostingRegressor; enabling GPU does not speed these
estimators and can add scheduling overhead on Kaggle.

Local clean-holdout audit: RMSE `6.7761` / MSE `45.9151`.

Local execution check: the notebook reproduced
`submissions/exp115_separate_component_smoothers.csv` exactly
(`max_abs_diff=0.0`, float32 exact) in about `411` seconds on this CPU machine.
