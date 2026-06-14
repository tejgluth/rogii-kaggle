# ROGII exp092 cached-model submission

Runtime inference notebook for exp092 using self-owned cached base models.

Inputs:

- Competition: `rogii-wellbore-geology-prediction`
- Private dataset: `tejasgluth/rogii-exp092-model-cache`

The private dataset must contain:

```text
models/lightgbm-1/models.pkl
models/lightgbm-1/oof_preds.pkl
models/lightgbm-2/models.pkl
models/lightgbm-2/oof_preds.pkl
models/lightgbm-3/models.pkl
models/lightgbm-3/oof_preds.pkl
models/catboost-1/models.pkl
models/catboost-1/oof_preds.pkl
models/catboost-2/models.pkl
models/catboost-2/oof_preds.pkl
models/catboost-3/models.pkl
models/catboost-3/oof_preds.pkl
```

This notebook still builds train/test features from official competition files so it can predict the runtime test rows, but it skips retraining the six base models. It ignores cached public test predictions.
