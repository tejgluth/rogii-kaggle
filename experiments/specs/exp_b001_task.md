# Baseline Task: LightGBM (exp_b001)

Build a complete LightGBM training pipeline and get a CV RMSE score.

## CRITICAL: Actual Data Schema (from EDA)

**Train columns**: MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input
**Test columns**:  MD, X, Y, Z, GR, TVT_input  ← test has only 6 columns
**Typewell columns**: TVT, GR, Geology

**ONLY use features available in test**: MD, X, Y, Z, GR, TVT_input (plus derived features from these + typewell)

## Data Loading

Data is at:
- Train horizontal wells: `data/raw/rogii-wellbore-geology-prediction/train/*__horizontal_well.csv`
- Test horizontal wells:  `data/raw/rogii-wellbore-geology-prediction/test/*__horizontal_well.csv`
- Typewells:              `data/raw/rogii-wellbore-geology-prediction/train/*__typewell.csv`

Each filename: `{well_hash}__horizontal_well.csv`. The well_hash is the well_id.

Load all train horizontal wells into one DataFrame. Add a `well_id` column = the hash prefix from the filename.

## Target
Column `TVT` in horizontal_well CSVs. This is the regression target.

## Features to Engineer (baseline)

Per well, sort by MD first. Then:

```python
# Position features
depth_from_heel = MD - MD.min()           # per well
depth_fraction  = depth_from_heel / depth_from_heel.max()  # 0-1 normalized

# GR features (handle NaN: fill with per-well median, then global median)
gr_filled = GR.fillna(GR.median())  # per well
gr_rolling_mean_10  = gr_filled.rolling(10, center=True, min_periods=1).mean()
gr_rolling_mean_30  = gr_filled.rolling(30, center=True, min_periods=1).mean()
gr_rolling_std_10   = gr_filled.rolling(10, center=True, min_periods=1).std().fillna(0)
gr_rolling_mean_100 = gr_filled.rolling(100, center=True, min_periods=1).mean()

# TVT_input (74% missing — treat as useful where available)
tvt_input_filled = TVT_input.fillna(TVT_input.median())  # per well

# XYZ trajectory  
x_diff = X.diff().fillna(0)
y_diff = Y.diff().fillna(0)
z_diff = Z.diff().fillna(0)
lateral_speed = (x_diff**2 + y_diff**2 + z_diff**2)**0.5
```

Final feature list:
```python
FEATURES = [
    "MD", "X", "Y", "Z",
    "depth_from_heel", "depth_fraction",
    "gr_filled", "gr_rolling_mean_10", "gr_rolling_mean_30",
    "gr_rolling_std_10", "gr_rolling_mean_100",
    "tvt_input_filled",
    "x_diff", "y_diff", "z_diff", "lateral_speed"
]
```

## Model

LightGBM with these params:
```python
params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "device": "gpu",
    "verbose": -1,
}
callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
```

If GPU fails (device="gpu" error), fall back to device="cpu".

## Validation

GroupKFold(n_splits=5) by well_id. NEVER mix rows from the same well across folds.

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(train_df))
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=well_ids)):
    ...
    oof[val_idx] = model.predict(X_val)
    fold_rmse = np.sqrt(np.mean((oof[val_idx] - y_val)**2))
    print(f"Fold {fold+1}: RMSE={fold_rmse:.4f}")
cv_rmse = np.sqrt(np.mean((oof - y_train)**2))
print(f"CV RMSE: {cv_rmse:.4f}")
```

## Final Model & Test Predictions

After CV, retrain on ALL train data with best n_estimators from CV.
Predict test set. Save test predictions.

## Output Files (ALL REQUIRED)

1. `experiments/oof/oof_lgbm_exp_b001.npy` — shape (5092255,) float32
2. `experiments/test_preds/test_lgbm_exp_b001.npy` — shape (19221,) float32
3. `experiments/results/exp_b001.json`:
```json
{
  "experiment_id": "exp_b001",
  "model": "lightgbm",
  "cv_rmse": <float>,
  "cv_rmse_std": <float>,
  "features_used": [...],
  "n_features": <int>,
  "training_time_seconds": <float>,
  "notes": "LightGBM baseline with basic MD/XYZ/GR features"
}
```

## After completion, print:
```
EXPERIMENT COMPLETE: cv_rmse=X.XXXX
```

Write the script to `experiments/scripts/run_lgbm_baseline.py` and run it.
Use pandas (CPU) — cudf not required for baseline.
