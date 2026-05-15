# Phase 2: Baseline Models

Build a complete training pipeline for your assigned model using only
basic features. Goal: establish a CV score to beat.

## Basic Features to Use (Baseline Only)

```python
BASELINE_FEATURES = [
    # Trajectory
    "md",          # measured depth
    "tvd",         # true vertical depth
    "inclination",
    "azimuth",
    # GR basic
    "gr",                    # raw GR value
    "gr_rolling_mean_10",    # 10-sample rolling mean
    "gr_rolling_mean_30",    # 30-sample rolling mean
    "gr_rolling_std_10",     # 10-sample rolling std
    # Depth position
    "depth_from_heel",       # distance from start of lateral
    "depth_fraction",        # normalized 0-1 position along well
]
```

## Model-Specific Instructions

### LightGBM Baseline (exp_b001)
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
    "device": "gpu",      # DGX Spark GPU
    "verbose": -1,
}
# Use early stopping with 100 rounds
```

### XGBoost Baseline (exp_b002)
```python
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "device": "cuda",     # DGX Spark GPU
    "verbosity": 0,
}
```

### Neural Network Baseline (exp_b003)
```python
# Simple 3-layer MLP with PyTorch
# Architecture: input -> 256 -> 128 -> 64 -> 1
# Activation: ReLU
# Dropout: 0.2
# Optimizer: Adam lr=1e-3
# Epochs: 100 with early stopping patience=10
# BatchNorm between layers
# device = torch.device("cuda")
```

## Pipeline Structure

```python
# 1. Load and merge all wells into a single DataFrame
# 2. Engineer basic features listed above
# 3. GroupKFold(n_splits=5) by well_id
# 4. For each fold: train, predict OOF
# 5. Save OOF to experiments/oof/oof_{model}_exp_b00X.npy
# 6. Train final model on all train data
# 7. Predict test set
# 8. Save test preds to experiments/test_preds/test_{model}_exp_b00X.npy
# 9. Save result JSON
```

## Success Criteria

A working baseline has:
- No data leakage (GroupKFold by well_id)
- OOF file with shape (n_train_rows,)
- Test pred file with shape (n_test_rows,)
- CV RMSE reported per fold and overall
- Training time under 10 minutes
