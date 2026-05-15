# Phase 4: Model Stacking and Ensemble

You have a collection of OOF and test prediction files. Your job is to combine
them into a stronger final prediction.

## Step 1: Hill Climbing Ensemble

Start with the best single model. Greedily add models that improve CV RMSE.

```python
import numpy as np
from pathlib import Path

def hill_climbing_ensemble(oof_paths, test_paths, y_train, n_iterations=200):
    """Greedy ensemble by hill climbing on CV RMSE."""
    from sklearn.metrics import mean_squared_error

    # Load all OOF predictions
    oofs = {p: np.load(p) for p in oof_paths}
    tests = {p: np.load(p) for p in test_paths}

    # Start with best single model
    best_rmse = float("inf")
    for path, oof in oofs.items():
        rmse = np.sqrt(mean_squared_error(y_train, oof))
        if rmse < best_rmse:
            best_rmse = rmse
            selected = [path]

    print(f"Starting RMSE: {best_rmse:.4f} with {selected[0]}")

    # Greedily add models
    for _ in range(n_iterations):
        improved = False
        for path in oof_paths:
            if path in selected:
                continue
            candidate = selected + [path]
            ensemble_oof = np.mean([oofs[p] for p in candidate], axis=0)
            rmse = np.sqrt(mean_squared_error(y_train, ensemble_oof))
            if rmse < best_rmse:
                best_rmse = rmse
                selected = candidate
                improved = True
                print(f"  Added {Path(path).stem}: RMSE={rmse:.4f}")
        if not improved:
            break

    # Final ensemble predictions
    final_oof = np.mean([oofs[p] for p in selected], axis=0)
    final_test = np.mean([tests[p] for p in selected], axis=0)
    return final_oof, final_test, best_rmse, selected
```

## Step 2: Stacking Meta-Learner

Use OOF predictions as features for a meta-model.

```python
# Stack OOF predictions as features
X_meta_train = np.column_stack([np.load(p) for p in oof_paths])
X_meta_test = np.column_stack([np.load(p) for p in test_paths])

# Try multiple meta-learners
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# Ridge regression stacker
ridge = Ridge(alpha=1.0)
# GroupKFold CV to evaluate stacker (use same well_id groups)

# LightGBM stacker
stacker_params = {
    "objective": "regression", "metric": "rmse",
    "n_estimators": 500, "learning_rate": 0.05,
    "num_leaves": 15, "device": "gpu",
}
```

## Step 3: Knowledge Distillation (Optional, if time allows)

Train a single strong model using all OOF predictions as soft targets.

```python
# Create pseudo-labels from ensemble
pseudo_labels = final_oof  # or weighted average of top models
# Retrain LightGBM on full train + pseudo-label loss
# This can sometimes beat the ensemble
```

## Output Required

Save to `experiments/results/stacking_result.json`:
```json
{
  "method": "hill_climbing + ridge_stacker",
  "n_models_selected": 12,
  "cv_rmse_ensemble": 9.84,
  "cv_rmse_best_single": 10.43,
  "improvement_pct": 5.6,
  "selected_models": ["oof_lgbm_exp042.npy", "..."],
  "final_test_pred_path": "experiments/test_preds/final_ensemble.npy"
}
```

Save final test predictions to `experiments/test_preds/final_ensemble.npy`.
