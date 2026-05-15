"""
Evaluation Utilities
====================
CV scoring, OOF management, and submission creation.
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def run_cv(
    model_fn,
    X,
    y,
    groups,
    n_splits=5,
    experiment_id="exp_unknown",
    save_oof=True,
    save_test=True,
    X_test=None,
):
    """
    Run GroupKFold CV and return OOF predictions + test predictions.

    Args:
        model_fn: Callable that returns a (model, train_fn, predict_fn) tuple
                  OR a sklearn-style object with fit/predict
        X: Feature array (numpy)
        y: Target array (numpy)
        groups: Well ID array for GroupKFold
        n_splits: Number of folds
        experiment_id: For file naming
        save_oof: Whether to save OOF to disk
        save_test: Whether to save test predictions to disk
        X_test: Test features (optional)

    Returns:
        dict with oof_preds, test_preds, cv_rmse, cv_rmse_std, fold_scores
    """
    oof_preds = np.zeros(len(y), dtype=np.float32)
    fold_scores = []
    test_preds_list = []

    gkf = GroupKFold(n_splits=n_splits)
    start = time.time()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_fn()
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        fold_rmse = rmse(y_val, val_preds)
        fold_scores.append(fold_rmse)
        print(f"  Fold {fold + 1}/{n_splits}: RMSE = {fold_rmse:.4f}")

        if X_test is not None:
            test_preds_list.append(model.predict(X_test))

    elapsed = time.time() - start
    cv_rmse = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    overall_oof_rmse = rmse(y, oof_preds)

    print(f"\nCV RMSE: {cv_rmse:.4f} ± {cv_std:.4f}")
    print(f"OOF RMSE: {overall_oof_rmse:.4f}")
    print(f"Training time: {elapsed:.0f}s")

    result = {
        "oof_preds": oof_preds,
        "fold_scores": fold_scores,
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": float(cv_std),
        "oof_rmse": float(overall_oof_rmse),
        "training_time_seconds": float(elapsed),
    }

    if X_test is not None and test_preds_list:
        test_preds = np.mean(test_preds_list, axis=0).astype(np.float32)
        result["test_preds"] = test_preds

    return result


def save_experiment_result(
    experiment_id: str,
    cv_rmse: float,
    cv_rmse_std: float,
    model: str,
    features_used: list,
    oof_preds: np.ndarray,
    test_preds: np.ndarray,
    training_time_seconds: float,
    notes: str = "",
    phase: str = "feature_engineering",
    base_experiment: str = None,
):
    """Save all outputs for a completed experiment."""
    root = Path(__file__).parent.parent

    # Save OOF
    oof_path = root / "experiments" / "oof" / f"oof_{model}_{experiment_id}.npy"
    np.save(str(oof_path), oof_preds)
    print(f"Saved OOF: {oof_path}")

    # Save test preds
    test_path = root / "experiments" / "test_preds" / f"test_{model}_{experiment_id}.npy"
    np.save(str(test_path), test_preds)
    print(f"Saved test preds: {test_path}")

    # Save result JSON
    result = {
        "experiment_id": experiment_id,
        "model": model,
        "phase": phase,
        "cv_rmse": round(cv_rmse, 4),
        "cv_rmse_std": round(cv_rmse_std, 4),
        "features_used": features_used,
        "n_features": len(features_used),
        "training_time_seconds": round(training_time_seconds, 1),
        "notes": notes,
        "base_experiment": base_experiment or "",
        "oof_path": str(oof_path.relative_to(root)),
        "test_path": str(test_path.relative_to(root)),
    }

    result_path = root / "experiments" / "results" / f"{experiment_id}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result: {result_path}")

    # Auto-log to tracker
    try:
        from agents.experiment_tracker import log_from_result_file
        log_from_result_file(str(result_path))
    except Exception as e:
        print(f"WARNING: Could not auto-log to tracker: {e}")

    print(f"\nEXPERIMENT COMPLETE: cv_rmse={cv_rmse:.4f}")
    return result
