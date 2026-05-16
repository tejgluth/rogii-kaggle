#!/usr/bin/env python3
"""Tuned LightGBM exp013 on the exact exp009 feature matrix."""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import cudf as gpu_pd  # noqa: F401
    import cuml  # noqa: F401

    GPU = True
except ImportError:
    import pandas as pd

    GPU = False
    print("WARNING: cuDF not available, using CPU pandas")
else:
    import pandas as pd

import lightgbm as lgb
import numpy as np
import xgboost as xgb  # noqa: F401
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features import (  # noqa: E402
    EXP009_FEATURES,
    EXP009_GEOLOGY_CATEGORICAL_FEATURES,
    add_fold_safe_target_encoding,
    build_exp009_feature_matrix,
)
from src.postprocess import postprocess_per_well  # noqa: E402


EXPERIMENT_ID = "exp013"
BASE_EXPERIMENT = "exp009"
EXP009_RMSE = 34.58112335205078

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
OOF_DIR = REPO_ROOT / "experiments" / "oof"
TEST_PREDS_DIR = REPO_ROOT / "experiments" / "test_preds"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

OOF_PATH = OOF_DIR / "oof_lightgbm_exp013.npy"
TEST_PREDS_PATH = TEST_PREDS_DIR / "test_lightgbm_exp013.npy"
RESULT_PATH = RESULTS_DIR / "exp013.json"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def well_id_from_path(path: Path) -> str:
    return path.name.split("__", maxsplit=1)[0]


def load_horizontal_wells(data_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if not paths:
        raise FileNotFoundError(f"No horizontal well CSVs found in {data_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df["well_id"] = well_id_from_path(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Loaded {split}: {len(paths)} wells, {len(out):,} rows, {out.shape[1]} columns")
    return out


def load_typewell_files(data_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*__typewell.csv"))
    if not paths:
        raise FileNotFoundError(f"No typewell CSVs found in {data_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df["well_id"] = well_id_from_path(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Loaded {split} typewells: {len(paths)} wells, {len(out):,} rows")
    return out


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def encode_geology_categories(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    category_maps: dict[str, list[str]] = {}
    for column in EXP009_GEOLOGY_CATEGORICAL_FEATURES:
        train_values = train_df[column].fillna("UNKNOWN").astype(str)
        test_values = test_df[column].fillna("UNKNOWN").astype(str)
        categories = pd.Index(train_values.unique()).union(pd.Index(test_values.unique()))
        category_maps[column] = [str(value) for value in categories.tolist()]
        train_df[column] = pd.Categorical(train_values, categories=categories)
        test_df[column] = pd.Categorical(test_values, categories=categories)
    return train_df, test_df, category_maps


def fit_lgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    params: dict[str, Any],
    callbacks: list[Any] | None = None,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    fit_kwargs: dict[str, Any] = {
        "categorical_feature": EXP009_GEOLOGY_CATEGORICAL_FEATURES,
    }
    if callbacks is not None:
        fit_kwargs["callbacks"] = callbacks
    if x_valid is None or y_valid is None:
        model.fit(x_train, y_train, **fit_kwargs)
    else:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], **fit_kwargs)
    return model


def write_experiment_log(result: dict[str, Any]) -> None:
    log_path = REPO_ROOT / "experiments" / "experiment_log.csv"
    row = pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "phase": "feature_engineering",
                "model": "lightgbm",
                "cv_rmse": result["cv_rmse"],
                "cv_rmse_std": result["cv_rmse_std"],
                "n_features": result["n_features"],
                "base_experiment": BASE_EXPERIMENT,
                "description": "Tuned LightGBM on the exact exp009 feature matrix.",
                "notes": result["notes"],
                "oof_path": str(OOF_PATH.relative_to(REPO_ROOT)),
                "test_path": str(TEST_PREDS_PATH.relative_to(REPO_ROOT)),
                "training_time_seconds": result["training_time_seconds"],
            }
        ]
    )
    if log_path.exists():
        existing = pd.read_csv(log_path)
        existing = existing[existing["experiment_id"] != EXPERIMENT_ID]
        row = pd.concat([existing, row], ignore_index=True, sort=False)
    row.to_csv(log_path, index=False)


def main() -> None:
    started = time.time()
    for directory in [OOF_DIR, TEST_PREDS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,
        "n_estimators": 8000,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "max_depth": -1,
        "min_gain_to_split": 0.0,
        "device": "gpu",
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    train_df = load_horizontal_wells(TRAIN_DIR, "train")
    test_df = load_horizontal_wells(TEST_DIR, "test")
    train_typewell = load_typewell_files(TRAIN_DIR, "train")
    test_typewell = load_typewell_files(TEST_DIR, "test")

    train_df, global_gr_median, global_tvt_input_median = build_exp009_feature_matrix(
        train_df,
        train_typewell,
    )
    test_df, _, _ = build_exp009_feature_matrix(
        test_df,
        test_typewell,
        global_gr_median,
        global_tvt_input_median,
    )

    train_df["TVT"] = pd.to_numeric(train_df["TVT"], errors="coerce")
    valid_target = train_df["TVT"].notna()
    if not valid_target.all():
        dropped = int((~valid_target).sum())
        print(f"Dropping {dropped:,} train rows with missing TVT")
        train_df = train_df.loc[valid_target].reset_index(drop=True)

    y_train = train_df["TVT"].to_numpy(dtype=np.float32)
    well_ids = train_df["well_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    fold_indices = list(gkf.split(train_df, y_train, groups=well_ids))
    train_df, test_df = add_fold_safe_target_encoding(
        train_df,
        test_df,
        target_col="TVT",
        category_col="tw_geology_at_aligned_tvt",
        output_col="tw_geology_target_enc",
        folds=fold_indices,
        smoothing=20.0,
    )
    train_df["tw_geology_target_enc"] = train_df["tw_geology_target_enc"].astype("float32")
    test_df["tw_geology_target_enc"] = test_df["tw_geology_target_enc"].astype("float32")
    train_df, test_df, category_maps = encode_geology_categories(train_df, test_df)

    x_train = train_df[EXP009_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    x_test = test_df[EXP009_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Training matrix: {x_train.shape}, test matrix: {x_test.shape}")
    print(f"Feature matrix exactly reuses {BASE_EXPERIMENT}: {EXP009_FEATURES}")
    print(f"Native categorical geology columns: {EXP009_GEOLOGY_CATEGORICAL_FEATURES}")

    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_scores: list[float] = []
    best_iterations: list[int] = []

    for fold, (tr_idx, val_idx) in enumerate(fold_indices, start=1):
        print(f"\nFold {fold}/5 with LightGBM device={params['device']}")
        callbacks = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)]
        model = fit_lgbm(
            x_train.iloc[tr_idx],
            y_train[tr_idx],
            x_train.iloc[val_idx],
            y_train[val_idx],
            params,
            callbacks,
        )
        preds = model.predict(
            x_train.iloc[val_idx],
            num_iteration=model.best_iteration_ if model.best_iteration_ else None,
        ).astype(np.float32)
        oof[val_idx] = preds
        fold_rmse = rmse(y_train[val_idx], preds)
        fold_scores.append(fold_rmse)
        best_iteration = int(model.best_iteration_ or params["n_estimators"])
        best_iterations.append(best_iteration)
        print(f"Fold {fold}: RMSE={fold_rmse:.4f}, best_iteration={best_iteration}")

        del model, preds
        gc.collect()

    cv_rmse = rmse(y_train, oof)
    cv_rmse_std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    final_n_estimators = max(1, int(round(float(np.mean(best_iterations)))))
    print(f"\nCV RMSE: {cv_rmse:.4f}")
    print(f"Fold RMSE std: {cv_rmse_std:.4f}")
    print(f"Final n_estimators from CV mean best_iteration: {final_n_estimators}")

    final_params = dict(params)
    final_params["n_estimators"] = final_n_estimators
    final_model = fit_lgbm(
        x_train,
        y_train,
        None,
        None,
        final_params,
        callbacks=[lgb.log_evaluation(200)],
    )
    raw_test_preds = final_model.predict(x_test).astype(np.float32)
    test_preds = postprocess_per_well(
        raw_test_preds,
        test_df["well_id"].to_numpy(),
        smooth=True,
        clip=True,
    ).astype(np.float32)

    np.save(OOF_PATH, oof.astype(np.float32))
    np.save(TEST_PREDS_PATH, test_preds.astype(np.float32))

    delta_vs_exp009 = cv_rmse - EXP009_RMSE
    notes = (
        f"Used the exact exp009 feature matrix with no added or removed features, "
        f"then trained one tuned LightGBM configuration with lower learning rate, "
        f"8000 max trees, stronger L1/L2 regularization, smaller leaves, row/feature "
        f"subsampling, 5-fold GroupKFold by well_id, and native geology categorical "
        f"features. Test predictions were smoothed and clipped per well. Compared "
        f"with exp009 RMSE {EXP009_RMSE:.4f}, exp013 "
        f"{'improved by' if delta_vs_exp009 < 0 else 'trailed by'} {abs(delta_vs_exp009):.4f} "
        f"RMSE. This hyperparameter-only change helps if exp009 was overfitting local "
        f"DTW/geology noise; it can hurt if the stronger regularization underfits "
        f"well-specific stratigraphic shifts."
    )

    training_time = time.time() - started
    result = {
        "experiment_id": EXPERIMENT_ID,
        "model": "lightgbm",
        "phase": "feature_engineering",
        "base_experiment": BASE_EXPERIMENT,
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": cv_rmse_std,
        "fold_rmse": fold_scores,
        "features_used": EXP009_FEATURES,
        "categorical_features": EXP009_GEOLOGY_CATEGORICAL_FEATURES,
        "category_maps": category_maps,
        "n_features": len(EXP009_FEATURES),
        "training_time_seconds": float(training_time),
        "notes": notes,
        "oof_path": str(OOF_PATH.relative_to(REPO_ROOT)),
        "test_path": str(TEST_PREDS_PATH.relative_to(REPO_ROOT)),
        "feature_params": {
            "source_feature_set": "exp009",
            "dtw_max_lag": 40,
            "dtw_lag_step": 2,
            "dtw_window": 41,
            "dtw_sakoe_chiba_radius": 5,
            "dtw_sample_step": 5,
            "target_encoding_smoothing": 20.0,
        },
        "model_params": {
            **params,
            "early_stopping_rounds": 200,
            "final_n_estimators": final_n_estimators,
            "actual_device": params["device"],
        },
        "postprocessing": {
            "test_predictions": "postprocess_per_well",
            "smooth": True,
            "clip": True,
        },
        "gpu_feature_loading_available": GPU,
    }
    RESULT_PATH.write_text(json.dumps(jsonable(result), indent=2) + "\n")
    write_experiment_log(result)

    if oof.shape != (len(train_df),):
        raise ValueError(f"Unexpected OOF shape: {oof.shape}, expected {(len(train_df),)}")
    if test_preds.shape != (len(test_df),):
        raise ValueError(
            f"Unexpected test prediction shape: {test_preds.shape}, expected {(len(test_df),)}"
        )
    for path in [OOF_PATH, TEST_PREDS_PATH, RESULT_PATH]:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Required output missing or empty: {path}")

    print(f"Saved OOF: {OOF_PATH} shape={oof.shape} dtype={oof.dtype}")
    print(f"Saved test predictions: {TEST_PREDS_PATH} shape={test_preds.shape} dtype={test_preds.dtype}")
    print(f"Saved result JSON: {RESULT_PATH}")
    print(f"EXPERIMENT COMPLETE: cv_rmse={cv_rmse:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"EXPERIMENT FAILED: {exc}")
        raise
