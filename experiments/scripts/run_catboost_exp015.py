#!/usr/bin/env python3
"""Tuned CatBoost exp015 on the exact exp014 feature matrix."""

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

import lightgbm as lgb  # noqa: F401
import numpy as np
import xgboost as xgb  # noqa: F401
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features import (  # noqa: E402
    EXP009_FEATURES,
    EXP009_GEOLOGY_CATEGORICAL_FEATURES,
    build_exp009_feature_matrix,
)
from src.postprocess import postprocess_per_well  # noqa: E402


EXPERIMENT_ID = "exp015"
BASE_EXPERIMENT = "exp014"
EXP009_RMSE = 34.58112335205078
EXP014_RMSE = 30.4541072845459

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
OOF_DIR = REPO_ROOT / "experiments" / "oof"
TEST_PREDS_DIR = REPO_ROOT / "experiments" / "test_preds"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

OOF_PATH = OOF_DIR / "oof_catboost_exp015.npy"
TEST_PREDS_PATH = TEST_PREDS_DIR / "test_catboost_exp015.npy"
RESULT_PATH = RESULTS_DIR / "exp015.json"

TARGET_ENCODING_COLUMN = "tw_geology_target_enc"
FEATURES = [feature for feature in EXP009_FEATURES if feature != TARGET_ENCODING_COLUMN]
NUMERIC_FEATURES = [
    feature for feature in FEATURES if feature not in EXP009_GEOLOGY_CATEGORICAL_FEATURES
]


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
    row_offset = 0
    for path in paths:
        df = pd.read_csv(path)
        df["well_id"] = well_id_from_path(path)
        df["__row_order"] = np.arange(row_offset, row_offset + len(df), dtype=np.int64)
        row_offset += len(df)
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


def prepare_catboost_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df[FEATURES].copy()
    matrix[NUMERIC_FEATURES] = (
        matrix[NUMERIC_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    for column in EXP009_GEOLOGY_CATEGORICAL_FEATURES:
        matrix[column] = matrix[column].fillna("UNKNOWN").astype(str)
    return matrix


def make_pool(
    x: pd.DataFrame,
    y: np.ndarray | None = None,
) -> Pool:
    return Pool(data=x, label=y, cat_features=EXP009_GEOLOGY_CATEGORICAL_FEATURES)


def make_model(params: dict[str, Any]) -> CatBoostRegressor:
    return CatBoostRegressor(**params)


def write_experiment_log(result: dict[str, Any]) -> None:
    log_path = REPO_ROOT / "experiments" / "experiment_log.csv"
    row = pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "phase": "feature_engineering",
                "model": "catboost",
                "cv_rmse": result["cv_rmse"],
                "cv_rmse_std": result["cv_rmse_std"],
                "n_features": result["n_features"],
                "base_experiment": BASE_EXPERIMENT,
                "description": "Tuned CatBoost trained on the exact exp014 feature matrix.",
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
        "task_type": "GPU",
        "iterations": 12000,
        "learning_rate": 0.03,
        "depth": 10,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "random_seed": 42,
        "od_type": "Iter",
        "od_wait": 300,
        "eval_metric": "RMSE",
        "loss_function": "RMSE",
        "grow_policy": "SymmetricTree",
        "allow_writing_files": False,
        "verbose": 200,
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

    train_df = train_df.sort_values("__row_order", kind="mergesort").reset_index(drop=True)
    test_df = test_df.sort_values("__row_order", kind="mergesort").reset_index(drop=True)

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

    x_train = prepare_catboost_matrix(train_df)
    x_test = prepare_catboost_matrix(test_df)
    test_pool = make_pool(x_test)

    print(f"Training matrix: {x_train.shape}, test matrix: {x_test.shape}")
    print(f"Source feature set: {BASE_EXPERIMENT} exact matrix")
    print(f"Dropped duplicate target-encoded geology column: {TARGET_ENCODING_COLUMN}")
    print(f"Native CatBoost categorical features: {EXP009_GEOLOGY_CATEGORICAL_FEATURES}")

    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred_sum = np.zeros(len(test_df), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []

    for fold, (tr_idx, val_idx) in enumerate(fold_indices, start=1):
        print(f"\nFold {fold}/5 with CatBoost task_type={params['task_type']}")
        model = make_model(params)
        train_pool = make_pool(x_train.iloc[tr_idx], y_train[tr_idx])
        valid_pool = make_pool(x_train.iloc[val_idx], y_train[val_idx])
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        preds = model.predict(valid_pool).astype(np.float32)
        oof[val_idx] = preds
        test_pred_sum += model.predict(test_pool).astype(np.float64)

        fold_rmse = rmse(y_train[val_idx], preds)
        fold_scores.append(fold_rmse)
        best_iteration = int(model.get_best_iteration() or params["iterations"])
        best_iterations.append(best_iteration)
        print(f"Fold {fold}: RMSE={fold_rmse:.4f}, best_iteration={best_iteration}")

        del model, train_pool, valid_pool, preds
        gc.collect()

    raw_test_preds = (test_pred_sum / len(fold_indices)).astype(np.float32)
    test_preds = postprocess_per_well(
        raw_test_preds,
        test_df["well_id"].to_numpy(),
        smooth=True,
        clip=True,
    ).astype(np.float32)

    cv_rmse = rmse(y_train, oof)
    cv_rmse_std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    print(f"\nCV RMSE: {cv_rmse:.4f}")
    print(f"Fold RMSE std: {cv_rmse_std:.4f}")

    np.save(OOF_PATH, oof.astype(np.float32))
    np.save(TEST_PREDS_PATH, test_preds.astype(np.float32))

    delta_vs_exp014 = cv_rmse - EXP014_RMSE
    delta_vs_exp009 = cv_rmse - EXP009_RMSE
    notes = (
        f"Used the exact exp014 feature matrix: exp009 DTW and Typewell geology features "
        f"with {TARGET_ENCODING_COLUMN} dropped and geology labels passed as native "
        f"CatBoost categoricals. No new features were added. Tuned CatBoost to depth=10, "
        f"learning_rate=0.03, 12000 max iterations, l2_leaf_reg=3.0, "
        f"bagging_temperature=0.5, random_strength=1.0, and od_wait=300. Compared with "
        f"exp014 RMSE {EXP014_RMSE:.4f}, exp015 "
        f"{'improved by' if delta_vs_exp014 < 0 else 'trailed by'} {abs(delta_vs_exp014):.4f} "
        f"RMSE. The tuning {'helped' if delta_vs_exp014 < 0 else 'did not help'} relative "
        f"to exp014; deeper trees and lighter regularization can capture more geology/DTW "
        f"interactions, but can also overfit well-specific alignment noise. Compared with "
        f"exp009 LightGBM RMSE {EXP009_RMSE:.4f}, this CatBoost run "
        f"{'improved by' if delta_vs_exp009 < 0 else 'trailed by'} {abs(delta_vs_exp009):.4f} RMSE."
    )

    training_time = time.time() - started
    result = {
        "experiment_id": EXPERIMENT_ID,
        "model": "catboost",
        "phase": "feature_engineering",
        "base_experiment": BASE_EXPERIMENT,
        "description": "Tuned CatBoost on exp014 features with no feature additions.",
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": cv_rmse_std,
        "fold_rmse": fold_scores,
        "features_used": FEATURES,
        "dropped_features": [TARGET_ENCODING_COLUMN],
        "categorical_features": EXP009_GEOLOGY_CATEGORICAL_FEATURES,
        "n_features": len(FEATURES),
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
            "target_encoding_column_dropped": TARGET_ENCODING_COLUMN,
        },
        "model_params": {
            **params,
            "fold_best_iterations": best_iterations,
            "test_prediction_strategy": "mean_of_5_fold_models",
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
