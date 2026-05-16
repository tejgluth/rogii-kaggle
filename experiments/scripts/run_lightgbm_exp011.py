#!/usr/bin/env python3
"""LightGBM exp011 with deeper Typewell geology context features."""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.run_lgbm_baseline import (  # noqa: E402
    BASE_PARAMS,
    FEATURES as BASE_FEATURES,
    TEST_DIR,
    TRAIN_DIR,
    add_baseline_features,
    jsonable,
    load_horizontal_wells,
    rmse,
    well_id_from_path,
)
from src.features import (  # noqa: E402
    add_dtw_lag_features,
    add_exp011_typewell_geology_context_features,
    add_fold_safe_target_encoding,
    add_typewell_geology_features,
)


EXPERIMENT_ID = "exp011"

REPO_ROOT = Path(__file__).resolve().parents[2]
OOF_DIR = REPO_ROOT / "experiments" / "oof"
TEST_PREDS_DIR = REPO_ROOT / "experiments" / "test_preds"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

OOF_PATH = OOF_DIR / "oof_lightgbm_exp011.npy"
TEST_PREDS_PATH = TEST_PREDS_DIR / "test_lightgbm_exp011.npy"
RESULT_PATH = RESULTS_DIR / "exp011.json"

DTW_FEATURES = [
    "dtw_best_lag",
    "dtw_min_distance",
    "dtw_best_lag_rollmean_50",
    "dtw_confidence",
]
GEOLOGY_CATEGORICAL_FEATURES = [
    "tw_geology_at_aligned_tvt",
    "tw_geology_above_layer",
    "tw_geology_below_layer",
    "tw_geology_lag_+5",
    "tw_geology_lag_-5",
    "tw_geology_lag_+15",
    "tw_geology_lag_-15",
]
GEOLOGY_NUMERIC_FEATURES = [
    "tw_geology_target_enc",
    "tw_geology_layer_thickness",
    "tw_geology_distance_to_layer_top",
    "tw_geology_distance_to_layer_bottom",
    "tw_geology_layer_index",
    "tw_geology_layer_index_from_bottom",
    "tw_geology_n_layers_total",
    "tw_layer_position_fraction",
    "tw_layer_top_tvt",
    "tw_layer_bottom_tvt",
    "tw_geology_layer_mean_gr",
    "tw_geology_layer_std_gr",
    "gr_minus_layer_mean_gr",
    "gr_minus_layer_mean_gr_zscore",
    "tw_distinct_geologies_in_window_50",
]
FEATURES = BASE_FEATURES + DTW_FEATURES + GEOLOGY_CATEGORICAL_FEATURES + GEOLOGY_NUMERIC_FEATURES


def fit_lgbm_categorical(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    params: dict[str, Any],
    callbacks: list[Any] | None = None,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    fit_kwargs: dict[str, Any] = {
        "callbacks": callbacks,
        "categorical_feature": GEOLOGY_CATEGORICAL_FEATURES,
    }
    if x_valid is None or y_valid is None:
        model.fit(x_train, y_train, **fit_kwargs)
    else:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], **fit_kwargs)
    return model


def fit_with_gpu_fallback_categorical(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    params: dict[str, Any],
    callbacks: list[Any] | None,
) -> tuple[lgb.LGBMRegressor, dict[str, Any]]:
    try:
        model = fit_lgbm_categorical(x_train, y_train, x_valid, y_valid, params, callbacks)
        return model, params
    except Exception as exc:
        if params.get("device") != "gpu":
            raise
        print(f"GPU training failed; falling back to CPU. Error: {exc}")
        cpu_params = dict(params)
        cpu_params["device"] = "cpu"
        model = fit_lgbm_categorical(x_train, y_train, x_valid, y_valid, cpu_params, callbacks)
        return model, cpu_params


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


def prepare_features(
    df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    global_gr_median: float | None = None,
    global_tvt_input_median: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    df, global_gr_median, global_tvt_input_median = add_baseline_features(
        df,
        global_gr_median,
        global_tvt_input_median,
    )
    df = add_dtw_lag_features(
        df,
        typewell_df,
        max_lag=40,
        lag_step=2,
        window=41,
        radius=5,
        sample_step=5,
    )
    df = add_typewell_geology_features(df, typewell_df)
    df = add_exp011_typewell_geology_context_features(df, typewell_df)

    numeric_features = BASE_FEATURES + DTW_FEATURES + [
        "tw_geology_layer_thickness",
        "tw_geology_distance_to_layer_top",
        "tw_geology_distance_to_layer_bottom",
        "tw_geology_layer_index",
        "tw_geology_layer_index_from_bottom",
        "tw_geology_n_layers_total",
        "tw_layer_position_fraction",
        "tw_layer_top_tvt",
        "tw_layer_bottom_tvt",
        "tw_geology_layer_mean_gr",
        "tw_geology_layer_std_gr",
        "gr_minus_layer_mean_gr",
        "gr_minus_layer_mean_gr_zscore",
        "tw_distinct_geologies_in_window_50",
    ]
    df[numeric_features] = df[numeric_features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    for column in GEOLOGY_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("UNKNOWN").astype("category")
    return df, global_gr_median, global_tvt_input_median


def encode_categories(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for column in GEOLOGY_CATEGORICAL_FEATURES:
        categories = pd.Index(train_df[column].astype(str).unique()).union(
            pd.Index(test_df[column].astype(str).unique())
        )
        train_df[column] = pd.Categorical(train_df[column].astype(str), categories=categories)
        test_df[column] = pd.Categorical(test_df[column].astype(str), categories=categories)
    return train_df, test_df


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
                "base_experiment": "exp009",
                "description": "Deeper Typewell geology context around exp009 DTW-aligned TVT.",
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

    train_df = load_horizontal_wells(TRAIN_DIR, "train")
    test_df = load_horizontal_wells(TEST_DIR, "test")
    train_typewell = load_typewell_files(TRAIN_DIR, "train")
    test_typewell = load_typewell_files(TEST_DIR, "test")

    train_df, global_gr_median, global_tvt_input_median = prepare_features(train_df, train_typewell)
    test_df, _, _ = prepare_features(test_df, test_typewell, global_gr_median, global_tvt_input_median)

    train_df["TVT"] = pd.to_numeric(train_df["TVT"], errors="coerce")
    valid_target = train_df["TVT"].notna()
    if not valid_target.all():
        dropped = int((~valid_target).sum())
        print(f"Dropping {dropped:,} train rows with missing TVT")
        train_df = train_df.loc[valid_target].reset_index(drop=True)

    train_df, test_df = encode_categories(train_df, test_df)

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

    x_train = train_df[FEATURES]
    x_test = test_df[FEATURES]

    print(f"Training matrix: {x_train.shape}, test matrix: {x_test.shape}")
    print(f"Categorical features: {GEOLOGY_CATEGORICAL_FEATURES}")

    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    active_params: dict[str, Any] = dict(BASE_PARAMS)

    for fold, (tr_idx, val_idx) in enumerate(fold_indices, start=1):
        print(f"\nFold {fold}/5")
        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
        model, active_params = fit_with_gpu_fallback_categorical(
            x_train.iloc[tr_idx],
            y_train[tr_idx],
            x_train.iloc[val_idx],
            y_train[val_idx],
            active_params,
            callbacks,
        )
        preds = model.predict(
            x_train.iloc[val_idx],
            num_iteration=model.best_iteration_ if model.best_iteration_ else None,
        ).astype(np.float32)
        oof[val_idx] = preds
        fold_rmse = rmse(y_train[val_idx], preds)
        fold_scores.append(fold_rmse)
        best_iter = int(model.best_iteration_ or active_params["n_estimators"])
        best_iterations.append(best_iter)
        print(f"Fold {fold}: RMSE={fold_rmse:.4f}, best_iteration={best_iter}")

        del model, preds
        gc.collect()

    cv_rmse = rmse(y_train, oof)
    cv_rmse_std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    final_n_estimators = max(1, int(round(float(np.mean(best_iterations)))))
    print(f"\nCV RMSE: {cv_rmse:.4f}")
    print(f"Fold RMSE std: {cv_rmse_std:.4f}")
    print(f"Final n_estimators from CV mean best_iteration: {final_n_estimators}")

    final_params = dict(active_params)
    final_params["n_estimators"] = final_n_estimators
    final_model = fit_lgbm_categorical(
        x_train,
        y_train,
        None,
        None,
        final_params,
        callbacks=[lgb.log_evaluation(200)],
    )
    test_preds = final_model.predict(x_test).astype(np.float32)

    np.save(OOF_PATH, oof.astype(np.float32))
    np.save(TEST_PREDS_PATH, test_preds.astype(np.float32))

    exp009_rmse = 34.58112335205078
    delta = exp009_rmse - cv_rmse
    notes = (
        f"Built on exp009 and added offset geology labels at +/-5 and +/-15 TVT, "
        f"contiguous typewell layer indexes, normalized within-layer position, layer "
        f"top/bottom TVT, typewell GR mean/std by aligned geology label, GR residual "
        f"against that layer mean, and a 50-TV window geology diversity count. "
        f"Compared with exp009 RMSE {exp009_rmse:.4f}, exp011 "
        f"{'improved' if delta > 0 else 'did not improve'} by {abs(delta):.4f} RMSE. "
        f"These features should help when the aligned layer context captures local "
        f"stratigraphic position; they can hurt if DTW alignment noise makes layer "
        f"indexes and residuals over-specific."
    )
    training_time = time.time() - started
    result = {
        "experiment_id": EXPERIMENT_ID,
        "model": "lightgbm",
        "phase": "feature_engineering",
        "base_experiment": "exp009",
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": cv_rmse_std,
        "fold_rmse": fold_scores,
        "features_used": FEATURES,
        "categorical_features": GEOLOGY_CATEGORICAL_FEATURES,
        "n_features": len(FEATURES),
        "training_time_seconds": float(training_time),
        "notes": notes,
        "oof_path": str(OOF_PATH.relative_to(REPO_ROOT)),
        "test_path": str(TEST_PREDS_PATH.relative_to(REPO_ROOT)),
        "feature_params": {
            "dtw_max_lag": 40,
            "dtw_lag_step": 2,
            "dtw_window": 41,
            "dtw_sakoe_chiba_radius": 5,
            "dtw_sample_step": 5,
            "target_encoding_smoothing": 20.0,
        },
    }
    RESULT_PATH.write_text(json.dumps(jsonable(result), indent=2) + "\n")
    write_experiment_log(result)

    if oof.shape != (len(train_df),):
        raise ValueError(f"Unexpected OOF shape: {oof.shape}, expected {(len(train_df),)}")
    if test_preds.shape != (len(test_df),):
        raise ValueError(f"Unexpected test prediction shape: {test_preds.shape}, expected {(len(test_df),)}")
    for path in [OOF_PATH, TEST_PREDS_PATH, RESULT_PATH]:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Required output missing or empty: {path}")

    print(f"Saved OOF: {OOF_PATH} shape={oof.shape} dtype={oof.dtype}")
    print(f"Saved test predictions: {TEST_PREDS_PATH} shape={test_preds.shape} dtype={test_preds.dtype}")
    print(f"Saved result JSON: {RESULT_PATH}")
    print(f"EXPERIMENT COMPLETE: cv_rmse={cv_rmse:.4f}")


if __name__ == "__main__":
    main()
