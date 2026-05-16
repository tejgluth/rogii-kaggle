#!/usr/bin/env python3
"""LightGBM exp005: baseline plus trajectory-derived geometry features."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import cudf as pd  # noqa: F401
    import cuml  # noqa: F401
    GPU = True
except ImportError:
    import pandas as pd  # noqa: F401
    GPU = False
    print("WARNING: cuDF not available, using CPU pandas")

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.run_lgbm_baseline import (  # noqa: E402
    BASE_PARAMS,
    DATA_ROOT,
    TEST_DIR,
    TRAIN_DIR,
    add_baseline_features,
    fit_lgbm,
    fit_with_gpu_fallback,
    jsonable,
    load_horizontal_wells,
    rmse,
)
from src.features import add_exp005_trajectory_features  # noqa: E402


EXPERIMENT_ID = "exp005"
BASE_EXPERIMENT = "exp_b001"
BASE_CV_RMSE = 37.11342239379883
DESCRIPTION = (
    "Trajectory-derived features: dog-leg severity proxies from XYZ deltas, "
    "Z gradient and curvature, cumulative lateral distance, XY displacement, "
    "inclination/azimuth proxies, and depth below heel."
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = REPO_ROOT / "experiments" / "oof" / "oof_lightgbm_exp005.npy"
TEST_PREDS_PATH = REPO_ROOT / "experiments" / "test_preds" / "test_lightgbm_exp005.npy"
RESULT_PATH = REPO_ROOT / "experiments" / "results" / "exp005.json"

BASE_FEATURES = [
    "MD",
    "X",
    "Y",
    "Z",
    "depth_from_heel",
    "depth_fraction",
    "gr_filled",
    "gr_rolling_mean_10",
    "gr_rolling_mean_30",
    "gr_rolling_std_10",
    "gr_rolling_mean_100",
    "tvt_input_filled",
    "x_diff",
    "y_diff",
    "z_diff",
    "lateral_speed",
]

EXP005_FEATURES = [
    "x_diff_5",
    "y_diff_5",
    "z_diff_5",
    "z_gradient_rollmean_30",
    "dls_xyz",
    "dls_xyz_rollmean_30",
    "lateral_dist_from_heel",
    "xy_dist_from_heel",
    "z_curvature",
    "md_step",
    "inclination_proxy",
    "azimuth_proxy",
    "depth_below_heel_z",
]

FEATURES = BASE_FEATURES + EXP005_FEATURES


def prepare_features(
    df: pd.DataFrame,
    global_gr_median: float | None = None,
    global_tvt_input_median: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    df, global_gr_median, global_tvt_input_median = add_baseline_features(
        df,
        global_gr_median=global_gr_median,
        global_tvt_input_median=global_tvt_input_median,
    )
    df = add_exp005_trajectory_features(df)
    df[FEATURES] = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    return df, global_gr_median, global_tvt_input_median


def write_result(
    cv_rmse: float,
    cv_rmse_std: float,
    fold_scores: list[float],
    best_iterations: list[int],
    training_time: float,
    active_params: dict[str, Any],
) -> None:
    delta = cv_rmse - BASE_CV_RMSE
    if delta < 0:
        helped = f"helped by {-delta:.4f} RMSE"
    else:
        helped = f"did not help; RMSE worsened by {delta:.4f}"

    notes = (
        f"Compared with {BASE_EXPERIMENT} CV RMSE {BASE_CV_RMSE:.4f}, "
        f"exp005 {helped}. The added trajectory features describe wellbore "
        "geometry and structural trend but do not add direct typewell alignment "
        "signal; improvements would indicate that geometric context explains "
        "systematic TVT drift beyond the baseline XYZ/MD features."
    )

    result = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "feature_engineering",
        "description": DESCRIPTION,
        "model": "lightgbm",
        "base_experiment": BASE_EXPERIMENT,
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": float(cv_rmse_std),
        "fold_rmse": [float(score) for score in fold_scores],
        "features_used": FEATURES,
        "features_added": EXP005_FEATURES,
        "n_features": len(FEATURES),
        "training_time_seconds": float(training_time),
        "best_iterations": [int(iteration) for iteration in best_iterations],
        "params": active_params,
        "notes": notes,
        "oof_path": str(OOF_PATH.relative_to(REPO_ROOT)),
        "test_path": str(TEST_PREDS_PATH.relative_to(REPO_ROOT)),
    }
    RESULT_PATH.write_text(json.dumps(jsonable(result), indent=2) + "\n")

    try:
        from agents.experiment_tracker import log_from_result_file

        log_from_result_file(str(RESULT_PATH))
    except Exception as exc:
        print(f"WARNING: Could not auto-log experiment: {exc}")


def verify_output(path: Path, expected_shape: tuple[int, ...]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty output: {path}")
    arr = np.load(path)
    if arr.shape != expected_shape:
        raise ValueError(f"Unexpected shape for {path}: {arr.shape}, expected {expected_shape}")
    if arr.dtype != np.float32:
        raise TypeError(f"Unexpected dtype for {path}: {arr.dtype}, expected float32")


def main() -> None:
    started = time.time()
    print(f"Data root: {DATA_ROOT}")
    for directory in [OOF_PATH.parent, TEST_PREDS_PATH.parent, RESULT_PATH.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    train_df = load_horizontal_wells(TRAIN_DIR, "train")
    test_df = load_horizontal_wells(TEST_DIR, "test")

    train_df, global_gr_median, global_tvt_input_median = prepare_features(train_df)
    test_df, _, _ = prepare_features(test_df, global_gr_median, global_tvt_input_median)

    train_df["TVT"] = pd.to_numeric(train_df["TVT"], errors="coerce")
    valid_target = train_df["TVT"].notna()
    if not valid_target.all():
        dropped = int((~valid_target).sum())
        print(f"Dropping {dropped:,} train rows with missing TVT")
        train_df = train_df.loc[valid_target].reset_index(drop=True)

    x_train = train_df[FEATURES]
    y_train = train_df["TVT"].to_numpy(dtype=np.float32)
    well_ids = train_df["well_id"].to_numpy()
    x_test = test_df[FEATURES]

    print(f"Training matrix: {x_train.shape}, test matrix: {x_test.shape}")
    print(f"Feature count: {len(FEATURES)}")

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    active_params = dict(BASE_PARAMS)

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(x_train, y_train, groups=well_ids),
        start=1,
    ):
        print(f"\nFold {fold}/5")
        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
        model, active_params = fit_with_gpu_fallback(
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
    final_model = fit_lgbm(
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

    training_time = time.time() - started
    write_result(
        cv_rmse,
        cv_rmse_std,
        fold_scores,
        best_iterations,
        training_time,
        active_params,
    )

    verify_output(OOF_PATH, (len(train_df),))
    verify_output(TEST_PREDS_PATH, (len(test_df),))
    if not RESULT_PATH.exists() or RESULT_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty output: {RESULT_PATH}")

    print(f"Saved OOF: {OOF_PATH} shape={oof.shape} dtype={oof.dtype}")
    print(f"Saved test predictions: {TEST_PREDS_PATH} shape={test_preds.shape} dtype={test_preds.dtype}")
    print(f"Saved result JSON: {RESULT_PATH}")
    print(f"EXPERIMENT COMPLETE: cv_rmse={cv_rmse:.4f}")


if __name__ == "__main__":
    main()
