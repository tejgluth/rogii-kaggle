#!/usr/bin/env python3
"""LightGBM exp003 with Typewell GR correlation features."""

from __future__ import annotations

import gc
import json
import time
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
    DATA_ROOT,
    FEATURES as BASE_FEATURES,
    TEST_DIR,
    TRAIN_DIR,
    fit_lgbm,
    fit_with_gpu_fallback,
    jsonable,
    load_horizontal_wells,
    rmse,
    well_id_from_path,
)
from src.features import add_typewell_correlation_features  # noqa: E402


EXPERIMENT_ID = "exp003"

REPO_ROOT = Path(__file__).resolve().parents[2]
OOF_DIR = REPO_ROOT / "experiments" / "oof"
TEST_PREDS_DIR = REPO_ROOT / "experiments" / "test_preds"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

OOF_PATH = OOF_DIR / "oof_lightgbm_exp003.npy"
TEST_PREDS_PATH = TEST_PREDS_DIR / "test_lightgbm_exp003.npy"
RESULT_PATH = RESULTS_DIR / "exp003.json"

TYPEWELL_FEATURES = [
    "typewell_best_lag",
    "typewell_best_corr",
    "typewell_corr_at_lag_0",
    "typewell_corr_max_minus_mean",
    "typewell_corr_peakiness",
    "typewell_gr_at_best_lag",
    "typewell_gr_minus_local_gr_at_best_lag",
    "typewell_lag_smoothness_rollmean_50",
]
FEATURES = BASE_FEATURES + TYPEWELL_FEATURES


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


def add_baseline_features(
    df: pd.DataFrame,
    global_gr_median: float | None = None,
    global_tvt_input_median: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    df = df.sort_values(["well_id", "MD"], kind="mergesort").reset_index(drop=True)

    for column in ["MD", "X", "Y", "Z", "GR", "TVT_input"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if global_gr_median is None:
        global_gr_median = float(df["GR"].median())
    if global_tvt_input_median is None:
        global_tvt_input_median = float(df["TVT_input"].median())

    if np.isnan(global_gr_median):
        global_gr_median = 0.0
    if np.isnan(global_tvt_input_median):
        global_tvt_input_median = 0.0

    grouped = df.groupby("well_id", sort=False, group_keys=False)

    md_min = grouped["MD"].transform("min")
    md_max = grouped["MD"].transform("max")
    depth_range = (md_max - md_min).replace(0, np.nan)
    df["depth_from_heel"] = df["MD"] - md_min
    df["depth_fraction"] = (df["depth_from_heel"] / depth_range).fillna(0)

    gr_median = grouped["GR"].transform("median").fillna(global_gr_median)
    df["gr_filled"] = df["GR"].fillna(gr_median).fillna(global_gr_median)
    gr_grouped = df.groupby("well_id", sort=False)["gr_filled"]
    df["gr_rolling_mean_10"] = gr_grouped.transform(
        lambda s: s.rolling(10, center=True, min_periods=1).mean()
    )
    df["gr_rolling_mean_30"] = gr_grouped.transform(
        lambda s: s.rolling(30, center=True, min_periods=1).mean()
    )
    df["gr_rolling_std_10"] = gr_grouped.transform(
        lambda s: s.rolling(10, center=True, min_periods=1).std().fillna(0)
    )
    df["gr_rolling_mean_100"] = gr_grouped.transform(
        lambda s: s.rolling(100, center=True, min_periods=1).mean()
    )

    tvt_input_median = grouped["TVT_input"].transform("median").fillna(global_tvt_input_median)
    df["tvt_input_filled"] = df["TVT_input"].fillna(tvt_input_median).fillna(global_tvt_input_median)

    df["x_diff"] = grouped["X"].diff().fillna(0)
    df["y_diff"] = grouped["Y"].diff().fillna(0)
    df["z_diff"] = grouped["Z"].diff().fillna(0)
    df["lateral_speed"] = np.sqrt(df["x_diff"] ** 2 + df["y_diff"] ** 2 + df["z_diff"] ** 2)
    return df, global_gr_median, global_tvt_input_median


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
    df = add_typewell_correlation_features(df, typewell_df, max_lag=60, window=51)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return df, global_gr_median, global_tvt_input_median


def write_experiment_log(cv_rmse: float, notes: str) -> None:
    log_path = REPO_ROOT / "experiments" / "experiment_log.csv"
    row = pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "cv_rmse": cv_rmse,
                "model": "lightgbm",
                "phase": "feature_engineering",
                "notes": notes,
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

    x_train = train_df[FEATURES]
    y_train = train_df["TVT"].to_numpy(dtype=np.float32)
    well_ids = train_df["well_id"].to_numpy()
    x_test = test_df[FEATURES]

    print(f"Training matrix: {x_train.shape}, test matrix: {x_test.shape}")

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    active_params: dict[str, Any] = dict(BASE_PARAMS)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(x_train, y_train, groups=well_ids), start=1):
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
    final_model = fit_lgbm(x_train, y_train, None, None, final_params, callbacks=[lgb.log_evaluation(200)])
    test_preds = final_model.predict(x_test).astype(np.float32)

    np.save(OOF_PATH, oof.astype(np.float32))
    np.save(TEST_PREDS_PATH, test_preds.astype(np.float32))

    baseline_rmse = 37.11342239379883
    improvement = baseline_rmse - cv_rmse
    helped = improvement > 0
    notes = (
        f"Added per-well Typewell GR rolling correlation features over lags [-60, 60] "
        f"centered on TVT_input_filled. Compared with exp_b001 RMSE {baseline_rmse:.4f}, "
        f"exp003 {'improved' if helped else 'did not improve'} by {improvement:.4f}. "
        f"The features expose local GR alignment quality and smooth lag estimates, "
        f"which should help when typewell/lateral GR motifs are coherent; weak or noisy "
        f"correlation can hurt wells where TVT_input already dominates."
    )
    training_time = time.time() - started
    result = {
        "experiment_id": EXPERIMENT_ID,
        "model": "lightgbm",
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": cv_rmse_std,
        "features_used": FEATURES,
        "n_features": len(FEATURES),
        "training_time_seconds": float(training_time),
        "notes": notes,
    }
    RESULT_PATH.write_text(json.dumps(jsonable(result), indent=2) + "\n")
    write_experiment_log(cv_rmse, notes)

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
