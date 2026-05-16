#!/usr/bin/env python3
"""LightGBM exp008 with normalized multi-scale Typewell xcorr features."""

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
    DATA_ROOT,
    FEATURES as BASE_FEATURES,
    TEST_DIR,
    TRAIN_DIR,
    add_baseline_features,
    fit_lgbm,
    fit_with_gpu_fallback,
    jsonable,
    load_horizontal_wells,
    rmse,
    well_id_from_path,
)
from src.features import add_exp008_typewell_xcorr_features  # noqa: E402


EXPERIMENT_ID = "exp008"

REPO_ROOT = Path(__file__).resolve().parents[2]
OOF_DIR = REPO_ROOT / "experiments" / "oof"
TEST_PREDS_DIR = REPO_ROOT / "experiments" / "test_preds"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

OOF_PATH = OOF_DIR / "oof_lightgbm_exp008.npy"
TEST_PREDS_PATH = TEST_PREDS_DIR / "test_lightgbm_exp008.npy"
RESULT_PATH = RESULTS_DIR / "exp008.json"

EXP008_FEATURES = [
    "tw_xcorr_best_lag_w21",
    "tw_xcorr_best_corr_w21",
    "tw_xcorr_best_lag_w51",
    "tw_xcorr_best_corr_w51",
    "tw_xcorr_best_lag_w101",
    "tw_xcorr_best_corr_w101",
    "tw_xcorr_lag_agreement",
    "tw_xcorr_lag_consensus_smooth_50",
    "tw_norm_gr_residual_at_best_lag",
    "tw_typewell_gr_zscore_at_best_lag",
]
FEATURES = BASE_FEATURES + EXP008_FEATURES


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
    df = add_exp008_typewell_xcorr_features(
        df,
        typewell_df,
        max_lag=80,
        windows=(21, 51, 101),
    )
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return df, global_gr_median, global_tvt_input_median


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
                "base_experiment": "exp_b001",
                "description": "Robust normalized Typewell xcorr at windows 21, 51, 101 and lags [-80, 80].",
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
    exp006_rmse = 36.39430618286133
    base_delta = baseline_rmse - cv_rmse
    exp006_delta = exp006_rmse - cv_rmse
    notes = (
        f"Added robust per-well/typewell z-scored Pearson Typewell alignment features "
        f"at windows 21, 51, and 101 over TVT lags [-80, 80]. The median lag across "
        f"scales was used for smooth consensus and normalized GR residual features. "
        f"Compared with exp_b001 RMSE {baseline_rmse:.4f}, exp008 "
        f"{'improved' if base_delta > 0 else 'did not improve'} by {abs(base_delta):.4f}; "
        f"against exp006 target {exp006_rmse:.4f}, it "
        f"{'improved' if exp006_delta > 0 else 'trailed'} by {abs(exp006_delta):.4f}. "
        f"Normalization should help when train/test GR scale differs; multi-scale lags "
        f"help only if local GR motifs have a stable Typewell correspondence."
    )
    training_time = time.time() - started
    result = {
        "experiment_id": EXPERIMENT_ID,
        "model": "lightgbm",
        "phase": "feature_engineering",
        "base_experiment": "exp_b001",
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": cv_rmse_std,
        "fold_rmse": fold_scores,
        "features_used": FEATURES,
        "n_features": len(FEATURES),
        "training_time_seconds": float(training_time),
        "notes": notes,
        "oof_path": str(OOF_PATH.relative_to(REPO_ROOT)),
        "test_path": str(TEST_PREDS_PATH.relative_to(REPO_ROOT)),
        "feature_params": {
            "max_lag": 80,
            "windows": [21, 51, 101],
            "normalization": "per well/typewell robust zscore median and MAD*1.4826",
            "lag_agreement": "std of best lags across windows",
            "lag_consensus": "centered rolling mean 50 of median best lag",
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
