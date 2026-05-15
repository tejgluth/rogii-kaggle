#!/usr/bin/env python3
"""LightGBM baseline for ROGII exp_b001."""

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


EXPERIMENT_ID = "exp_b001"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
OOF_DIR = REPO_ROOT / "experiments" / "oof"
TEST_PREDS_DIR = REPO_ROOT / "experiments" / "test_preds"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

OOF_PATH = OOF_DIR / "oof_lgbm_exp_b001.npy"
TEST_PREDS_PATH = TEST_PREDS_DIR / "test_lgbm_exp_b001.npy"
RESULT_PATH = RESULTS_DIR / "exp_b001.json"

FEATURES = [
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

BASE_PARAMS: dict[str, Any] = {
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
    "random_state": 42,
    "n_jobs": -1,
}


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

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return df, global_gr_median, global_tvt_input_median


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def fit_lgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    params: dict[str, Any],
    callbacks: list[Any] | None = None,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    if x_valid is None or y_valid is None:
        model.fit(x_train, y_train, callbacks=callbacks)
    else:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], callbacks=callbacks)
    return model


def fit_with_gpu_fallback(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    params: dict[str, Any],
    callbacks: list[Any] | None,
) -> tuple[lgb.LGBMRegressor, dict[str, Any]]:
    try:
        model = fit_lgbm(x_train, y_train, x_valid, y_valid, params, callbacks)
        return model, params
    except Exception as exc:
        if params.get("device") != "gpu":
            raise
        print(f"GPU training failed; falling back to CPU. Error: {exc}")
        cpu_params = dict(params)
        cpu_params["device"] = "cpu"
        model = fit_lgbm(x_train, y_train, x_valid, y_valid, cpu_params, callbacks)
        return model, cpu_params


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    started = time.time()
    for directory in [OOF_DIR, TEST_PREDS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    train_df = load_horizontal_wells(TRAIN_DIR, "train")
    test_df = load_horizontal_wells(TEST_DIR, "test")

    train_df, global_gr_median, global_tvt_input_median = add_baseline_features(train_df)
    test_df, _, _ = add_baseline_features(test_df, global_gr_median, global_tvt_input_median)

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
    active_params = dict(BASE_PARAMS)

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

    training_time = time.time() - started
    result = {
        "experiment_id": EXPERIMENT_ID,
        "model": "lightgbm",
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": cv_rmse_std,
        "features_used": FEATURES,
        "n_features": len(FEATURES),
        "training_time_seconds": float(training_time),
        "notes": "LightGBM baseline with basic MD/XYZ/GR features",
    }
    RESULT_PATH.write_text(json.dumps(jsonable(result), indent=2) + "\n")

    if oof.shape != (5_092_255,):
        raise ValueError(f"Unexpected OOF shape: {oof.shape}")
    if test_preds.shape != (19_221,):
        raise ValueError(f"Unexpected test prediction shape: {test_preds.shape}")

    print(f"Saved OOF: {OOF_PATH} shape={oof.shape} dtype={oof.dtype}")
    print(f"Saved test predictions: {TEST_PREDS_PATH} shape={test_preds.shape} dtype={test_preds.dtype}")
    print(f"Saved result JSON: {RESULT_PATH}")
    print(f"EXPERIMENT COMPLETE: cv_rmse={cv_rmse:.4f}")


if __name__ == "__main__":
    main()
