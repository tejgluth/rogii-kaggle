#!/usr/bin/env python3
"""XGBoost baseline for ROGII experiment exp_b002."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold


EXPERIMENT_ID = "exp_b002"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
OOF_DIR = REPO_ROOT / "experiments" / "oof"
TEST_PREDS_DIR = REPO_ROOT / "experiments" / "test_preds"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

OOF_PATH = OOF_DIR / "oof_xgb_exp_b002.npy"
TEST_PREDS_PATH = TEST_PREDS_DIR / "test_xgb_exp_b002.npy"
RESULTS_PATH = RESULTS_DIR / "exp_b002.json"

BASE_FEATURES = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
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

EXPECTED_TRAIN_ROWS = 5_092_255
EXPECTED_TEST_ROWS = 19_221


def get_well_id(path: Path) -> str:
    return path.name.split("__", maxsplit=1)[0]


def load_horizontal_wells(split_dir: Path, require_target: bool) -> pd.DataFrame:
    paths = sorted(split_dir.glob("*__horizontal_well.csv"))
    if not paths:
        raise FileNotFoundError(f"No horizontal well files found in {split_dir}")

    usecols = BASE_FEATURES + (["TVT"] if require_target else [])
    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=usecols)
        df["well_id"] = get_well_id(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    print(
        f"Loaded {split_dir.name}: {len(out):,} rows, "
        f"{out['well_id'].nunique():,} wells"
    )
    return out


def load_typewells(split_dir: Path) -> pd.DataFrame:
    paths = sorted(split_dir.glob("*__typewell.csv"))
    if not paths:
        raise FileNotFoundError(f"No typewell files found in {split_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=["TVT", "GR", "Geology"])
        df["well_id"] = get_well_id(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    print(f"Loaded typewells: {len(out):,} rows, {out['well_id'].nunique():,} wells")
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["well_id", "MD"], kind="mergesort").reset_index(drop=True)
    grouped = df.groupby("well_id", sort=False)

    depth_from_heel = df["MD"] - grouped["MD"].transform("min")
    depth_max = depth_from_heel.groupby(df["well_id"], sort=False).transform("max")
    df["depth_from_heel"] = depth_from_heel
    df["depth_fraction"] = np.divide(
        depth_from_heel,
        depth_max,
        out=np.zeros(len(df), dtype=np.float64),
        where=depth_max.to_numpy() != 0,
    )

    global_gr_median = df["GR"].median()
    per_well_gr_median = grouped["GR"].transform("median")
    df["gr_filled"] = df["GR"].fillna(per_well_gr_median).fillna(global_gr_median)

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

    global_tvt_input_median = df["TVT_input"].median()
    per_well_tvt_input_median = grouped["TVT_input"].transform("median")
    df["tvt_input_filled"] = (
        df["TVT_input"].fillna(per_well_tvt_input_median).fillna(global_tvt_input_median)
    )

    df["x_diff"] = grouped["X"].diff().fillna(0)
    df["y_diff"] = grouped["Y"].diff().fillna(0)
    df["z_diff"] = grouped["Z"].diff().fillna(0)
    df["lateral_speed"] = np.sqrt(
        df["x_diff"] ** 2 + df["y_diff"] ** 2 + df["z_diff"] ** 2
    )

    for col in FEATURES:
        df[col] = df[col].astype("float32")
    return df


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def make_model(params: dict[str, object], early_stopping_rounds: int | None) -> xgb.XGBRegressor:
    model_params = params.copy()
    if early_stopping_rounds is not None:
        model_params["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBRegressor(**model_params)


def main() -> None:
    start = time.time()
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "device": "cuda",
        "verbosity": 0,
        "tree_method": "hist",
    }

    train_df = load_horizontal_wells(TRAIN_DIR, require_target=True)
    test_df = load_horizontal_wells(TEST_DIR, require_target=False)
    _ = load_typewells(TRAIN_DIR)

    if len(train_df) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Unexpected train rows: {len(train_df):,}")
    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Unexpected test rows: {len(test_df):,}")

    combined = pd.concat(
        [
            train_df.assign(_is_train=True),
            test_df.assign(TVT=np.nan, _is_train=False),
        ],
        ignore_index=True,
    )
    combined = add_features(combined)

    train_mask = combined["_is_train"].to_numpy(dtype=bool)
    X_train = combined.loc[train_mask, FEATURES].reset_index(drop=True)
    X_test = combined.loc[~train_mask, FEATURES].reset_index(drop=True)
    well_ids = combined.loc[train_mask, "well_id"].reset_index(drop=True)
    y_train = combined.loc[train_mask, "TVT"].astype("float32").reset_index(drop=True)

    del train_df, test_df, combined
    gc.collect()

    oof = np.zeros(len(X_train), dtype=np.float32)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    gkf = GroupKFold(n_splits=5)

    active_params = params.copy()
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=well_ids)):
        print(f"Training fold {fold + 1}/5 with device={active_params['device']}")
        model = make_model(active_params, early_stopping_rounds=100)
        try:
            model.fit(
                X_train.iloc[tr_idx],
                y_train.iloc[tr_idx],
                eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                verbose=200,
            )
        except xgb.core.XGBoostError as err:
            if active_params.get("device") != "cuda":
                raise
            print(f"CUDA training failed, falling back to CPU: {err}")
            active_params["device"] = "cpu"
            model = make_model(active_params, early_stopping_rounds=100)
            model.fit(
                X_train.iloc[tr_idx],
                y_train.iloc[tr_idx],
                eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                verbose=200,
            )

        preds = model.predict(X_train.iloc[val_idx]).astype("float32")
        oof[val_idx] = preds
        fold_rmse = rmse(preds, y_train.iloc[val_idx].to_numpy(dtype=np.float32))
        best_iteration = int(getattr(model, "best_iteration", active_params["n_estimators"] - 1))
        fold_scores.append(fold_rmse)
        best_iterations.append(best_iteration + 1)
        print(
            f"Fold {fold + 1}: RMSE={fold_rmse:.4f}, "
            f"best_n_estimators={best_iteration + 1}"
        )

        del model, preds
        gc.collect()

    cv_rmse = rmse(oof, y_train.to_numpy(dtype=np.float32))
    cv_rmse_std = float(np.std(fold_scores, ddof=1))
    final_n_estimators = max(1, int(round(float(np.mean(best_iterations)))))
    print(f"CV RMSE: {cv_rmse:.4f}")
    print(f"Retraining final model with n_estimators={final_n_estimators}")

    final_params = active_params.copy()
    final_params["n_estimators"] = final_n_estimators
    final_model = make_model(final_params, early_stopping_rounds=None)
    final_model.fit(X_train, y_train, verbose=200)
    test_preds = final_model.predict(X_test).astype("float32")

    if oof.shape != (EXPECTED_TRAIN_ROWS,):
        raise ValueError(f"Unexpected OOF shape: {oof.shape}")
    if test_preds.shape != (EXPECTED_TEST_ROWS,):
        raise ValueError(f"Unexpected test prediction shape: {test_preds.shape}")

    np.save(OOF_PATH, oof.astype("float32"))
    np.save(TEST_PREDS_PATH, test_preds.astype("float32"))

    elapsed = time.time() - start
    results = {
        "experiment_id": EXPERIMENT_ID,
        "model": "xgboost",
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": cv_rmse_std,
        "features_used": FEATURES,
        "n_features": len(FEATURES),
        "training_time_seconds": float(elapsed),
        "notes": "XGBoost baseline with basic MD/XYZ/GR features",
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"Saved OOF: {OOF_PATH}")
    print(f"Saved test predictions: {TEST_PREDS_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"EXPERIMENT COMPLETE: cv_rmse={cv_rmse:.4f}")


if __name__ == "__main__":
    main()
