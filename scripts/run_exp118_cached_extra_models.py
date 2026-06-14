#!/usr/bin/env python3
"""exp118: train extra base models on exp116 cached features.

The submitted exp116 cache used one LightGBM and one CatBoost model. This script
adds diverse CatBoost members using the same official-data feature matrix. It
uses GroupKFold by well, saves OOF predictions, and writes native CatBoost model
files so a Kaggle inference notebook can load them without pickled-version risk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SRC_RUN = ROOT / "local_runs" / "exp116_fast_cache"
OUT = ROOT / "local_runs" / "exp118_cached_extra_models"
OUT.mkdir(parents=True, exist_ok=True)


def read_cached_frame() -> tuple[pd.DataFrame, list[str]]:
    columns = pd.read_csv(SRC_RUN / "train.csv", nrows=0).columns.tolist()
    feature_cols = [c for c in columns if c not in {"well", "id", "target"}]
    dtypes = {c: "float32" for c in feature_cols + ["target"]}
    dtypes["well"] = "string"
    dtypes["id"] = "string"
    print(f"reading train.csv with {len(feature_cols)} features", flush=True)
    df = pd.read_csv(SRC_RUN / "train.csv", dtype=dtypes)
    return df, feature_cols


def train_catboost(df: pd.DataFrame, feature_cols: list[str], name: str, params: dict, n_splits: int) -> dict:
    X = df[feature_cols]
    y = df["target"].to_numpy(np.float32)
    groups = df["well"].astype(str).to_numpy()
    oof = np.zeros(len(df), dtype=np.float32)
    fold_scores = []
    model_dir = OUT / name
    model_dir.mkdir(parents=True, exist_ok=True)
    models = []

    for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=groups)):
        t0 = time.time()
        model = CatBoostRegressor(**params)
        model.fit(
            Pool(X.iloc[tr_idx].values, label=y[tr_idx]),
            eval_set=Pool(X.iloc[va_idx].values, label=y[va_idx]),
            use_best_model=True,
        )
        pred = model.predict(X.iloc[va_idx].values).astype(np.float32)
        oof[va_idx] = pred
        score = float(root_mean_squared_error(y[va_idx], pred))
        fold_scores.append(score)
        model.save_model(str(model_dir / f"{name}_fold{fold}.cbm"))
        models.append(model)
        print(f"{name} fold={fold} best_iter={model.get_best_iteration()} rmse={score:.6f} elapsed={time.time()-t0:.1f}s", flush=True)

    oof_path = OUT / f"{name}_oof_preds.pkl"
    joblib.dump(oof, oof_path)
    joblib.dump(models, OUT / f"{name}_models.pkl")
    overall = float(root_mean_squared_error(y, oof))
    return {
        "name": name,
        "params": params,
        "n_splits": n_splits,
        "fold_scores": fold_scores,
        "overall_oof_rmse": overall,
        "oof_path": str(oof_path),
    }


def main() -> None:
    n_splits = int(os.environ.get("EXP118_N_SPLITS", "3"))
    max_iter = int(os.environ.get("EXP118_MAX_ITER", "3500"))
    df, feature_cols = read_cached_frame()

    base = dict(
        iterations=max_iter,
        depth=7,
        l2_leaf_reg=2.0,
        min_data_in_leaf=15,
        border_count=254,
        loss_function="RMSE",
        od_type="Iter",
        od_wait=min(150, max(50, max_iter // 20)),
        verbose=0,
        task_type="CPU",
        thread_count=-1,
        allow_writing_files=False,
    )
    configs = [
        ("catboost-2", dict(learning_rate=0.020, random_seed=7, **base)),
        ("catboost-3", dict(learning_rate=0.030, random_seed=123, depth=6, l2_leaf_reg=4.0, **{k: v for k, v in base.items() if k not in {"depth", "l2_leaf_reg"}})),
    ]
    if os.environ.get("EXP118_ONE_MODEL", "0") == "1":
        configs = configs[:1]

    reports = []
    for name, params in configs:
        print(f"training {name}", flush=True)
        reports.append(train_catboost(df, feature_cols, name, params, n_splits))

    meta = {
        "experiment_id": "exp118",
        "route": "extra_cached_catboost_members",
        "source_run": str(SRC_RUN),
        "feature_count": len(feature_cols),
        "row_count": int(len(df)),
        "reports": reports,
    }
    (OUT / "exp118_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
