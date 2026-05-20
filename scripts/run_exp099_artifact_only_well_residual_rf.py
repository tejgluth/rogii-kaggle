"""exp099: Kaggle-input-only artifact blend with well-level RF residual.

This is the same clean well-level correction idea as exp098, but the base blend
uses only predictions available in the public artifact dataset.  It is intended
to mirror the Kaggle-ready notebook: competition sample_submission plus the
`ravaghi/wellbore-geology-prediction-artifacts` files.
"""
from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data/artifacts/wellbore-geology-prediction-artifacts"
RESULT = ROOT / "experiments/results/exp099.json"
SUBMISSION = ROOT / "submissions/exp099_artifact_only_well_residual_clean_oof.csv"

HELDOUT_WELLS = np.array(["000d7d20", "00bbac68", "00e12e8b"])
WEIGHTS = np.linspace(-1.0, 1.0, 41)


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    err = pred.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt(np.mean(err * err)))


def load_artifact_frame() -> tuple[pd.DataFrame, list[str]]:
    columns = pd.read_csv(ART / "train.csv", nrows=0).columns.tolist()
    feature_cols = [c for c in columns if c not in ("well", "id", "target")]
    dtypes = {c: "float32" for c in feature_cols + ["target"]}
    dtypes.update({"well": "string", "id": "string"})
    return pd.read_csv(ART / "train.csv", dtype=dtypes), feature_cols


def load_artifact_base(last_known_tvt: np.ndarray) -> tuple[np.ndarray, list[str]]:
    members: dict[str, np.ndarray] = {}
    for path in sorted(ART.glob("*_oof_preds.pkl")):
        name = path.name.replace("_oof_preds.pkl", "")
        members[name] = (last_known_tvt + joblib.load(path).astype(np.float32)).astype(np.float32)
    return np.mean(list(members.values()), axis=0).astype(np.float32), sorted(members)


def build_well_meta(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    core_tokens = ("last_known_tvt", "pf_", "beam_", "sc", "hyb", "sig_", "tw_", "gr", "frm_rmse")
    core_cols = [c for c in feature_cols if any(token in c for token in core_tokens)]
    agg = {c: ["mean"] for c in feature_cols}
    for col in core_cols:
        agg[col].append("std")
    meta = df[["well"] + feature_cols].groupby("well", sort=True).agg(agg)
    meta.columns = ["__".join(col).strip("_") for col in meta.columns.to_flat_index()]
    row_count = df.groupby("well", sort=True).size().rename("row_count").astype("float32")
    return meta.join(row_count).fillna(0.0)


def model_factories() -> dict[str, Callable[[], object]]:
    return {
        "ridge_1000": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=1000.0, solver="lsqr", fit_intercept=True)
        ),
        "ridge_10000": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=10000.0, solver="lsqr", fit_intercept=True)
        ),
        "rf_200_d4": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "et_300_d4": lambda: ExtraTreesRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "hgb_l2_1": lambda: HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.03,
            l2_regularization=1.0,
            max_leaf_nodes=15,
            random_state=42,
        ),
    }


def main() -> None:
    t0 = time.time()
    df, feature_cols = load_artifact_frame()
    wells = df["well"].astype(str).to_numpy()
    ids = df["id"].astype(str).to_numpy()
    last = df["last_known_tvt"].to_numpy(np.float32)
    y = (last + df["target"].to_numpy(np.float32)).astype(np.float32)
    heldout = np.isin(wells, HELDOUT_WELLS)
    dev = ~heldout

    base, members = load_artifact_base(last)
    residual = (y - base).astype(np.float32)
    base_dev_rmse = rmse(base[dev], y[dev])
    base_holdout_rmse = rmse(base[heldout], y[heldout])

    meta = build_well_meta(df, feature_cols)
    well_ids = meta.index.astype(str).to_numpy()
    Xw = meta.to_numpy(np.float32)
    well_target = (
        pd.DataFrame({"well": wells, "residual": residual})
        .groupby("well", sort=True)["residual"]
        .mean()
        .loc[well_ids]
        .to_numpy(np.float32)
    )
    well_to_pos = {well: pos for pos, well in enumerate(well_ids)}
    row_well_pos = np.array([well_to_pos[well] for well in wells], dtype=np.int32)
    held_well_mask = np.isin(well_ids, HELDOUT_WELLS)
    dev_well_positions = np.flatnonzero(~held_well_mask)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(dev_well_positions))

    factories = model_factories()
    results = []
    for name, factory in factories.items():
        oof_well = np.zeros(dev_well_positions.shape[0], dtype=np.float32)
        model_start = time.time()
        for tr_rel, va_rel in folds:
            tr_pos = dev_well_positions[tr_rel]
            va_pos = dev_well_positions[va_rel]
            model = factory()
            model.fit(Xw[tr_pos], well_target[tr_pos])
            oof_well[va_rel] = model.predict(Xw[va_pos]).astype(np.float32)
            del model
            gc.collect()

        correction_by_well = np.zeros(len(well_ids), dtype=np.float32)
        correction_by_well[dev_well_positions] = oof_well
        row_correction = correction_by_well[row_well_pos]
        weight_scores = [
            {
                "weight": float(weight),
                "dev_cv_rmse": rmse(base[dev] + weight * row_correction[dev], y[dev]),
            }
            for weight in WEIGHTS
        ]
        best_weight = min(weight_scores, key=lambda row: row["dev_cv_rmse"])
        row = {
            "name": name,
            "raw_dev_cv_rmse": rmse(base[dev] + row_correction[dev], y[dev]),
            "best_weight": best_weight["weight"],
            "best_dev_cv_rmse": best_weight["dev_cv_rmse"],
            "seconds": time.time() - model_start,
        }
        print(f"{name}: {row}", flush=True)
        results.append(row)

    selected = min(results, key=lambda row: row["best_dev_cv_rmse"])
    final_model = factories[selected["name"]]()
    final_model.fit(Xw[dev_well_positions], well_target[dev_well_positions])
    held_well_positions = np.flatnonzero(held_well_mask)
    held_correction = final_model.predict(Xw[held_well_positions]).astype(np.float32)
    del final_model
    gc.collect()

    correction_by_well = np.zeros(len(well_ids), dtype=np.float32)
    correction_by_well[held_well_positions] = held_correction
    row_correction = correction_by_well[row_well_pos]
    clean_pred = base[heldout] + selected["best_weight"] * row_correction[heldout]

    sample_ids = pd.read_csv(
        ROOT / "data/raw/rogii-wellbore-geology-prediction/sample_submission.csv", usecols=["id"]
    )
    clean_rows = pd.DataFrame({"id": ids[heldout], "tvt": clean_pred.astype(np.float32)})
    sub = sample_ids.merge(clean_rows, on="id", how="left")
    if int(sub["tvt"].isna().sum()):
        raise ValueError("missing clean artifact-only well residual predictions")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION, index=False)

    held_correction_by_well = {
        well: float(corr) for well, corr in zip(well_ids[held_well_positions], held_correction, strict=True)
    }
    payload = {
        "experiment_id": "exp099",
        "phase": "artifact_only_well_level_residual",
        "selection_rule": (
            "artifact-only base; model family and residual weight selected by 5-fold CV over non-heldout wells only; "
            "final well residual model trained on non-heldout wells and audited once on heldout wells"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "base_experiment": "artifact_only_equal_oof",
        "base_members": members,
        "feature_count": int(Xw.shape[1]),
        "weights": [float(w) for w in WEIGHTS],
        "base_dev_rmse": base_dev_rmse,
        "base_clean_holdout_rmse": base_holdout_rmse,
        "selected": selected,
        "clean_holdout_rmse": rmse(clean_pred, y[heldout]),
        "clean_holdout_mse": rmse(clean_pred, y[heldout]) ** 2,
        "clean_holdout_bias": float(np.mean(clean_pred - y[heldout])),
        "heldout_correction_by_well": held_correction_by_well,
        "candidate_results": sorted(results, key=lambda row: row["best_dev_cv_rmse"]),
        "submission_path": str(SUBMISSION.relative_to(ROOT)),
        "elapsed_seconds": time.time() - t0,
        "note": "No local exp028 files are used. Heldout labels are not used for fitting or selection.",
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
