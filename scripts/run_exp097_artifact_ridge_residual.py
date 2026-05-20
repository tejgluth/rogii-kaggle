"""exp097: dev-CV selected ridge residual on artifact features.

This keeps the exp095 clean base blend and learns only a small residual
correction from the artifact feature table.  Hyperparameters and residual
shrinkage are selected using GroupKFold on non-test wells only.  The three
sample-submission wells are used once at the end for the clean audit.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data/artifacts/wellbore-geology-prediction-artifacts"
RESULT = ROOT / "experiments/results/exp097.json"
SUBMISSION = ROOT / "submissions/exp097_artifact_ridge_residual_clean_oof.csv"

HELDOUT_WELLS = np.array(["000d7d20", "00bbac68", "00e12e8b"])
ALPHAS = [1.0, 100.0, 10000.0]
WEIGHTS = np.linspace(-0.5, 0.5, 21)


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    err = pred.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt(np.mean(err * err)))


def load_artifact_frame() -> tuple[pd.DataFrame, list[str]]:
    columns = pd.read_csv(ART / "train.csv", nrows=0).columns.tolist()
    feature_cols = [c for c in columns if c not in ("well", "id", "target")]
    dtypes = {c: "float32" for c in feature_cols + ["target"]}
    dtypes.update({"well": "string", "id": "string"})
    return pd.read_csv(ART / "train.csv", dtype=dtypes), feature_cols


def load_exp095_base(last_known_tvt: np.ndarray, wells: np.ndarray) -> tuple[np.ndarray, list[str]]:
    members: dict[str, np.ndarray] = {}
    for path in sorted(ART.glob("*_oof_preds.pkl")):
        name = path.name.replace("_oof_preds.pkl", "")
        members[name] = (last_known_tvt + joblib.load(path).astype(np.float32)).astype(np.float32)

    cache = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
    local_groups = np.asarray(cache["groups"]).astype(str)
    local_lateral = cache["is_lateral"].astype(bool)
    if not np.array_equal(local_groups[local_lateral], wells):
        raise ValueError("local exp028 lateral row order does not match artifact train.csv")
    members["local_exp028"] = np.load(ROOT / "experiments/oof/oof_combined_exp028.npy").astype(np.float32)[
        local_lateral
    ]

    return np.mean(list(members.values()), axis=0).astype(np.float32), sorted(members)


def fit_ridge(alpha: float):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", fit_intercept=True, random_state=42),
    )


def main() -> None:
    t0 = time.time()
    df, feature_cols = load_artifact_frame()
    wells = df["well"].astype(str).to_numpy()
    ids = df["id"].astype(str).to_numpy()
    last = df["last_known_tvt"].to_numpy(np.float32)
    y = (last + df["target"].to_numpy(np.float32)).astype(np.float32)
    X = df[feature_cols].to_numpy(np.float32, copy=False)
    heldout = np.isin(wells, HELDOUT_WELLS)
    dev = ~heldout
    idx_dev = np.flatnonzero(dev)

    base, members = load_exp095_base(last, wells)
    residual = (y - base).astype(np.float32)
    folds = list(GroupKFold(n_splits=3).split(np.zeros(idx_dev.shape[0]), groups=wells[dev]))

    base_dev_rmse = rmse(base[dev], y[dev])
    base_holdout_rmse = rmse(base[heldout], y[heldout])
    alpha_results = []
    best_selection: dict[str, float] | None = None

    for alpha in ALPHAS:
        oof_residual = np.zeros(idx_dev.shape[0], dtype=np.float32)
        fold_scores = []
        alpha_start = time.time()
        for fold_idx, (tr_rel, va_rel) in enumerate(folds):
            tr_idx = idx_dev[tr_rel]
            va_idx = idx_dev[va_rel]
            model = fit_ridge(alpha)
            model.fit(X[tr_idx], residual[tr_idx])
            pred = model.predict(X[va_idx]).astype(np.float32)
            oof_residual[va_rel] = pred
            fold_scores.append(
                {
                    "fold": fold_idx,
                    "base_rmse": rmse(base[va_idx], y[va_idx]),
                    "raw_residual_rmse": rmse(base[va_idx] + pred, y[va_idx]),
                    "rows": int(va_idx.size),
                    "wells": int(np.unique(wells[va_idx]).size),
                }
            )
            print(
                f"alpha={alpha:g} fold={fold_idx} "
                f"base={fold_scores[-1]['base_rmse']:.4f} "
                f"raw={fold_scores[-1]['raw_residual_rmse']:.4f}",
                flush=True,
            )
            del model
            gc.collect()

        weight_scores = [
            {
                "weight": float(weight),
                "dev_cv_rmse": rmse(base[idx_dev] + weight * oof_residual, y[idx_dev]),
            }
            for weight in WEIGHTS
        ]
        best_weight = min(weight_scores, key=lambda row: row["dev_cv_rmse"])
        row = {
            "alpha": float(alpha),
            "base_dev_rmse": base_dev_rmse,
            "raw_residual_dev_cv_rmse": rmse(base[idx_dev] + oof_residual, y[idx_dev]),
            "best_weight": best_weight["weight"],
            "best_dev_cv_rmse": best_weight["dev_cv_rmse"],
            "fold_scores": fold_scores,
            "seconds": time.time() - alpha_start,
        }
        print(f"alpha={alpha:g} best={best_weight}", flush=True)
        alpha_results.append(row)
        if best_selection is None or row["best_dev_cv_rmse"] < best_selection["best_dev_cv_rmse"]:
            best_selection = {
                "alpha": row["alpha"],
                "weight": row["best_weight"],
                "best_dev_cv_rmse": row["best_dev_cv_rmse"],
            }

    if best_selection is None:
        raise RuntimeError("no ridge selection was produced")

    final_model = fit_ridge(best_selection["alpha"])
    final_model.fit(X[dev], residual[dev])
    heldout_residual = final_model.predict(X[heldout]).astype(np.float32)
    clean_pred = base[heldout] + best_selection["weight"] * heldout_residual
    del final_model
    gc.collect()

    sample_ids = pd.read_csv(
        ROOT / "data/raw/rogii-wellbore-geology-prediction/sample_submission.csv", usecols=["id"]
    )
    clean_rows = pd.DataFrame({"id": ids[heldout], "tvt": clean_pred.astype(np.float32)})
    sub = sample_ids.merge(clean_rows, on="id", how="left")
    if int(sub["tvt"].isna().sum()):
        raise ValueError("missing clean ridge residual predictions")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION, index=False)

    payload = {
        "experiment_id": "exp097",
        "phase": "artifact_feature_ridge_residual",
        "selection_rule": (
            "alpha and residual weight selected by 3-fold GroupKFold on non-heldout wells only; "
            "final residual model trained on non-heldout wells and audited once on the three heldout wells"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "base_experiment": "exp095",
        "base_members": members,
        "feature_count": len(feature_cols),
        "alphas": ALPHAS,
        "weights": [float(w) for w in WEIGHTS],
        "selected": best_selection,
        "base_dev_rmse": base_dev_rmse,
        "selected_dev_cv_rmse": best_selection["best_dev_cv_rmse"],
        "base_clean_holdout_rmse": base_holdout_rmse,
        "clean_holdout_rmse": rmse(clean_pred, y[heldout]),
        "clean_holdout_mse": rmse(clean_pred, y[heldout]) ** 2,
        "clean_holdout_bias": float(np.mean(clean_pred - y[heldout])),
        "alpha_results": alpha_results,
        "submission_path": str(SUBMISSION.relative_to(ROOT)),
        "elapsed_seconds": time.time() - t0,
        "note": "The residual target is y_abs - exp095_base. No heldout labels are used for fitting or selection.",
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
