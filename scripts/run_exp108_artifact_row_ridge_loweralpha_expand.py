"""exp108: controlled expansion after exp107 selected alpha/row-weight edges."""
from __future__ import annotations

import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from run_exp100_artifact_only_well_residual_gbr import (
    ART,
    HELDOUT_WELLS,
    ROOT,
    build_well_meta,
    load_artifact_frame,
    rmse,
)
from run_exp104_artifact_row_ridge_plus_well import build_row_features, gbr_d4, make_well_oof, ridge


RESULT = ROOT / "experiments/results/exp108.json"
SUBMISSION = ROOT / "submissions/exp108_artifact_row_ridge_loweralpha_expand_clean_oof.csv"

ALPHAS = [10.0, 20.0, 50.0, 100.0, 150.0, 200.0]
WELL_WEIGHTS = np.arange(0.50, 0.55 + 1e-12, 0.00625)
ROW_WEIGHTS = np.arange(-0.18, -0.10 + 1e-12, 0.005)


def load_artifact_members(last_known_tvt: np.ndarray) -> dict[str, np.ndarray]:
    members: dict[str, np.ndarray] = {}
    for path in sorted(ART.glob("*_oof_preds.pkl")):
        name = path.name.replace("_oof_preds.pkl", "")
        members[name] = (last_known_tvt + joblib.load(path).astype(np.float32)).astype(np.float32)
    return members


def best_combo_score(residual_dev: np.ndarray, well_dev: np.ndarray, row_dev: np.ndarray) -> dict[str, float]:
    r = residual_dev.astype(np.float64)
    a = well_dev.astype(np.float64)
    b = row_dev.astype(np.float64)
    n = float(r.size)
    rr = float(np.dot(r, r))
    ra = float(np.dot(r, a))
    rb = float(np.dot(r, b))
    aa = float(np.dot(a, a))
    ab = float(np.dot(a, b))
    bb = float(np.dot(b, b))
    best: dict[str, float] | None = None
    for well_weight in WELL_WEIGHTS:
        ww = float(well_weight)
        for row_weight in ROW_WEIGHTS:
            rw = float(row_weight)
            mse = (rr - 2 * ww * ra - 2 * rw * rb + ww * ww * aa + 2 * ww * rw * ab + rw * rw * bb) / n
            score = float(np.sqrt(max(mse, 0.0)))
            if best is None or score < best["dev_cv_rmse"]:
                best = {"well_weight": ww, "row_weight": rw, "dev_cv_rmse": score}
    if best is None:
        raise RuntimeError("no combo score produced")
    return best


def main() -> None:
    t0 = time.time()
    df, feature_cols = load_artifact_frame()
    wells = df["well"].astype(str).to_numpy()
    ids = df["id"].astype(str).to_numpy()
    last = df["last_known_tvt"].to_numpy(np.float32)
    y = (last + df["target"].to_numpy(np.float32)).astype(np.float32)
    heldout = np.isin(wells, HELDOUT_WELLS)
    dev = ~heldout
    idx_dev = np.flatnonzero(dev)

    artifact_members = load_artifact_members(last)
    base = np.mean(list(artifact_members.values()), axis=0).astype(np.float32)
    residual = (y - base).astype(np.float32)

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
    held_well_positions = np.flatnonzero(held_well_mask)
    well_folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(dev_well_positions))
    well_oof = make_well_oof(
        Xw=Xw,
        well_target=well_target,
        dev_well_positions=dev_well_positions,
        folds=well_folds,
    )
    correction_by_well = np.zeros(len(well_ids), dtype=np.float32)
    correction_by_well[dev_well_positions] = well_oof
    well_row_oof = correction_by_well[row_well_pos]

    X, row_feature_names = build_row_features(df, wells, base, artifact_members)
    del artifact_members
    gc.collect()

    group_folds = list(GroupKFold(n_splits=3).split(np.zeros(idx_dev.shape[0]), groups=wells[dev]))
    residual_dev = residual[idx_dev].astype(np.float32)
    alpha_results = []
    best_selection: dict[str, float] | None = None
    for alpha in ALPHAS:
        row_oof = np.zeros(idx_dev.shape[0], dtype=np.float32)
        start = time.time()
        fold_scores = []
        for fold_idx, (tr_rel, va_rel) in enumerate(group_folds):
            tr_idx = idx_dev[tr_rel]
            va_idx = idx_dev[va_rel]
            model = ridge(alpha)
            model.fit(X[tr_idx], residual[tr_idx])
            pred = model.predict(X[va_idx]).astype(np.float32)
            row_oof[va_rel] = pred
            fold_scores.append(
                {
                    "fold": fold_idx,
                    "base_rmse": rmse(base[va_idx], y[va_idx]),
                    "raw_row_rmse": rmse(base[va_idx] + pred, y[va_idx]),
                    "rows": int(va_idx.size),
                    "wells": int(np.unique(wells[va_idx]).size),
                }
            )
            print(
                f"alpha={alpha:g} fold={fold_idx} "
                f"base={fold_scores[-1]['base_rmse']:.4f} "
                f"raw_row={fold_scores[-1]['raw_row_rmse']:.4f}",
                flush=True,
            )
            del model
            gc.collect()
        combo = best_combo_score(residual_dev, well_row_oof[idx_dev], row_oof)
        row = {
            "alpha": float(alpha),
            "raw_row_dev_cv_rmse": rmse(base[idx_dev] + row_oof, y[idx_dev]),
            "best_well_weight": combo["well_weight"],
            "best_row_weight": combo["row_weight"],
            "best_dev_cv_rmse": combo["dev_cv_rmse"],
            "fold_scores": fold_scores,
            "seconds": time.time() - start,
        }
        print(f"alpha={alpha:g} combo={combo}", flush=True)
        alpha_results.append(row)
        if best_selection is None or row["best_dev_cv_rmse"] < best_selection["best_dev_cv_rmse"]:
            best_selection = {
                "alpha": row["alpha"],
                "well_weight": row["best_well_weight"],
                "row_weight": row["best_row_weight"],
                "best_dev_cv_rmse": row["best_dev_cv_rmse"],
            }

    if best_selection is None:
        raise RuntimeError("no row ridge selection was produced")

    final_well = gbr_d4()
    final_well.fit(Xw[dev_well_positions], well_target[dev_well_positions])
    held_well_correction = final_well.predict(Xw[held_well_positions]).astype(np.float32)
    del final_well
    gc.collect()

    final_row = ridge(best_selection["alpha"])
    final_row.fit(X[dev], residual[dev])
    held_row_residual = final_row.predict(X[heldout]).astype(np.float32)
    del final_row, X
    gc.collect()

    correction_by_well = np.zeros(len(well_ids), dtype=np.float32)
    correction_by_well[held_well_positions] = held_well_correction
    held_well_row = correction_by_well[row_well_pos][heldout]
    clean_pred = (
        base[heldout]
        + float(best_selection["well_weight"]) * held_well_row
        + float(best_selection["row_weight"]) * held_row_residual
    )

    sample_ids = pd.read_csv(
        ROOT / "data/raw/rogii-wellbore-geology-prediction/sample_submission.csv", usecols=["id"]
    )
    clean_rows = pd.DataFrame({"id": ids[heldout], "tvt": clean_pred.astype(np.float32)})
    sub = sample_ids.merge(clean_rows, on="id", how="left")
    if int(sub["tvt"].isna().sum()):
        raise ValueError("missing clean lower-alpha expanded row ridge predictions")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION, index=False)

    clean_holdout_rmse = rmse(clean_pred, y[heldout])
    payload = {
        "experiment_id": "exp108",
        "phase": "artifact_row_ridge_loweralpha_expand",
        "selection_rule": (
            "exp107 structure with lower ridge alphas and stronger negative row weights after exp107 selected grid edges; "
            "all alpha/weight selection uses non-heldout OOF rows only; final models train on non-heldout data and are "
            "audited once on heldout wells"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "base_members": sorted([path.name.replace("_oof_preds.pkl", "") for path in ART.glob("*_oof_preds.pkl")]),
        "row_feature_count": int(len(row_feature_names)),
        "alphas": ALPHAS,
        "well_weights": [float(w) for w in WELL_WEIGHTS],
        "row_weights": [float(w) for w in ROW_WEIGHTS],
        "base_dev_rmse": rmse(base[dev], y[dev]),
        "base_clean_holdout_rmse": rmse(base[heldout], y[heldout]),
        "selected": best_selection,
        "clean_holdout_rmse": clean_holdout_rmse,
        "clean_holdout_mse": clean_holdout_rmse**2,
        "clean_holdout_bias": float(np.mean(clean_pred - y[heldout])),
        "heldout_well_correction_by_well": {
            well: float(corr) for well, corr in zip(well_ids[held_well_positions], held_well_correction, strict=True)
        },
        "alpha_results": alpha_results,
        "submission_path": str(SUBMISSION.relative_to(ROOT)),
        "elapsed_seconds": time.time() - t0,
        "note": "No local exp028 files are used. Heldout labels are not used for fitting or selection.",
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
