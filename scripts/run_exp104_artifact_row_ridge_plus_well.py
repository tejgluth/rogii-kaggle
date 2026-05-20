"""exp104: conservative row-level ridge residual plus exp100 well residual.

This tests whether a low-dimensional row model can add useful within-well
structure to exp100's artifact-only well residual correction.  The row model is
cross-fitted with GroupKFold over non-heldout wells.  The final well and row
weights are selected only from non-heldout OOF predictions.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_exp100_artifact_only_well_residual_gbr import (
    ART,
    HELDOUT_WELLS,
    ROOT,
    build_well_meta,
    load_artifact_frame,
    rmse,
)


RESULT = ROOT / "experiments/results/exp104.json"
SUBMISSION = ROOT / "submissions/exp104_artifact_row_ridge_plus_well_clean_oof.csv"

ALPHAS = [10_000.0, 100_000.0, 1_000_000.0]
WELL_WEIGHTS = np.linspace(0.0, 1.0, 21)
ROW_WEIGHTS = np.linspace(-0.30, 0.30, 13)
ROW_COLUMNS = [
    "last_known_tvt",
    "pf_ancc",
    "pf_ancc_std",
    "pf_ancc_delta",
    "pf_z",
    "pf_z_delta",
    "pf_vs_z",
    "beam_mean_d",
    "beam_std_d",
    "sc8_d",
    "sc15_d",
    "sc25_d",
    "sc_cons_d",
    "sc_ens_d",
    "sc_trust",
    "hyb_d",
    "sig_std",
    "sig_mean_d",
    "tw_range",
    "tw_gr_mean",
    "grm21",
    "grs21",
    "grm51",
    "grs51",
    "grm101",
    "grs101",
    "glag1",
    "glead1",
    "glag5",
    "glead5",
    "glag15",
    "glead15",
    "tdsc-15",
    "tdsc-8",
    "tdsc0",
    "tdsc8",
    "tdsc15",
    "tdpf-15",
    "tdpf-8",
    "tdpf0",
    "tdpf8",
    "tdpf15",
]


def load_artifact_members(last_known_tvt: np.ndarray) -> dict[str, np.ndarray]:
    members: dict[str, np.ndarray] = {}
    for path in sorted(ART.glob("*_oof_preds.pkl")):
        name = path.name.replace("_oof_preds.pkl", "")
        members[name] = (last_known_tvt + joblib.load(path).astype(np.float32)).astype(np.float32)
    return members


def gbr_d4() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )


def ridge(alpha: float):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", fit_intercept=True),
    )


def build_row_features(
    df: pd.DataFrame,
    wells: np.ndarray,
    base: np.ndarray,
    artifact_members: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    selected_cols = [col for col in ROW_COLUMNS if col in df.columns]
    feature_blocks = [df[selected_cols].to_numpy(np.float32, copy=True)]
    feature_names = list(selected_cols)

    member_names = sorted(artifact_members)
    member_stack = np.vstack([artifact_members[name] for name in member_names]).astype(np.float32)
    artifact_std = member_stack.std(axis=0).astype(np.float32)
    artifact_range = (member_stack.max(axis=0) - member_stack.min(axis=0)).astype(np.float32)
    cat_names = [name for name in member_names if name.startswith("catboost")]
    lgb_names = [name for name in member_names if name.startswith("lightgbm")]
    cat_mean = np.mean([artifact_members[name] for name in cat_names], axis=0).astype(np.float32)
    lgb_mean = np.mean([artifact_members[name] for name in lgb_names], axis=0).astype(np.float32)

    row_num = df.groupby("well", sort=False).cumcount().to_numpy(np.float32)
    row_count = df.groupby("well", sort=False)["id"].transform("size").to_numpy(np.float32)
    row_frac = row_num / np.maximum(row_count - 1.0, 1.0)
    last = df["last_known_tvt"].to_numpy(np.float32)
    grouped = pd.DataFrame({"well": wells, "base": base, "last": last})
    base_centered = (grouped["base"] - grouped.groupby("well", sort=False)["base"].transform("mean")).to_numpy(
        np.float32
    )
    last_centered = (grouped["last"] - grouped.groupby("well", sort=False)["last"].transform("mean")).to_numpy(
        np.float32
    )

    derived = np.column_stack(
        [
            base.astype(np.float32),
            artifact_std,
            artifact_range,
            cat_mean,
            lgb_mean,
            (cat_mean - lgb_mean).astype(np.float32),
            row_frac.astype(np.float32),
            np.log1p(row_count).astype(np.float32),
            base_centered,
            last_centered,
        ]
    ).astype(np.float32)
    feature_blocks.append(derived)
    feature_names.extend(
        [
            "artifact_base",
            "artifact_std",
            "artifact_range",
            "catboost_mean",
            "lightgbm_mean",
            "catboost_minus_lightgbm",
            "row_frac",
            "log_row_count",
            "base_centered_by_well",
            "last_centered_by_well",
        ]
    )
    X = np.column_stack(feature_blocks).astype(np.float32)
    del member_stack, derived, feature_blocks
    gc.collect()
    return X, feature_names


def make_well_oof(
    *,
    Xw: np.ndarray,
    well_target: np.ndarray,
    dev_well_positions: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    oof_well = np.zeros(dev_well_positions.shape[0], dtype=np.float32)
    for tr_rel, va_rel in folds:
        tr_pos = dev_well_positions[tr_rel]
        va_pos = dev_well_positions[va_rel]
        model = gbr_d4()
        model.fit(Xw[tr_pos], well_target[tr_pos])
        oof_well[va_rel] = model.predict(Xw[va_pos]).astype(np.float32)
        del model
        gc.collect()
    return oof_well


def best_combo_score(
    residual_dev: np.ndarray,
    well_dev: np.ndarray,
    row_dev: np.ndarray,
) -> dict[str, float]:
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
                best = {
                    "well_weight": ww,
                    "row_weight": rw,
                    "dev_cv_rmse": score,
                }
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
    well_only_dev_rmse = rmse(base[dev] + 0.5 * well_row_oof[dev], y[dev])

    X, row_feature_names = build_row_features(df, wells, base, artifact_members)
    del artifact_members
    gc.collect()

    group_folds = list(GroupKFold(n_splits=3).split(np.zeros(idx_dev.shape[0]), groups=wells[dev]))
    residual_dev = residual[idx_dev].astype(np.float32)
    alpha_results = []
    best_selection: dict[str, float] | None = None
    best_row_oof: np.ndarray | None = None
    for alpha in ALPHAS:
        row_oof = np.zeros(idx_dev.shape[0], dtype=np.float32)
        fold_scores = []
        start = time.time()
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
            best_row_oof = row_oof.copy()

    if best_selection is None or best_row_oof is None:
        raise RuntimeError("no row ridge selection was produced")

    final_well = gbr_d4()
    final_well.fit(Xw[dev_well_positions], well_target[dev_well_positions])
    held_well_correction = final_well.predict(Xw[held_well_positions]).astype(np.float32)
    del final_well
    gc.collect()

    if abs(best_selection["row_weight"]) > 1e-12:
        final_row = ridge(best_selection["alpha"])
        final_row.fit(X[dev], residual[dev])
        held_row_residual = final_row.predict(X[heldout]).astype(np.float32)
        del final_row
    else:
        held_row_residual = np.zeros(int(heldout.sum()), dtype=np.float32)
    del X
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
        raise ValueError("missing clean row ridge plus well predictions")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION, index=False)

    clean_holdout_rmse = rmse(clean_pred, y[heldout])
    payload = {
        "experiment_id": "exp104",
        "phase": "artifact_row_ridge_plus_well",
        "selection_rule": (
            "artifact-only equal OOF base; exp100 gbr_d4 well correction OOF and row-level ridge residual OOF are "
            "combined by weights selected on non-heldout rows only; row OOF uses GroupKFold by non-heldout wells; "
            "final models are trained on non-heldout rows/wells and audited once on heldout wells"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "base_members": sorted([path.name.replace("_oof_preds.pkl", "") for path in ART.glob("*_oof_preds.pkl")]),
        "row_feature_count": int(len(row_feature_names)),
        "row_features": row_feature_names,
        "alphas": ALPHAS,
        "well_weights": [float(w) for w in WELL_WEIGHTS],
        "row_weights": [float(w) for w in ROW_WEIGHTS],
        "base_dev_rmse": rmse(base[dev], y[dev]),
        "well_only_dev_rmse": well_only_dev_rmse,
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
