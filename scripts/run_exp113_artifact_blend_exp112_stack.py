"""exp113: non-heldout convex artifact blend feeding the exp112 stack.

This tests whether replacing exp109's equal artifact average with a constrained
blend improves the clean model.  Blend methods are selected by GroupKFold over
non-heldout wells only.  The sample-submission wells are audited only after the
blend method, residual weights, and row alpha are selected.
"""
from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold, KFold

from clean_artifact_experiment_utils import (
    HELDOUT_WELLS,
    ROOT,
    audit_metrics,
    load_artifact_frame,
    load_artifact_members,
    make_submission,
    rmse,
    submission_report,
    well_equal_rmse,
)
from run_exp100_artifact_only_well_residual_gbr import build_well_meta
from run_exp104_artifact_row_ridge_plus_well import build_row_features, gbr_d4, make_well_oof, ridge
from run_exp111_exp109_oof_postprocess import apply_by_well


RESULT = ROOT / "experiments/results/exp113.json"
SUBMISSION = ROOT / "submissions/exp113_artifact_blend_exp112_stack.csv"

BLEND_METHODS = [
    {"name": "equal", "objective": "fixed", "l2_to_equal": 0.0},
    {"name": "row_mse_l2_0", "objective": "row_mse", "l2_to_equal": 0.0},
    {"name": "row_mse_l2_1", "objective": "row_mse", "l2_to_equal": 1.0},
    {"name": "row_mse_l2_10", "objective": "row_mse", "l2_to_equal": 10.0},
    {"name": "well_equal_l2_0", "objective": "well_equal", "l2_to_equal": 0.0},
    {"name": "well_equal_l2_1", "objective": "well_equal", "l2_to_equal": 1.0},
    {"name": "well_equal_l2_10", "objective": "well_equal", "l2_to_equal": 10.0},
]
ALPHAS = [0.01, 0.1, 1.0, 10.0]
WELL_WEIGHTS = np.arange(0.45, 0.60 + 1e-12, 0.0125)
ROW_WEIGHTS = np.arange(-0.20, -0.06 + 1e-12, 0.01)
SELECTED_POSTPROCESS = {
    "name": "savgol_w551_p2_s0.875",
    "kind": "savgol",
    "window": 551,
    "polyorder": 2,
    "shrink": 0.875,
}


def _normalise_weights(sample_weight: np.ndarray | None, n: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)
    weight = sample_weight.astype(np.float64, copy=True)
    return weight * (float(n) / float(weight.sum()))


def fit_convex_weights(
    P: np.ndarray,
    y: np.ndarray,
    *,
    l2_to_equal: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    n_models = P.shape[1]
    equal = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    weight = _normalise_weights(sample_weight, P.shape[0])
    P64 = P.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    l2 = float(l2_to_equal)

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        pred = P64 @ w
        err = pred - y64
        loss = float(np.mean(weight * err * err) + l2 * np.sum((w - equal) ** 2))
        grad = (2.0 / P64.shape[0]) * (P64.T @ (weight * err)) + 2.0 * l2 * (w - equal)
        return loss, grad

    result = minimize(
        fun=lambda w: objective(w)[0],
        jac=lambda w: objective(w)[1],
        x0=equal,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"blend optimizer failed: {result.message}")
    weights = np.clip(result.x.astype(np.float64), 0.0, 1.0)
    weights /= weights.sum()
    return weights


def well_equal_sample_weight(wells: np.ndarray) -> np.ndarray:
    counts = pd.Series(wells.astype(str)).map(pd.Series(wells.astype(str)).value_counts()).to_numpy(np.float64)
    return 1.0 / counts


def fit_method_weights(method: dict[str, object], P: np.ndarray, y: np.ndarray, wells: np.ndarray) -> np.ndarray:
    n_models = P.shape[1]
    if method["objective"] == "fixed":
        return np.full(n_models, 1.0 / n_models, dtype=np.float64)
    sample_weight = None
    if method["objective"] == "well_equal":
        sample_weight = well_equal_sample_weight(wells)
    return fit_convex_weights(
        P,
        y,
        l2_to_equal=float(method["l2_to_equal"]),
        sample_weight=sample_weight,
    )


def select_blend_method(
    P: np.ndarray,
    y: np.ndarray,
    wells: np.ndarray,
    dev: np.ndarray,
    member_names: list[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    idx_dev = np.flatnonzero(dev)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(idx_dev.shape[0]), groups=wells[dev]))
    results = []
    for method in BLEND_METHODS:
        oof = np.zeros(idx_dev.shape[0], dtype=np.float32)
        fold_rows = []
        start = time.time()
        for fold_idx, (tr_rel, va_rel) in enumerate(folds):
            tr_idx = idx_dev[tr_rel]
            va_idx = idx_dev[va_rel]
            weights = fit_method_weights(method, P[tr_idx], y[tr_idx], wells[tr_idx])
            pred = P[va_idx] @ weights
            oof[va_rel] = pred.astype(np.float32)
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "rmse": rmse(pred, y[va_idx]),
                    "well_equal_rmse": well_equal_rmse(pred, y[va_idx], wells[va_idx]),
                    "weights": {name: float(w) for name, w in zip(member_names, weights, strict=True)},
                    "rows": int(va_idx.size),
                    "wells": int(np.unique(wells[va_idx]).size),
                }
            )
        final_weights = fit_method_weights(method, P[idx_dev], y[idx_dev], wells[idx_dev])
        row = {
            "name": str(method["name"]),
            "objective": str(method["objective"]),
            "l2_to_equal": float(method["l2_to_equal"]),
            "dev_oof_rmse": rmse(oof, y[idx_dev]),
            "dev_well_equal_rmse": well_equal_rmse(oof, y[idx_dev], wells[idx_dev]),
            "dev_bias": float(np.mean(oof.astype(np.float64) - y[idx_dev].astype(np.float64))),
            "final_weights": {name: float(w) for name, w in zip(member_names, final_weights, strict=True)},
            "folds": fold_rows,
            "seconds": time.time() - start,
        }
        print(f"blend {row['name']}: rmse={row['dev_oof_rmse']:.5f} weights={row['final_weights']}", flush=True)
        results.append(row)
    selected = min(results, key=lambda row: (float(row["dev_oof_rmse"]), float(row["dev_well_equal_rmse"])))
    return selected, results


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
    member_names = sorted(artifact_members)
    P = np.column_stack([artifact_members[name] for name in member_names]).astype(np.float32)
    selected_blend, blend_results = select_blend_method(P, y, wells, dev, member_names)
    blend_weights = np.array([selected_blend["final_weights"][name] for name in member_names], dtype=np.float64)
    base = (P @ blend_weights).astype(np.float32)
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
    del artifact_members, P
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
            print(f"alpha={alpha:g} fold={fold_idx} raw_row={fold_scores[-1]['raw_row_rmse']:.4f}", flush=True)
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
    raw_clean_pred = (
        base[heldout]
        + float(best_selection["well_weight"]) * correction_by_well[row_well_pos][heldout]
        + float(best_selection["row_weight"]) * held_row_residual
    ).astype(np.float32)
    clean_pred = apply_by_well(raw_clean_pred, wells[heldout], SELECTED_POSTPROCESS)
    sub = make_submission(ids, clean_pred, heldout, SUBMISSION)

    payload = {
        "experiment_id": "exp113",
        "phase": "artifact_blend_exp112_stack",
        "selection_rule": (
            "convex artifact blend method is selected by 5-fold GroupKFold on non-heldout wells; "
            "well correction, row alpha, and residual weights are selected only on non-heldout OOF rows; "
            "exp112 smoother is fixed before heldout audit"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "member_names": member_names,
        "selected_blend": selected_blend,
        "blend_results": sorted(blend_results, key=lambda row: (float(row["dev_oof_rmse"]), float(row["dev_well_equal_rmse"]))),
        "base_dev_rmse": rmse(base[dev], y[dev]),
        "base_clean_holdout_rmse": rmse(base[heldout], y[heldout]),
        "row_feature_count": int(len(row_feature_names)),
        "alphas": ALPHAS,
        "well_weights": [float(w) for w in WELL_WEIGHTS],
        "row_weights": [float(w) for w in ROW_WEIGHTS],
        "selected_stack": best_selection,
        "alpha_results": alpha_results,
        "heldout_raw_metrics": audit_metrics(raw_clean_pred, y[heldout], wells[heldout]),
        "heldout_selected_metrics": audit_metrics(clean_pred, y[heldout], wells[heldout]),
        "selected_postprocess": SELECTED_POSTPROCESS,
        "submission_path": str(SUBMISSION.relative_to(ROOT)),
        "submission_report": submission_report(sub),
        "elapsed_seconds": time.time() - t0,
        "note": "Heldout labels are not used for fitting, blend selection, stack selection, or postprocess selection.",
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
