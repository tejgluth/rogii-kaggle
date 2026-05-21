"""exp114: select residual weights after smoothing OOF components.

exp112 selected the smoother on the final exp109 OOF prediction, but kept
exp109's residual weights fixed.  This experiment uses the linearity of the
Savitzky-Golay smoother to smooth base, well-residual, and row-residual OOF
components separately, then selects alpha, smoother, and residual weights only
on non-heldout OOF rows.
"""
from __future__ import annotations

import gc
import json
import time

import numpy as np
import pandas as pd
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


RESULT = ROOT / "experiments/results/exp114.json"
SUBMISSION = ROOT / "submissions/exp114_smoothed_component_weight_select.csv"

ALPHAS = [0.01, 0.1, 1.0, 10.0]
WELL_WEIGHTS = np.arange(0.40, 0.65 + 1e-12, 0.0125)
ROW_WEIGHTS = np.arange(-0.22, -0.04 + 1e-12, 0.01)


def smoother_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = [{"name": "none", "kind": "none"}]
    for window in (451, 501, 551, 601, 651):
        for polyorder in (2, 3):
            for shrink in (0.75, 0.875, 1.0):
                candidates.append(
                    {
                        "name": f"savgol_w{window}_p{polyorder}_s{shrink:g}",
                        "kind": "savgol",
                        "window": window,
                        "polyorder": polyorder,
                        "shrink": shrink,
                    }
                )
    return candidates


def best_component_combo(
    y: np.ndarray,
    base_component: np.ndarray,
    well_component: np.ndarray,
    row_component: np.ndarray,
) -> dict[str, float]:
    r = (y - base_component).astype(np.float64)
    a = well_component.astype(np.float64)
    b = row_component.astype(np.float64)
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
            if best is None or score < best["dev_oof_rmse"]:
                best = {
                    "well_weight": ww,
                    "row_weight": rw,
                    "dev_oof_rmse": score,
                }
    if best is None:
        raise RuntimeError("no component combo produced")
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
    candidates = smoother_candidates()
    y_dev = y[idx_dev]
    wells_dev = wells[idx_dev]
    base_dev = base[idx_dev]
    well_dev = well_row_oof[idx_dev]
    alpha_results = []
    best_selection: dict[str, object] | None = None

    for alpha in ALPHAS:
        row_oof = np.zeros(idx_dev.shape[0], dtype=np.float32)
        fold_scores = []
        alpha_start = time.time()
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
                    "raw_row_rmse": rmse(base[va_idx] + pred, y[va_idx]),
                    "rows": int(va_idx.size),
                    "wells": int(np.unique(wells[va_idx]).size),
                }
            )
            print(f"alpha={alpha:g} fold={fold_idx} raw_row={fold_scores[-1]['raw_row_rmse']:.4f}", flush=True)
            del model
            gc.collect()

        smoother_scores = []
        for candidate in candidates:
            start = time.time()
            smooth_base = apply_by_well(base_dev, wells_dev, candidate)
            smooth_well = apply_by_well(well_dev, wells_dev, candidate)
            smooth_row = apply_by_well(row_oof, wells_dev, candidate)
            combo = best_component_combo(y_dev, smooth_base, smooth_well, smooth_row)
            pred = (
                smooth_base
                + float(combo["well_weight"]) * smooth_well
                + float(combo["row_weight"]) * smooth_row
            ).astype(np.float32)
            score = {
                "alpha": float(alpha),
                "smoother": candidate,
                "smoother_name": candidate["name"],
                "well_weight": combo["well_weight"],
                "row_weight": combo["row_weight"],
                "dev_oof_rmse": combo["dev_oof_rmse"],
                "dev_well_equal_rmse": well_equal_rmse(pred, y_dev, wells_dev),
                "dev_bias": float(np.mean(pred.astype(np.float64) - y_dev.astype(np.float64))),
                "seconds": time.time() - start,
            }
            smoother_scores.append(score)
            print(
                f"alpha={alpha:g} smoother={candidate['name']} "
                f"rmse={score['dev_oof_rmse']:.5f} ww={score['well_weight']:.4f} rw={score['row_weight']:.4f}",
                flush=True,
            )
            del smooth_base, smooth_well, smooth_row, pred
            gc.collect()

        alpha_row = {
            "alpha": float(alpha),
            "fold_scores": fold_scores,
            "best_smoother": min(
                smoother_scores,
                key=lambda row: (float(row["dev_oof_rmse"]), float(row["dev_well_equal_rmse"])),
            ),
            "smoother_scores": sorted(
                smoother_scores,
                key=lambda row: (float(row["dev_oof_rmse"]), float(row["dev_well_equal_rmse"])),
            ),
            "seconds": time.time() - alpha_start,
        }
        alpha_results.append(alpha_row)
        candidate_best = alpha_row["best_smoother"]
        if best_selection is None or (
            float(candidate_best["dev_oof_rmse"]),
            float(candidate_best["dev_well_equal_rmse"]),
        ) < (
            float(best_selection["dev_oof_rmse"]),
            float(best_selection["dev_well_equal_rmse"]),
        ):
            best_selection = {
                "alpha": float(alpha),
                "smoother": candidate_best["smoother"],
                "smoother_name": candidate_best["smoother_name"],
                "well_weight": float(candidate_best["well_weight"]),
                "row_weight": float(candidate_best["row_weight"]),
                "dev_oof_rmse": float(candidate_best["dev_oof_rmse"]),
                "dev_well_equal_rmse": float(candidate_best["dev_well_equal_rmse"]),
                "dev_bias": float(candidate_best["dev_bias"]),
            }

    if best_selection is None:
        raise RuntimeError("no smoothed component selection was produced")

    final_well = gbr_d4()
    final_well.fit(Xw[dev_well_positions], well_target[dev_well_positions])
    held_well_correction = final_well.predict(Xw[held_well_positions]).astype(np.float32)
    del final_well
    gc.collect()

    final_row = ridge(float(best_selection["alpha"]))
    final_row.fit(X[dev], residual[dev])
    held_row_residual = final_row.predict(X[heldout]).astype(np.float32)
    del final_row, X
    gc.collect()

    correction_by_well = np.zeros(len(well_ids), dtype=np.float32)
    correction_by_well[held_well_positions] = held_well_correction
    held_well_component = correction_by_well[row_well_pos][heldout]
    smoother = dict(best_selection["smoother"])
    smooth_base_held = apply_by_well(base[heldout], wells[heldout], smoother)
    smooth_well_held = apply_by_well(held_well_component, wells[heldout], smoother)
    smooth_row_held = apply_by_well(held_row_residual, wells[heldout], smoother)
    clean_pred = (
        smooth_base_held
        + float(best_selection["well_weight"]) * smooth_well_held
        + float(best_selection["row_weight"]) * smooth_row_held
    ).astype(np.float32)
    raw_exp109_weight_pred = (
        base[heldout]
        + 0.525 * held_well_component
        - 0.14 * held_row_residual
    ).astype(np.float32)

    sub = make_submission(ids, clean_pred, heldout, SUBMISSION)
    payload = {
        "experiment_id": "exp114",
        "phase": "smoothed_component_weight_select",
        "selection_rule": (
            "base, well residual, and row residual components are smoothed separately; "
            "alpha, smoother, and residual weights are selected only on non-heldout OOF rows; "
            "sample-submission wells are used only for the final audit"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "row_feature_count": int(len(row_feature_names)),
        "alphas": ALPHAS,
        "well_weights": [float(w) for w in WELL_WEIGHTS],
        "row_weights": [float(w) for w in ROW_WEIGHTS],
        "selected": best_selection,
        "alpha_results": alpha_results,
        "heldout_raw_exp109_weight_metrics": audit_metrics(raw_exp109_weight_pred, y[heldout], wells[heldout]),
        "heldout_selected_metrics": audit_metrics(clean_pred, y[heldout], wells[heldout]),
        "submission_path": str(SUBMISSION.relative_to(ROOT)),
        "submission_report": submission_report(sub),
        "elapsed_seconds": time.time() - t0,
        "note": "Heldout labels are not used for fitting or selection.",
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
