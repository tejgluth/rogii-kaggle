"""exp115: independently smooth base and row residual components.

exp114 applied one smoother to the base, well residual, and row residual
components.  The well residual is constant within each well, so smoothing it is
effectively a no-op.  This experiment keeps the same leak-safe residual system,
but lets the base artifact prediction and row residual use separate smoothers.

All alpha, smoother, and residual-weight selection is performed on non-heldout
OOF rows only.  The sample-submission wells are used only for the final audit.
"""
from __future__ import annotations

import gc
import json
import time
from collections.abc import Iterable

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


RESULT = ROOT / "experiments/results/exp115.json"
SUBMISSION = ROOT / "submissions/exp115_separate_component_smoothers.csv"

ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1]
WELL_WEIGHTS = np.arange(0.4875, 0.5625 + 1e-12, 0.00625)
ROW_WEIGHTS = np.arange(-0.18, -0.115 + 1e-12, 0.005)
CHUNK_ROWS = 250_000


def smoother_candidates(prefix: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = [{"name": f"{prefix}_none", "kind": "none", "base_name": "none"}]
    for window in (501, 531, 551, 571, 601, 631):
        for shrink in (0.75, 0.825, 0.875, 0.925, 1.0):
            base_name = f"savgol_w{window}_p2_s{shrink:g}"
            candidates.append(
                {
                    "name": f"{prefix}_{base_name}",
                    "base_name": base_name,
                    "kind": "savgol",
                    "window": window,
                    "polyorder": 2,
                    "shrink": shrink,
                }
            )
    return candidates


def _candidate_for_apply(candidate: dict[str, object]) -> dict[str, object]:
    if candidate["kind"] == "none":
        return {"name": "none", "kind": "none"}
    return {
        "name": candidate["base_name"],
        "kind": candidate["kind"],
        "window": candidate["window"],
        "polyorder": candidate["polyorder"],
        "shrink": candidate["shrink"],
    }


def build_smoothed_matrix(
    values: np.ndarray,
    wells: np.ndarray,
    candidates: list[dict[str, object]],
    *,
    label: str,
) -> np.ndarray:
    out = np.empty((values.shape[0], len(candidates)), dtype=np.float32)
    for col, candidate in enumerate(candidates):
        start = time.time()
        out[:, col] = apply_by_well(values, wells, _candidate_for_apply(candidate))
        print(f"{label} smoother {candidate['name']} built in {time.time() - start:.1f}s", flush=True)
    return out


def column_dots(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    out = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, matrix.shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, matrix.shape[0])
        out += matrix[start:stop].astype(np.float64).T @ vector[start:stop].astype(np.float64)
    return out


def column_squares(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, matrix.shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, matrix.shape[0])
        block = matrix[start:stop].astype(np.float64)
        out += np.sum(block * block, axis=0)
    return out


def cross_dots(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.zeros((left.shape[1], right.shape[1]), dtype=np.float64)
    for start in range(0, left.shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, left.shape[0])
        out += left[start:stop].astype(np.float64).T @ right[start:stop].astype(np.float64)
    return out


def best_weight_grid(
    *,
    rr: float,
    r_well: float,
    r_row: float,
    well_well: float,
    well_row: float,
    row_row: float,
    n_rows: int,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    n = float(n_rows)
    for well_weight in WELL_WEIGHTS:
        ww = float(well_weight)
        for row_weight in ROW_WEIGHTS:
            rw = float(row_weight)
            mse = (
                rr
                - 2.0 * ww * r_well
                - 2.0 * rw * r_row
                + ww * ww * well_well
                + 2.0 * ww * rw * well_row
                + rw * rw * row_row
            ) / n
            score = float(np.sqrt(max(mse, 0.0)))
            if best is None or score < best["dev_oof_rmse"]:
                best = {
                    "well_weight": ww,
                    "row_weight": rw,
                    "dev_oof_rmse": score,
                }
    if best is None:
        raise RuntimeError("no weight-grid result produced")
    return best


def evaluate_component_grid(
    *,
    y: np.ndarray,
    base_matrix: np.ndarray,
    row_matrix: np.ndarray,
    well_component: np.ndarray,
    base_candidates: list[dict[str, object]],
    row_candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    y64 = y.astype(np.float64)
    well64 = well_component.astype(np.float64)
    yy = float(np.dot(y64, y64))
    y_well = float(np.dot(y64, well64))
    well_well = float(np.dot(well64, well64))
    y_base = column_dots(base_matrix, y)
    base_base = column_squares(base_matrix)
    base_well = column_dots(base_matrix, well_component)
    y_row = column_dots(row_matrix, y)
    row_row = column_squares(row_matrix)
    row_well = column_dots(row_matrix, well_component)
    base_row = cross_dots(base_matrix, row_matrix)

    rows: list[dict[str, object]] = []
    n_rows = int(y.shape[0])
    for base_idx, base_candidate in enumerate(base_candidates):
        rr = yy - 2.0 * y_base[base_idx] + base_base[base_idx]
        r_well = y_well - base_well[base_idx]
        for row_idx, row_candidate in enumerate(row_candidates):
            r_row = y_row[row_idx] - base_row[base_idx, row_idx]
            combo = best_weight_grid(
                rr=rr,
                r_well=r_well,
                r_row=r_row,
                well_well=well_well,
                well_row=row_well[row_idx],
                row_row=row_row[row_idx],
                n_rows=n_rows,
            )
            rows.append(
                {
                    "base_smoother": _candidate_for_apply(base_candidate),
                    "base_smoother_name": str(base_candidate["base_name"]),
                    "row_smoother": _candidate_for_apply(row_candidate),
                    "row_smoother_name": str(row_candidate["base_name"]),
                    "well_weight": combo["well_weight"],
                    "row_weight": combo["row_weight"],
                    "dev_oof_rmse": combo["dev_oof_rmse"],
                }
            )
    rows.sort(key=lambda row: float(row["dev_oof_rmse"]))
    return rows


def add_selected_diagnostics(
    rows: Iterable[dict[str, object]],
    *,
    y: np.ndarray,
    wells: np.ndarray,
    base_matrix: np.ndarray,
    row_matrix: np.ndarray,
    well_component: np.ndarray,
    base_candidates: list[dict[str, object]],
    row_candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    base_pos = {str(candidate["base_name"]): pos for pos, candidate in enumerate(base_candidates)}
    row_pos = {str(candidate["base_name"]): pos for pos, candidate in enumerate(row_candidates)}
    enriched = []
    for row in rows:
        base_idx = base_pos[str(row["base_smoother_name"])]
        row_idx = row_pos[str(row["row_smoother_name"])]
        pred = (
            base_matrix[:, base_idx]
            + float(row["well_weight"]) * well_component
            + float(row["row_weight"]) * row_matrix[:, row_idx]
        ).astype(np.float32)
        enriched.append(
            row
            | {
                "dev_well_equal_rmse": well_equal_rmse(pred, y, wells),
                "dev_bias": float(np.mean(pred.astype(np.float64) - y.astype(np.float64))),
            }
        )
    return enriched


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

    y_dev = y[idx_dev]
    wells_dev = wells[idx_dev]
    well_dev = well_row_oof[idx_dev].astype(np.float32)
    base_candidates = smoother_candidates("base")
    row_candidates = smoother_candidates("row")
    base_matrix = build_smoothed_matrix(base[idx_dev], wells_dev, base_candidates, label="base")

    group_folds = list(GroupKFold(n_splits=3).split(np.zeros(idx_dev.shape[0]), groups=wells[dev]))
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

        row_matrix = build_smoothed_matrix(row_oof, wells_dev, row_candidates, label=f"row alpha={alpha:g}")
        pair_rows = evaluate_component_grid(
            y=y_dev,
            base_matrix=base_matrix,
            row_matrix=row_matrix,
            well_component=well_dev,
            base_candidates=base_candidates,
            row_candidates=row_candidates,
        )
        top_rows = add_selected_diagnostics(
            pair_rows[:25],
            y=y_dev,
            wells=wells_dev,
            base_matrix=base_matrix,
            row_matrix=row_matrix,
            well_component=well_dev,
            base_candidates=base_candidates,
            row_candidates=row_candidates,
        )
        alpha_best = sorted(top_rows, key=lambda row: (float(row["dev_oof_rmse"]), float(row["dev_well_equal_rmse"])))[0]
        alpha_best = alpha_best | {"alpha": float(alpha)}
        alpha_results.append(
            {
                "alpha": float(alpha),
                "fold_scores": fold_scores,
                "best": alpha_best,
                "top_pairs": top_rows,
                "seconds": time.time() - alpha_start,
            }
        )
        print(
            "alpha={alpha:g} best rmse={rmse:.6f} base={base_name} row={row_name} ww={ww:.4f} rw={rw:.4f}".format(
                alpha=alpha,
                rmse=float(alpha_best["dev_oof_rmse"]),
                base_name=alpha_best["base_smoother_name"],
                row_name=alpha_best["row_smoother_name"],
                ww=float(alpha_best["well_weight"]),
                rw=float(alpha_best["row_weight"]),
            ),
            flush=True,
        )
        if best_selection is None or (
            float(alpha_best["dev_oof_rmse"]),
            float(alpha_best["dev_well_equal_rmse"]),
        ) < (
            float(best_selection["dev_oof_rmse"]),
            float(best_selection["dev_well_equal_rmse"]),
        ):
            best_selection = dict(alpha_best)
        del row_matrix, row_oof
        gc.collect()

    if best_selection is None:
        raise RuntimeError("no separate-smoother selection was produced")

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
    smooth_base_held = apply_by_well(base[heldout], wells[heldout], dict(best_selection["base_smoother"]))
    smooth_row_held = apply_by_well(held_row_residual, wells[heldout], dict(best_selection["row_smoother"]))
    clean_pred = (
        smooth_base_held
        + float(best_selection["well_weight"]) * held_well_component
        + float(best_selection["row_weight"]) * smooth_row_held
    ).astype(np.float32)

    sub = make_submission(ids, clean_pred, heldout, SUBMISSION)
    payload = {
        "experiment_id": "exp115",
        "phase": "separate_component_smoothers",
        "selection_rule": (
            "base smoother, row-residual smoother, row alpha, and residual weights are selected only on "
            "non-heldout OOF rows; sample-submission wells are used only for the final audit"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "row_feature_count": int(len(row_feature_names)),
        "alphas": [float(a) for a in ALPHAS],
        "well_weights": [float(w) for w in WELL_WEIGHTS],
        "row_weights": [float(w) for w in ROW_WEIGHTS],
        "base_smoother_candidates": [_candidate_for_apply(candidate) for candidate in base_candidates],
        "row_smoother_candidates": [_candidate_for_apply(candidate) for candidate in row_candidates],
        "selected": best_selection,
        "alpha_results": alpha_results,
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
