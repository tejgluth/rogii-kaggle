"""exp111: exp109 plus OOF-selected per-well postprocess.

This experiment keeps exp109's model family fixed.  It tests only test-time
sequence postprocessing selected from non-heldout OOF predictions.  The three
sample-submission wells remain audit-only and do not participate in smoother
selection.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
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


RESULT = ROOT / "experiments/results/exp111.json"
SUBMISSION = ROOT / "submissions/exp111_exp109_oof_postprocess.csv"
EXPERIMENT_ID = "exp111"
PHASE = "exp109_oof_selected_postprocess"

ALPHA = 0.1
WELL_WEIGHT = 0.525
ROW_WEIGHT = -0.14
DEV_IMPROVEMENT_EPS = 0.002


def _odd_window(requested: int, n: int, polyorder: int = 0) -> int | None:
    if n <= polyorder + 2:
        return None
    window = min(int(requested), n if n % 2 else n - 1)
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window > n:
        window = n if n % 2 else n - 1
    if window <= polyorder or window < 3:
        return None
    return window


def smooth_values(values: np.ndarray, candidate: dict[str, object]) -> np.ndarray:
    kind = str(candidate["kind"])
    if kind == "none":
        return values.astype(np.float32, copy=True)

    shrink = float(candidate["shrink"])
    smoothed = values.astype(np.float64, copy=True)
    if kind == "savgol":
        polyorder = int(candidate["polyorder"])
        window = _odd_window(int(candidate["window"]), len(values), polyorder)
        if window is None:
            return values.astype(np.float32, copy=True)
        target = savgol_filter(smoothed, window_length=window, polyorder=polyorder, mode="interp")
    elif kind == "median":
        window = int(candidate["window"])
        target = (
            pd.Series(smoothed)
            .rolling(window=window, min_periods=1, center=True)
            .median()
            .to_numpy(dtype=np.float64)
        )
    elif kind == "median_savgol":
        median_window = int(candidate["median_window"])
        first = (
            pd.Series(smoothed)
            .rolling(window=median_window, min_periods=1, center=True)
            .median()
            .to_numpy(dtype=np.float64)
        )
        polyorder = int(candidate["polyorder"])
        window = _odd_window(int(candidate["window"]), len(values), polyorder)
        if window is None:
            target = first
        else:
            target = savgol_filter(first, window_length=window, polyorder=polyorder, mode="interp")
    else:
        raise ValueError(f"unknown candidate kind: {kind}")

    return (smoothed + shrink * (target - smoothed)).astype(np.float32)


def apply_by_well(pred: np.ndarray, wells: np.ndarray, candidate: dict[str, object]) -> np.ndarray:
    out = pred.astype(np.float32, copy=True)
    frame = pd.DataFrame({"well": wells.astype(str), "pos": np.arange(len(wells), dtype=np.int64)})
    for positions in frame.groupby("well", sort=False)["pos"]:
        idx = positions[1].to_numpy(dtype=np.int64)
        out[idx] = smooth_values(out[idx], candidate)
    return out


def postprocess_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = [{"name": "none", "kind": "none"}]
    for window in (101, 201, 301, 501, 801, 1001):
        for polyorder in (2, 3):
            for shrink in (0.25, 0.5, 0.75, 1.0):
                candidates.append(
                    {
                        "name": f"savgol_w{window}_p{polyorder}_s{shrink:g}",
                        "kind": "savgol",
                        "window": window,
                        "polyorder": polyorder,
                        "shrink": shrink,
                    }
                )
    for window in (51, 101, 201):
        for shrink in (0.25, 0.5, 0.75, 1.0):
            candidates.append(
                {
                    "name": f"median_w{window}_s{shrink:g}",
                    "kind": "median",
                    "window": window,
                    "shrink": shrink,
                }
            )
    for shrink in (0.25, 0.5, 0.75):
        candidates.append(
            {
                "name": f"median101_savgol301_p2_s{shrink:g}",
                "kind": "median_savgol",
                "median_window": 101,
                "window": 301,
                "polyorder": 2,
                "shrink": shrink,
            }
        )
    return candidates


def best_candidate(scores: list[dict[str, object]]) -> dict[str, object]:
    baseline = next(row for row in scores if row["name"] == "none")
    ranked = sorted(scores, key=lambda row: (float(row["dev_oof_rmse"]), float(row["dev_well_equal_rmse"])))
    candidate = ranked[0]
    if (
        candidate["name"] == "none"
        or float(candidate["dev_oof_rmse"]) < float(baseline["dev_oof_rmse"]) - DEV_IMPROVEMENT_EPS
    ):
        return candidate
    return baseline | {
        "selection_note": (
            f"best smoother improved less than {DEV_IMPROVEMENT_EPS} RMSE on non-heldout OOF; "
            "kept exp109 raw predictions"
        )
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

    row_oof = np.zeros(idx_dev.shape[0], dtype=np.float32)
    group_folds = list(GroupKFold(n_splits=3).split(np.zeros(idx_dev.shape[0]), groups=wells[dev]))
    row_fold_scores = []
    for fold_idx, (tr_rel, va_rel) in enumerate(group_folds):
        tr_idx = idx_dev[tr_rel]
        va_idx = idx_dev[va_rel]
        model = ridge(ALPHA)
        model.fit(X[tr_idx], residual[tr_idx])
        pred = model.predict(X[va_idx]).astype(np.float32)
        row_oof[va_rel] = pred
        row_fold_scores.append(
            {
                "fold": fold_idx,
                "base_rmse": rmse(base[va_idx], y[va_idx]),
                "raw_row_rmse": rmse(base[va_idx] + pred, y[va_idx]),
                "rows": int(va_idx.size),
                "wells": int(np.unique(wells[va_idx]).size),
            }
        )
        print(f"row_oof fold={fold_idx} score={row_fold_scores[-1]}", flush=True)
        del model
        gc.collect()

    dev_raw = (
        base[idx_dev]
        + WELL_WEIGHT * well_row_oof[idx_dev]
        + ROW_WEIGHT * row_oof
    ).astype(np.float32)
    y_dev = y[idx_dev]
    wells_dev = wells[idx_dev]

    candidate_scores = []
    for candidate in postprocess_candidates():
        start = time.time()
        dev_pp = apply_by_well(dev_raw, wells_dev, candidate)
        score = {
            "name": candidate["name"],
            "kind": candidate["kind"],
            "candidate": candidate,
            "dev_oof_rmse": rmse(dev_pp, y_dev),
            "dev_well_equal_rmse": well_equal_rmse(dev_pp, y_dev, wells_dev),
            "dev_bias": float(np.mean(dev_pp.astype(np.float64) - y_dev.astype(np.float64))),
            "seconds": time.time() - start,
        }
        candidate_scores.append(score)
        print(f"postprocess {score}", flush=True)
        del dev_pp
        gc.collect()

    selected = best_candidate(candidate_scores)
    selected_candidate = dict(selected["candidate"])

    final_well = gbr_d4()
    final_well.fit(Xw[dev_well_positions], well_target[dev_well_positions])
    held_well_correction = final_well.predict(Xw[held_well_positions]).astype(np.float32)
    del final_well
    gc.collect()

    final_row = ridge(ALPHA)
    final_row.fit(X[dev], residual[dev])
    held_row_residual = final_row.predict(X[heldout]).astype(np.float32)
    del final_row, X
    gc.collect()

    correction_by_well = np.zeros(len(well_ids), dtype=np.float32)
    correction_by_well[held_well_positions] = held_well_correction
    held_raw = (
        base[heldout]
        + WELL_WEIGHT * correction_by_well[row_well_pos][heldout]
        + ROW_WEIGHT * held_row_residual
    ).astype(np.float32)
    held_pp = apply_by_well(held_raw, wells[heldout], selected_candidate)

    sub = make_submission(ids, held_pp, heldout, SUBMISSION)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "phase": PHASE,
        "selection_rule": (
            "exp109 model constants are fixed; per-well postprocess candidates are selected only on "
            "non-heldout OOF predictions; the three sample-submission wells are audited only after selection"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "fixed_exp109_params": {
            "alpha": ALPHA,
            "well_weight": WELL_WEIGHT,
            "row_weight": ROW_WEIGHT,
        },
        "row_feature_count": int(len(row_feature_names)),
        "dev_raw_metrics": audit_metrics(dev_raw, y_dev, wells_dev),
        "selected": selected,
        "candidate_results": sorted(
            candidate_scores,
            key=lambda row: (float(row["dev_oof_rmse"]), float(row["dev_well_equal_rmse"])),
        ),
        "heldout_raw_metrics": audit_metrics(held_raw, y[heldout], wells[heldout]),
        "heldout_selected_metrics": audit_metrics(held_pp, y[heldout], wells[heldout]),
        "row_fold_scores": row_fold_scores,
        "submission_path": str(SUBMISSION.relative_to(ROOT)),
        "submission_report": submission_report(sub),
        "elapsed_seconds": time.time() - t0,
        "note": "Heldout labels are not used for fitting or postprocess selection.",
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
