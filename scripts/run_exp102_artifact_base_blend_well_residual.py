"""exp102: artifact base blend sweep with exp100-style well residuals.

The exp100/exp101 residual layer used the equal average of all artifact OOFs as
its base.  This experiment keeps the same clean protocol but lets non-heldout
OOF RMSE choose among simple artifact-only base blends before fitting a fixed
`gbr_d4` well residual correction.

Heldout/sample wells are excluded from base ranking, residual fitting, model
selection, and residual-shrink selection.  They are audited once after the
non-heldout winner is selected.
"""
from __future__ import annotations

import gc
import json
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from run_exp100_artifact_only_well_residual_gbr import (
    ART,
    HELDOUT_WELLS,
    ROOT,
    WEIGHTS,
    build_well_meta,
    load_artifact_frame,
    rmse,
)


RESULT = ROOT / "experiments/results/exp102.json"
SUBMISSION = ROOT / "submissions/exp102_artifact_base_blend_well_residual_clean_oof.csv"


def gbr_d4() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )


def load_artifact_members(last_known_tvt: np.ndarray) -> dict[str, np.ndarray]:
    members: dict[str, np.ndarray] = {}
    for path in sorted(ART.glob("*_oof_preds.pkl")):
        name = path.name.replace("_oof_preds.pkl", "")
        members[name] = (last_known_tvt + joblib.load(path).astype(np.float32)).astype(np.float32)
    return members


def base_candidates(single_dev_rank: list[str]) -> OrderedDict[str, list[str]]:
    candidates: OrderedDict[str, list[str]] = OrderedDict()
    for name in single_dev_rank[:3]:
        candidates[f"single_{name}"] = [name]
    for size in range(2, len(single_dev_rank) + 1):
        candidates[f"mean_dev_top{size}"] = single_dev_rank[:size]
    catboost = [name for name in single_dev_rank if name.startswith("catboost")]
    lightgbm = [name for name in single_dev_rank if name.startswith("lightgbm")]
    candidates["mean_catboost_all"] = catboost
    candidates["mean_lightgbm_all"] = lightgbm
    candidates["mean_all_artifacts"] = single_dev_rank
    return candidates


def score_candidate(
    name: str,
    members: list[str],
    base: np.ndarray,
    residual: np.ndarray,
    *,
    y: np.ndarray,
    wells: np.ndarray,
    dev: np.ndarray,
    Xw: np.ndarray,
    well_ids: np.ndarray,
    row_well_pos: np.ndarray,
    dev_well_positions: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    well_target = (
        pd.DataFrame({"well": wells, "residual": residual})
        .groupby("well", sort=True)["residual"]
        .mean()
        .loc[well_ids]
        .to_numpy(np.float32)
    )
    oof_well = np.zeros(dev_well_positions.shape[0], dtype=np.float32)
    start = time.time()
    for tr_rel, va_rel in folds:
        tr_pos = dev_well_positions[tr_rel]
        va_pos = dev_well_positions[va_rel]
        model = gbr_d4()
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
    return {
        "name": name,
        "members": members,
        "base_dev_rmse": rmse(base[dev], y[dev]),
        "raw_residual_dev_cv_rmse": rmse(base[dev] + row_correction[dev], y[dev]),
        "best_weight": best_weight["weight"],
        "best_dev_cv_rmse": best_weight["dev_cv_rmse"],
        "seconds": time.time() - start,
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

    artifact_members = load_artifact_members(last)
    single_scores = [
        {
            "name": name,
            "dev_rmse": rmse(pred[dev], y[dev]),
        }
        for name, pred in artifact_members.items()
    ]
    single_dev_rank = [row["name"] for row in sorted(single_scores, key=lambda row: row["dev_rmse"])]
    candidates = base_candidates(single_dev_rank)

    meta = build_well_meta(df, feature_cols)
    well_ids = meta.index.astype(str).to_numpy()
    Xw = meta.to_numpy(np.float32)
    well_to_pos = {well: pos for pos, well in enumerate(well_ids)}
    row_well_pos = np.array([well_to_pos[well] for well in wells], dtype=np.int32)
    held_well_mask = np.isin(well_ids, HELDOUT_WELLS)
    dev_well_positions = np.flatnonzero(~held_well_mask)
    held_well_positions = np.flatnonzero(held_well_mask)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(dev_well_positions))

    results = []
    base_by_name: dict[str, np.ndarray] = {}
    for name, member_names in candidates.items():
        base = np.mean([artifact_members[member] for member in member_names], axis=0).astype(np.float32)
        base_by_name[name] = base
        residual = (y - base).astype(np.float32)
        row = score_candidate(
            name,
            member_names,
            base,
            residual,
            y=y,
            wells=wells,
            dev=dev,
            Xw=Xw,
            well_ids=well_ids,
            row_well_pos=row_well_pos,
            dev_well_positions=dev_well_positions,
            folds=folds,
        )
        print(f"{name}: {row}", flush=True)
        results.append(row)

    selected = min(results, key=lambda row: row["best_dev_cv_rmse"])
    selected_base = base_by_name[str(selected["name"])]
    selected_residual = (y - selected_base).astype(np.float32)
    well_target = (
        pd.DataFrame({"well": wells, "residual": selected_residual})
        .groupby("well", sort=True)["residual"]
        .mean()
        .loc[well_ids]
        .to_numpy(np.float32)
    )
    final_model = gbr_d4()
    final_model.fit(Xw[dev_well_positions], well_target[dev_well_positions])
    held_correction = final_model.predict(Xw[held_well_positions]).astype(np.float32)
    del final_model
    gc.collect()

    correction_by_well = np.zeros(len(well_ids), dtype=np.float32)
    correction_by_well[held_well_positions] = held_correction
    row_correction = correction_by_well[row_well_pos]
    clean_pred = selected_base[heldout] + float(selected["best_weight"]) * row_correction[heldout]

    sample_ids = pd.read_csv(
        ROOT / "data/raw/rogii-wellbore-geology-prediction/sample_submission.csv", usecols=["id"]
    )
    clean_rows = pd.DataFrame({"id": ids[heldout], "tvt": clean_pred.astype(np.float32)})
    sub = sample_ids.merge(clean_rows, on="id", how="left")
    if int(sub["tvt"].isna().sum()):
        raise ValueError("missing clean artifact base blend predictions")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION, index=False)

    clean_holdout_rmse = rmse(clean_pred, y[heldout])
    held_correction_by_well = {
        well: float(corr) for well, corr in zip(well_ids[held_well_positions], held_correction, strict=True)
    }
    payload = {
        "experiment_id": "exp102",
        "phase": "artifact_base_blend_well_residual",
        "selection_rule": (
            "artifact-only base candidates are built from non-heldout single-model OOF ranking; "
            "base candidate and gbr_d4 residual weight are selected by 5-fold CV over non-heldout wells only; "
            "final residual model is trained on non-heldout wells and audited once on heldout wells"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "single_model_dev_rank": sorted(single_scores, key=lambda row: row["dev_rmse"]),
        "candidate_count": len(results),
        "feature_count": int(Xw.shape[1]),
        "weights": [float(w) for w in WEIGHTS],
        "selected": selected,
        "clean_holdout_rmse": clean_holdout_rmse,
        "clean_holdout_mse": clean_holdout_rmse**2,
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
