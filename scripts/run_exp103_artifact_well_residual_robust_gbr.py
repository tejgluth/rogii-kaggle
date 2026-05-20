"""exp103: robust/stochastic GBR variants for artifact-only well residuals.

exp100's selected `gbr_d4` well residual model remains the best clean heldout
candidate.  exp101/exp102 improved non-heldout CV by ensembling or changing the
base, but their heldout audits were worse.  This sweep stays on the exp100 base
and tests conservative GBR variants that may generalize better while preserving
the same no-heldout-selection rule.
"""
from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from run_exp100_artifact_only_well_residual_gbr import (
    HELDOUT_WELLS,
    ROOT,
    WEIGHTS,
    build_well_meta,
    load_artifact_base,
    load_artifact_frame,
    rmse,
)


RESULT = ROOT / "experiments/results/exp103.json"
SUBMISSION = ROOT / "submissions/exp103_artifact_well_residual_robust_gbr_clean_oof.csv"


def model_factories() -> dict[str, Callable[[], object]]:
    return {
        "gbr_d4_base": lambda: GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        ),
        "gbr_d4_huber_a90": lambda: GradientBoostingRegressor(
            loss="huber",
            alpha=0.9,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        ),
        "gbr_d4_huber_a80": lambda: GradientBoostingRegressor(
            loss="huber",
            alpha=0.8,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        ),
        "gbr_d4_sub80": lambda: GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        ),
        "gbr_d4_sub65": lambda: GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.65,
            random_state=42,
        ),
        "gbr_d4_lr02_500": lambda: GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        ),
        "gbr_d4_lr015_700": lambda: GradientBoostingRegressor(
            n_estimators=700,
            learning_rate=0.015,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        ),
        "gbr_d5_l10": lambda: GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
        ),
        "gbr_d3_l3": lambda: GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            random_state=42,
        ),
        "gbr_d4_sqrt": lambda: GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
        ),
        "hgb_l2_10_leaf15": lambda: HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.025,
            l2_regularization=10.0,
            max_leaf_nodes=15,
            random_state=42,
        ),
    }


def score_well_oof(
    name: str,
    oof_well: np.ndarray,
    *,
    base: np.ndarray,
    y: np.ndarray,
    dev: np.ndarray,
    dev_well_positions: np.ndarray,
    row_well_pos: np.ndarray,
    well_count: int,
    seconds: float,
) -> dict[str, object]:
    correction_by_well = np.zeros(well_count, dtype=np.float32)
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
        "raw_dev_cv_rmse": rmse(base[dev] + row_correction[dev], y[dev]),
        "best_weight": best_weight["weight"],
        "best_dev_cv_rmse": best_weight["dev_cv_rmse"],
        "seconds": seconds,
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
    held_well_positions = np.flatnonzero(held_well_mask)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(dev_well_positions))

    factories = model_factories()
    results = []
    for name, factory in factories.items():
        oof_well = np.zeros(dev_well_positions.shape[0], dtype=np.float32)
        start = time.time()
        for tr_rel, va_rel in folds:
            tr_pos = dev_well_positions[tr_rel]
            va_pos = dev_well_positions[va_rel]
            model = factory()
            model.fit(Xw[tr_pos], well_target[tr_pos])
            oof_well[va_rel] = model.predict(Xw[va_pos]).astype(np.float32)
            del model
            gc.collect()
        row = score_well_oof(
            name,
            oof_well,
            base=base,
            y=y,
            dev=dev,
            dev_well_positions=dev_well_positions,
            row_well_pos=row_well_pos,
            well_count=len(well_ids),
            seconds=time.time() - start,
        )
        print(f"{name}: {row}", flush=True)
        results.append(row)

    selected = min(results, key=lambda row: row["best_dev_cv_rmse"])
    final_model = factories[str(selected["name"])]()
    final_model.fit(Xw[dev_well_positions], well_target[dev_well_positions])
    held_correction = final_model.predict(Xw[held_well_positions]).astype(np.float32)
    del final_model
    gc.collect()

    correction_by_well = np.zeros(len(well_ids), dtype=np.float32)
    correction_by_well[held_well_positions] = held_correction
    row_correction = correction_by_well[row_well_pos]
    clean_pred = base[heldout] + float(selected["best_weight"]) * row_correction[heldout]

    sample_ids = pd.read_csv(
        ROOT / "data/raw/rogii-wellbore-geology-prediction/sample_submission.csv", usecols=["id"]
    )
    clean_rows = pd.DataFrame({"id": ids[heldout], "tvt": clean_pred.astype(np.float32)})
    sub = sample_ids.merge(clean_rows, on="id", how="left")
    if int(sub["tvt"].isna().sum()):
        raise ValueError("missing clean robust GBR well residual predictions")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION, index=False)

    clean_holdout_rmse = rmse(clean_pred, y[heldout])
    held_correction_by_well = {
        well: float(corr) for well, corr in zip(well_ids[held_well_positions], held_correction, strict=True)
    }
    payload = {
        "experiment_id": "exp103",
        "phase": "artifact_well_residual_robust_gbr",
        "selection_rule": (
            "artifact-only equal OOF base; robust/stochastic GBR well residual variants and residual weight selected by "
            "5-fold CV over non-heldout wells only; final selected model trained on non-heldout wells and audited once "
            "on heldout wells"
        ),
        "heldout_wells": HELDOUT_WELLS.tolist(),
        "base_experiment": "artifact_only_equal_oof",
        "base_members": members,
        "feature_count": int(Xw.shape[1]),
        "weights": [float(w) for w in WEIGHTS],
        "base_dev_rmse": base_dev_rmse,
        "base_clean_holdout_rmse": base_holdout_rmse,
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
