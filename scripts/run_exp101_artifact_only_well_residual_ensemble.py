"""exp101: artifact-only well residual model/ensemble sweep.

This extends exp100 without touching the heldout/sample wells during selection:

- build the same artifact-only OOF base and well summary table
- train candidate well-level residual models with 5-fold CV on non-heldout wells
- compare row-count-weighted training variants and simple mean ensembles
- select the candidate and residual shrink by non-heldout row RMSE only
- train the selected member model(s) on all non-heldout wells and audit once on
  the three sample-submission wells
"""
from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

from run_exp100_artifact_only_well_residual_gbr import (
    HELDOUT_WELLS,
    ROOT,
    WEIGHTS,
    build_well_meta,
    load_artifact_base,
    load_artifact_frame,
    model_factories,
    rmse,
)


RESULT = ROOT / "experiments/results/exp101.json"
SUBMISSION = ROOT / "submissions/exp101_artifact_only_well_residual_ensemble_clean_oof.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    factory_name: str
    use_sample_weight: bool = False


def extended_factories() -> dict[str, Callable[[], object]]:
    factories = model_factories()
    factories.update(
        {
            "gbr_d2": lambda: GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=5,
                random_state=42,
            ),
            "gbr_d3_lr02": lambda: GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=3,
                min_samples_leaf=5,
                random_state=42,
            ),
            "gbr_d4_l10": lambda: GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=4,
                min_samples_leaf=10,
                random_state=42,
            ),
            "et_d10_l3": lambda: ExtraTreesRegressor(
                n_estimators=800,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            "rf_d10_l3": lambda: RandomForestRegressor(
                n_estimators=800,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        }
    )
    return factories


def candidate_specs() -> list[CandidateSpec]:
    base = [
        "gbr_d4",
        "gbr_d3",
        "et_d8_l5",
        "et_d6_l5",
        "rf_d8_l5",
        "gbr_d2",
        "gbr_d3_lr02",
        "gbr_d4_l10",
        "et_d10_l3",
        "rf_d10_l3",
    ]
    specs = [CandidateSpec(name=name, factory_name=name) for name in base]
    specs.extend(CandidateSpec(name=f"{name}_roww", factory_name=name, use_sample_weight=True) for name in base)
    return specs


def fit_model(model: object, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=sample_weight)


def score_oof_correction(
    name: str,
    oof_well: np.ndarray,
    *,
    base: np.ndarray,
    y: np.ndarray,
    dev: np.ndarray,
    dev_well_positions: np.ndarray,
    row_well_pos: np.ndarray,
    well_count: int,
    members: list[str],
    candidate_type: str,
    seconds: float,
) -> dict[str, object]:
    correction_by_well = np.zeros(well_count, dtype=np.float32)
    correction_by_well[dev_well_positions] = oof_well.astype(np.float32)
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
        "type": candidate_type,
        "members": members,
        "raw_dev_cv_rmse": rmse(base[dev] + row_correction[dev], y[dev]),
        "best_weight": best_weight["weight"],
        "best_dev_cv_rmse": best_weight["dev_cv_rmse"],
        "seconds": seconds,
    }


def cv_predict_spec(
    spec: CandidateSpec,
    factory: Callable[[], object],
    *,
    Xw: np.ndarray,
    well_target: np.ndarray,
    well_weight: np.ndarray,
    dev_well_positions: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, float]:
    oof_well = np.zeros(dev_well_positions.shape[0], dtype=np.float32)
    start = time.time()
    for tr_rel, va_rel in folds:
        tr_pos = dev_well_positions[tr_rel]
        va_pos = dev_well_positions[va_rel]
        model = factory()
        sample_weight = well_weight[tr_pos] if spec.use_sample_weight else None
        fit_model(model, Xw[tr_pos], well_target[tr_pos], sample_weight)
        oof_well[va_rel] = model.predict(Xw[va_pos]).astype(np.float32)
        del model
        gc.collect()
    return oof_well, time.time() - start


def combo_definitions(available_names: set[str], ranked_names: list[str]) -> dict[str, list[str]]:
    combos = {
        "mean_gbr_d4_gbr_d3": ["gbr_d4", "gbr_d3"],
        "mean_gbr_d4_gbr_d3_roww": ["gbr_d4_roww", "gbr_d3_roww"],
        "mean_gbr_pair_all": ["gbr_d4", "gbr_d3", "gbr_d4_roww", "gbr_d3_roww"],
        "mean_exp100_top3": ["gbr_d4", "gbr_d3", "et_d8_l5"],
        "mean_exp100_top5": ["gbr_d4", "gbr_d3", "et_d8_l5", "et_d6_l5", "rf_d8_l5"],
        "mean_roww_top5": ["gbr_d4_roww", "gbr_d3_roww", "et_d8_l5_roww", "et_d6_l5_roww", "rf_d8_l5_roww"],
        "mean_gbr_extra": ["gbr_d4", "gbr_d3", "gbr_d2", "gbr_d3_lr02", "gbr_d4_l10"],
        "mean_gbr_extra_roww": [
            "gbr_d4_roww",
            "gbr_d3_roww",
            "gbr_d2_roww",
            "gbr_d3_lr02_roww",
            "gbr_d4_l10_roww",
        ],
    }
    for size in (2, 3, 5, 8):
        combos[f"mean_dev_rank_top{size}"] = ranked_names[:size]
    return {
        name: members
        for name, members in combos.items()
        if members and all(member in available_names for member in members)
    }


def predict_holdout_member(
    spec: CandidateSpec,
    factory: Callable[[], object],
    *,
    Xw: np.ndarray,
    well_target: np.ndarray,
    well_weight: np.ndarray,
    dev_well_positions: np.ndarray,
    held_well_positions: np.ndarray,
) -> np.ndarray:
    model = factory()
    sample_weight = well_weight[dev_well_positions] if spec.use_sample_weight else None
    fit_model(model, Xw[dev_well_positions], well_target[dev_well_positions], sample_weight)
    pred = model.predict(Xw[held_well_positions]).astype(np.float32)
    del model
    gc.collect()
    return pred


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
    well_stats = pd.DataFrame({"well": wells, "residual": residual}).groupby("well", sort=True)["residual"].agg(
        ["mean", "size"]
    )
    well_target = well_stats.loc[well_ids, "mean"].to_numpy(np.float32)
    well_weight = well_stats.loc[well_ids, "size"].to_numpy(np.float32)
    well_to_pos = {well: pos for pos, well in enumerate(well_ids)}
    row_well_pos = np.array([well_to_pos[well] for well in wells], dtype=np.int32)
    held_well_mask = np.isin(well_ids, HELDOUT_WELLS)
    dev_well_positions = np.flatnonzero(~held_well_mask)
    held_well_positions = np.flatnonzero(held_well_mask)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(dev_well_positions))

    factories = extended_factories()
    specs = candidate_specs()
    spec_by_name = {spec.name: spec for spec in specs}
    oof_by_name: dict[str, np.ndarray] = {}
    results: list[dict[str, object]] = []

    for spec in specs:
        oof_well, seconds = cv_predict_spec(
            spec,
            factories[spec.factory_name],
            Xw=Xw,
            well_target=well_target,
            well_weight=well_weight,
            dev_well_positions=dev_well_positions,
            folds=folds,
        )
        oof_by_name[spec.name] = oof_well
        row = score_oof_correction(
            spec.name,
            oof_well,
            base=base,
            y=y,
            dev=dev,
            dev_well_positions=dev_well_positions,
            row_well_pos=row_well_pos,
            well_count=len(well_ids),
            members=[spec.name],
            candidate_type="single",
            seconds=seconds,
        )
        print(f"{spec.name}: {row}", flush=True)
        results.append(row)

    ranked_names = [
        str(row["name"])
        for row in sorted((row for row in results if row["type"] == "single"), key=lambda row: row["best_dev_cv_rmse"])
    ]
    combos = combo_definitions(set(oof_by_name), ranked_names)
    for name, combo_members in combos.items():
        start = time.time()
        oof_well = np.mean([oof_by_name[member] for member in combo_members], axis=0).astype(np.float32)
        row = score_oof_correction(
            name,
            oof_well,
            base=base,
            y=y,
            dev=dev,
            dev_well_positions=dev_well_positions,
            row_well_pos=row_well_pos,
            well_count=len(well_ids),
            members=combo_members,
            candidate_type="mean_ensemble",
            seconds=time.time() - start,
        )
        print(f"{name}: {row}", flush=True)
        results.append(row)

    selected = min(results, key=lambda row: row["best_dev_cv_rmse"])
    held_member_preds = [
        predict_holdout_member(
            spec_by_name[member],
            factories[spec_by_name[member].factory_name],
            Xw=Xw,
            well_target=well_target,
            well_weight=well_weight,
            dev_well_positions=dev_well_positions,
            held_well_positions=held_well_positions,
        )
        for member in selected["members"]
    ]
    held_correction = np.mean(held_member_preds, axis=0).astype(np.float32)

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
        raise ValueError("missing clean artifact-only ensemble well residual predictions")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION, index=False)

    held_correction_by_well = {
        well: float(corr) for well, corr in zip(well_ids[held_well_positions], held_correction, strict=True)
    }
    clean_holdout_rmse = rmse(clean_pred, y[heldout])
    payload = {
        "experiment_id": "exp101",
        "phase": "artifact_only_well_level_residual_ensemble",
        "selection_rule": (
            "artifact-only base; single well residual models, row-count-weighted variants, and simple mean ensembles "
            "selected by 5-fold CV over non-heldout wells only; final selected member model(s) trained on non-heldout "
            "wells and audited once on heldout wells"
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
