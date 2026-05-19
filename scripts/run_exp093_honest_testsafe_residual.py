"""exp093: honest test-safe residual selection on top of exp028.

This is a stricter successor to exp092.

Guardrails:
- The three public test well IDs are excluded from every fitted residual model.
- Residual weight and smoothing parameters are selected only on OOF predictions
  from the remaining train wells.
- The local overlap diagnostic for the three test wells is scored with exp028
  OOF predictions for those wells, not exp028 averaged test predictions that may
  have been trained on the same wells.

The saved submission still uses the exp028 test prediction array as its base
because that is the available Kaggle inference artifact. The JSON result reports
both the strict clean-holdout score and the legacy test-pred diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
SAMPLE = DATA / "sample_submission.csv"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def public_test_wells() -> list[str]:
    return sorted(p.stem.replace("__horizontal_well", "") for p in TEST_DIR.glob("*__horizontal_well.csv"))


def load_test_frame(groups: np.ndarray, base_oof: np.ndarray) -> dict[str, np.ndarray]:
    truth, lateral, test_groups, known_tvt, clean_base = [], [], [], [], []
    for path in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
        well_id = path.stem.replace("__horizontal_well", "")
        test_df = pd.read_csv(path, usecols=["TVT_input"])
        train_df = pd.read_csv(TRAIN_DIR / f"{well_id}__horizontal_well.csv", usecols=["TVT"])
        well_mask = groups == well_id
        if int(well_mask.sum()) != len(test_df):
            raise ValueError(f"row mismatch for {well_id}: cache={int(well_mask.sum())} test={len(test_df)}")
        truth.append(train_df["TVT"].to_numpy(np.float32))
        tvt_input = test_df["TVT_input"].to_numpy(np.float32)
        known_tvt.append(tvt_input)
        lateral.append(np.isnan(tvt_input))
        test_groups.extend([well_id] * len(test_df))
        clean_base.append(base_oof[well_mask])
    return {
        "truth": np.concatenate(truth).astype(np.float32),
        "lateral": np.concatenate(lateral).astype(bool),
        "groups": np.asarray(test_groups).astype(str),
        "known_tvt": np.concatenate(known_tvt).astype(np.float32),
        "clean_base": np.concatenate(clean_base).astype(np.float32),
    }


def safe_feature_mask(feature_cols: np.ndarray) -> np.ndarray:
    blocked_exact = {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}
    return np.array([
        not str(col).startswith("fm_") and str(col) not in blocked_exact
        for col in feature_cols
    ])


def smooth_by_group(pred: np.ndarray, groups: np.ndarray, win: int, poly: int) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for well_id in np.unique(groups):
        idx = np.flatnonzero(groups == well_id)
        n = len(idx)
        w = min(int(win), n if n % 2 else n - 1)
        if w >= poly + 2:
            out[idx] = savgol_filter(out[idx], w, int(poly)).astype(np.float32)
    return out


def make_candidate(
    base: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    lateral: np.ndarray,
    known_tvt: np.ndarray,
    weight: float,
    win: int,
    poly: int,
    anchor_known: bool,
) -> np.ndarray:
    pred = (base + float(weight) * residual).astype(np.float32)
    if anchor_known:
        known = ~lateral & np.isfinite(known_tvt)
        pred[known] = known_tvt[known]
    if win:
        pred = smooth_by_group(pred, groups, win, poly)
        if anchor_known:
            known = ~lateral & np.isfinite(known_tvt)
            pred[known] = known_tvt[known]
    return pred.astype(np.float32)


def select_postprocess(
    base: np.ndarray,
    residual: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    lateral: np.ndarray,
    known_tvt: np.ndarray,
) -> dict:
    rows = []
    best = None
    weights = np.round(np.arange(-1.5, 1.5001, 0.05), 4)
    windows = [(0, 0), (17, 3), (31, 3), (51, 3), (75, 3), (101, 3), (151, 3), (201, 3), (301, 3)]
    known = ~lateral & np.isfinite(known_tvt)
    for win, poly in windows:
        for anchor_known in (False, True):
            base_part = base.copy().astype(np.float32)
            resid_part = residual.copy().astype(np.float32)
            if anchor_known:
                base_part[known] = known_tvt[known]
                resid_part[known] = 0.0
            if win:
                base_part = smooth_by_group(base_part, groups, win, poly)
                resid_part = smooth_by_group(resid_part, groups, win, poly)
                if anchor_known:
                    base_part[known] = known_tvt[known]
                    resid_part[known] = 0.0
            base_lat = base_part[lateral].astype(np.float64)
            resid_lat = resid_part[lateral].astype(np.float64)
            y_lat = y[lateral].astype(np.float64)
            for weight in weights:
                pred_lat = base_lat + float(weight) * resid_lat
                score = rmse(pred_lat, y_lat)
                rec = {
                    "rmse": score,
                    "mse": score * score,
                    "weight": float(weight),
                    "savgol_win": int(win),
                    "savgol_poly": int(poly),
                    "anchor_known": bool(anchor_known),
                }
                rows.append(rec)
                if best is None or score < best["rmse"]:
                    best = rec
    assert best is not None
    return {"best": best, "top": sorted(rows, key=lambda r: r["rmse"])[:40]}


def build_submission(preds: np.ndarray, output_path: Path) -> None:
    sample = pd.read_csv(SAMPLE)
    rows = []
    cursor = 0
    for path in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
        well_id = path.stem.replace("__horizontal_well", "")
        df = pd.read_csv(path, usecols=["TVT_input"])
        n = len(df)
        well_preds = preds[cursor:cursor + n]
        cursor += n
        lateral = df["TVT_input"].isna().to_numpy()
        rows.append(pd.DataFrame({
            "id": [f"{well_id}_{i}" for i in np.flatnonzero(lateral)],
            "tvt": well_preds[lateral],
        }))
    if cursor != len(preds):
        raise ValueError(f"prediction length mismatch: consumed {cursor}, got {len(preds)}")
    pred_df = pd.concat(rows, ignore_index=True)
    sub = sample[["id"]].merge(pred_df, on="id", how="left")
    missing = int(sub["tvt"].isna().sum())
    if missing:
        raise ValueError(f"missing {missing} submission predictions")
    sub.to_csv(output_path, index=False)


def main() -> None:
    t0 = time()
    cache = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
    X = cache["X"]
    Xt = cache["Xt"]
    y = cache["y_abs"].astype(np.float32)
    is_lateral = cache["is_lateral"].astype(bool)
    groups = np.asarray(cache["groups"]).astype(str)
    feature_cols = cache["feature_cols"].astype(str)

    base_oof = np.load(ROOT / "experiments/oof/oof_combined_exp028.npy").astype(np.float32)
    base_test = np.load(ROOT / "experiments/test_preds/test_combined_exp028.npy").astype(np.float32)

    heldout_wells = public_test_wells()
    heldout_mask = np.isin(groups, heldout_wells)
    dev_mask = ~heldout_mask
    train_lateral = dev_mask & is_lateral & np.isfinite(base_oof)

    keep = safe_feature_mask(feature_cols)
    Xs = X[:, keep]
    Xts = Xt[:, keep]
    kept_cols = feature_cols[keep].tolist()
    residual_target = (y - base_oof).astype(np.float32)

    print(
        f"dev_lateral_rows={int(train_lateral.sum())} heldout_wells={heldout_wells} "
        f"features={int(keep.sum())}",
        flush=True,
    )

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.025,
        num_leaves=63,
        min_data_in_leaf=800,
        feature_fraction=0.75,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=1.0,
        lambda_l2=10.0,
        verbose=-1,
        seed=9301,
        num_threads=-1,
    )

    residual_oof = np.zeros(len(y), dtype=np.float32)
    fold_rows = []
    dev_indices = np.flatnonzero(dev_mask)
    splitter = GroupKFold(5)
    for fold, (train_pos, val_pos) in enumerate(splitter.split(Xs[dev_indices], y[dev_indices], groups[dev_indices])):
        train_idx = dev_indices[train_pos]
        val_idx = dev_indices[val_pos]
        fit_idx = train_idx[is_lateral[train_idx] & np.isfinite(base_oof[train_idx])]
        model = lgb.train(
            params,
            lgb.Dataset(Xs[fit_idx], label=residual_target[fit_idx]),
            num_boost_round=650,
        )
        residual_oof[val_idx] = model.predict(Xs[val_idx]).astype(np.float32)
        val_lat = is_lateral[val_idx]
        base_score = rmse(base_oof[val_idx][val_lat], y[val_idx][val_lat])
        resid_score = rmse((base_oof[val_idx] + residual_oof[val_idx])[val_lat], y[val_idx][val_lat])
        fold_rows.append({
            "fold": fold,
            "fit_rows": int(len(fit_idx)),
            "val_wells": int(np.unique(groups[val_idx]).size),
            "base_rmse": base_score,
            "base_plus_w1_rmse": resid_score,
        })
        print(f"fold{fold}: base={base_score:.6f} base_plus_w1={resid_score:.6f}", flush=True)

    np.save(ROOT / "experiments/oof/oof_lgbm_exp093_resid.npy", residual_oof.astype(np.float32))

    dev_known_tvt = np.where(np.isfinite(X[:, np.where(feature_cols == "TVT_input")[0][0]]), y, np.nan).astype(np.float32)
    selection = select_postprocess(
        base_oof[dev_mask],
        residual_oof[dev_mask],
        y[dev_mask],
        groups[dev_mask],
        is_lateral[dev_mask],
        dev_known_tvt[dev_mask],
    )
    best = selection["best"]
    print(f"selected={best}", flush=True)

    final_model = lgb.train(
        params,
        lgb.Dataset(Xs[train_lateral], label=residual_target[train_lateral]),
        num_boost_round=650,
    )
    residual_test = final_model.predict(Xts).astype(np.float32)

    test_frame = load_test_frame(groups, base_oof)
    clean_pred = make_candidate(
        test_frame["clean_base"],
        residual_test,
        test_frame["groups"],
        test_frame["lateral"],
        test_frame["known_tvt"],
        best["weight"],
        best["savgol_win"],
        best["savgol_poly"],
        best["anchor_known"],
    )
    legacy_pred = make_candidate(
        base_test,
        residual_test,
        test_frame["groups"],
        test_frame["lateral"],
        test_frame["known_tvt"],
        best["weight"],
        best["savgol_win"],
        best["savgol_poly"],
        best["anchor_known"],
    )

    clean_base_rmse = rmse(test_frame["clean_base"][test_frame["lateral"]], test_frame["truth"][test_frame["lateral"]])
    clean_rmse = rmse(clean_pred[test_frame["lateral"]], test_frame["truth"][test_frame["lateral"]])
    legacy_base_rmse = rmse(base_test[test_frame["lateral"]], test_frame["truth"][test_frame["lateral"]])
    legacy_rmse = rmse(legacy_pred[test_frame["lateral"]], test_frame["truth"][test_frame["lateral"]])

    pred_path = ROOT / "experiments/test_preds/test_stack_exp093.npy"
    sub_path = ROOT / "submissions/exp093_honest_testsafe_residual.csv"
    result_path = ROOT / "experiments/results/exp093.json"
    np.save(pred_path, legacy_pred.astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_lgbm_exp093_resid.npy", residual_test.astype(np.float32))
    build_submission(legacy_pred.astype(np.float32), sub_path)

    payload = {
        "experiment_id": "exp093",
        "phase": "honest_testsafe_residual",
        "base": "exp028",
        "heldout_wells": heldout_wells,
        "features_used": kept_cols,
        "residual_params": params | {"num_boost_round": 650},
        "dev_selection": selection,
        "folds": fold_rows,
        "clean_holdout_base_rmse": clean_base_rmse,
        "clean_holdout_base_mse": clean_base_rmse * clean_base_rmse,
        "clean_holdout_rmse": clean_rmse,
        "clean_holdout_mse": clean_rmse * clean_rmse,
        "legacy_testpred_base_rmse": legacy_base_rmse,
        "legacy_testpred_base_mse": legacy_base_rmse * legacy_base_rmse,
        "legacy_testpred_rmse": legacy_rmse,
        "legacy_testpred_mse": legacy_rmse * legacy_rmse,
        "test_pred_path": str(pred_path.relative_to(ROOT)),
        "submission_path": str(sub_path.relative_to(ROOT)),
        "elapsed_seconds": time() - t0,
        "note": (
            "clean_holdout uses exp028 OOF predictions for held-out test wells; "
            "legacy_testpred uses the existing exp028 test prediction artifact for submission compatibility."
        ),
    }
    result_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:8000], flush=True)


if __name__ == "__main__":
    main()
