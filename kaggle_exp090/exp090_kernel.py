"""ROGII exp090 Kaggle kernel.

This is a kernel-only competition submission notebook source. It trains from the
competition input files available at runtime, predicts the hidden test wells, and
writes /kaggle/working/submission.csv.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
from sklearn.model_selection import GroupKFold

import lightgbm as lgb
import xgboost as xgb


T0 = time.time()
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def log(message: str) -> None:
    elapsed = time.time() - T0
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def rmse(pred, true) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def find_data_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path(os.environ.get("DATA_ROOT", "")),
        Path("data/raw/rogii-wellbore-geology-prediction"),
    ]
    for root in candidates:
        if (
            root
            and (root / "train").exists()
            and (root / "test").exists()
            and any((root / "train").glob("*__horizontal_well.csv"))
            and any((root / "test").glob("*__horizontal_well.csv"))
        ):
            return root

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for train_dir in input_root.rglob("train"):
            root = train_dir.parent
            test_dir = root / "test"
            if (
                test_dir.exists()
                and any(train_dir.glob("*__horizontal_well.csv"))
                and any(test_dir.glob("*__horizontal_well.csv"))
            ):
                return root

    seen = []
    if input_root.exists():
        for path in sorted(input_root.glob("*")):
            seen.append(str(path))
    if not seen:
        raise FileNotFoundError(
            "/kaggle/input is empty. Add the competition data source to this notebook: "
            "right sidebar -> Add Input -> Competitions -> "
            "rogii-wellbore-geology-prediction, then commit again."
        )
    raise FileNotFoundError(
        "Could not find competition data root with train/test horizontal well CSVs. "
        f"Top-level /kaggle/input entries: {seen[:30]}"
    )


DATA_ROOT = find_data_root()
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
SAMPLE_SUBMISSION = DATA_ROOT / "sample_submission.csv"
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

XGB_DEVICE = "cuda" if shutil.which("nvidia-smi") else "cpu"
log(f"DATA_ROOT={DATA_ROOT}")
log(f"XGBoost device={XGB_DEVICE}")


def multi_scale_ncc_offsets(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out_tvt = []
    out_score = []
    for half_window in hws:
        win = 2 * half_window + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < win + 1 or nh == 0:
            fill = ktvt[-1] if len(ktvt) else 0.0
            out_tvt.append(np.full(nh, fill, dtype=np.float32))
            out_score.append(np.zeros(nh, dtype=np.float32))
            continue

        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float32)
        starts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(starts) == 0:
            fill = ktvt[-1] if len(ktvt) else 0.0
            out_tvt.append(np.full(nh, fill, dtype=np.float32))
            out_score.append(np.zeros(nh, dtype=np.float32))
            continue

        offsets = np.arange(win, dtype=np.int32)
        candidates = kg[starts[:, None] + offsets[None, :]].astype(np.float32)
        candidates = (candidates - candidates.mean(1, keepdims=True)) / (
            candidates.std(1, keepdims=True) + 1e-6
        )

        padded = np.pad(hg, half_window, mode="edge")
        windows = padded[np.arange(nh)[:, None] + offsets[None, :]].astype(np.float32)
        windows = (windows - windows.mean(1, keepdims=True)) / (windows.std(1, keepdims=True) + 1e-6)

        ncc = windows @ candidates.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out_tvt.append(ktvt[np.clip(starts[best] + half_window, 0, nk - 1)].astype(np.float32))
        out_score.append(score)
    return out_tvt, out_score


def build_well(hw_path: Path, tw_path: Path, is_train: bool) -> pd.DataFrame | None:
    well_id = hw_path.stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path)
    tw_exists = tw_path.exists()

    known_mask = hw["TVT_input"].notna()
    if not known_mask.any():
        return None
    known = hw[known_mask]
    if is_train and "TVT" not in hw.columns:
        return None

    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_md = float(last["MD"])
    last_x = float(last["X"])
    last_y = float(last["Y"])
    last_z = float(last["Z"])
    last_gr = float(last["GR"]) if pd.notna(last["GR"]) else float(known["GR"].mean())

    out = pd.DataFrame(index=hw.index)
    out["well_id"] = well_id
    out["MD"] = hw["MD"].to_numpy()
    out["X"] = hw["X"].to_numpy()
    out["Y"] = hw["Y"].to_numpy()
    out["Z"] = hw["Z"].to_numpy()
    out["GR"] = hw["GR"].to_numpy()
    out["TVT_input"] = hw["TVT_input"].to_numpy()
    out["is_lateral"] = (~known_mask).to_numpy()
    out["last_known_tvt"] = last_tvt
    out["last_known_md"] = last_md
    out["last_known_gr"] = last_gr
    out["md_since_anchor"] = hw["MD"].to_numpy() - last_md
    out["dx_since_anchor"] = hw["X"].to_numpy() - last_x
    out["dy_since_anchor"] = hw["Y"].to_numpy() - last_y
    out["dz_since_anchor"] = hw["Z"].to_numpy() - last_z
    out["xy_dist"] = np.sqrt(out["dx_since_anchor"] ** 2 + out["dy_since_anchor"] ** 2)

    gr = pd.Series(hw["GR"]).ffill().bfill().fillna(0)
    out["gr_filled"] = gr.to_numpy()
    out["gr_rm_30"] = gr.rolling(30, center=True, min_periods=1).mean().to_numpy()
    out["gr_rm_100"] = gr.rolling(100, center=True, min_periods=1).mean().to_numpy()
    out["gr_rs_30"] = gr.rolling(30, center=True, min_periods=1).std().fillna(0).to_numpy()

    ktvt = known["TVT_input"].to_numpy(np.float64)
    kz = known["Z"].to_numpy(np.float64)
    z_all = hw["Z"].to_numpy(np.float64)

    for fm in FORMATIONS:
        if fm in hw.columns and hw[fm].notna().any():
            f_all = hw[fm].to_numpy(np.float64)
            f_known = f_all[known_mask.to_numpy()]
            b_values = ktvt + kz - f_known
            b_full = float(np.median(b_values))
            b_late = float(np.median(b_values[-50:])) if len(b_values) >= 50 else b_full
            weights = np.exp(0.02 * np.arange(len(b_values)))
            weights /= weights.sum()
            b_wls = float(np.dot(weights, b_values))

            out[f"fm_{fm}_tvt"] = (-z_all + f_all + b_full).astype(np.float32)
            out[f"fm_{fm}_tvt_late"] = (-z_all + f_all + b_late).astype(np.float32)
            out[f"fm_{fm}_tvt_wls"] = (-z_all + f_all + b_wls).astype(np.float32)
            out[f"fm_{fm}_bwell"] = b_full
            out[f"fm_{fm}_bwell_late"] = b_late
            out[f"fm_{fm}_minus_z"] = (f_all - z_all).astype(np.float32)
            out[f"fm_{fm}_minus_lk"] = float(f_all[0] - last_tvt)
        else:
            for col in [f"fm_{fm}_tvt", f"fm_{fm}_tvt_late", f"fm_{fm}_tvt_wls", f"fm_{fm}_minus_z"]:
                out[col] = 0.0
            for col in [f"fm_{fm}_bwell", f"fm_{fm}_bwell_late", f"fm_{fm}_minus_lk"]:
                out[col] = 0.0

    fm_preds = np.column_stack([out[f"fm_{fm}_tvt"].to_numpy() for fm in FORMATIONS])
    out["fm_mean_tvt"] = fm_preds.mean(1)
    out["fm_std_tvt"] = fm_preds.std(1)
    out["fm_min_tvt"] = fm_preds.min(1)
    out["fm_max_tvt"] = fm_preds.max(1)
    out["fm_mean_minus_lk"] = out["fm_mean_tvt"].to_numpy() - last_tvt

    if tw_exists:
        kgr = known["GR"].ffill().bfill().fillna(0).to_numpy(np.float32)
        hgr = gr.to_numpy(np.float32)
        ncc_tvts, ncc_scores = multi_scale_ncc_offsets(kgr, ktvt.astype(np.float32), hgr, hws=(8, 15, 25))
        for i, half_window in enumerate([8, 15, 25]):
            out[f"ncc{half_window}_tvt"] = ncc_tvts[i]
            out[f"ncc{half_window}_score"] = ncc_scores[i]
        tvts = np.stack(ncc_tvts, 1)
        scores = np.stack(ncc_scores, 1)
        score_weights = np.exp(3.0 * scores)
        score_weights /= score_weights.sum(1, keepdims=True) + 1e-9
        out["ncc_ens_tvt"] = (tvts * score_weights).sum(1).astype(np.float32)
        out["ncc_max_score"] = scores.max(1)
    else:
        for half_window in [8, 15, 25]:
            out[f"ncc{half_window}_tvt"] = last_tvt
            out[f"ncc{half_window}_score"] = 0.0
        out["ncc_ens_tvt"] = last_tvt
        out["ncc_max_score"] = 0.0

    if is_train:
        out["TVT"] = hw["TVT"].to_numpy()
        out["delta_target"] = hw["TVT"].to_numpy() - last_tvt
    return out


def build_dataset():
    train_parts = []
    train_files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    for i, path in enumerate(train_files, start=1):
        tw_path = path.parent / (path.stem.replace("__horizontal_well", "__typewell") + ".csv")
        part = build_well(path, tw_path, is_train=True)
        if part is not None:
            train_parts.append(part)
        if i % 100 == 0:
            log(f"features train wells {i}/{len(train_files)}")
    train = pd.concat(train_parts, ignore_index=True)

    test_parts = []
    test_files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    for i, path in enumerate(test_files, start=1):
        tw_path = path.parent / (path.stem.replace("__horizontal_well", "__typewell") + ".csv")
        part = build_well(path, tw_path, is_train=False)
        if part is not None:
            test_parts.append(part)
        if i % 100 == 0:
            log(f"features test wells {i}/{len(test_files)}")
    test = pd.concat(test_parts, ignore_index=True)

    skip = {"well_id", "TVT", "delta_target", "is_lateral"}
    feature_cols = [c for c in train.columns if c not in skip]
    X = train[feature_cols].to_numpy(np.float32)
    Xt = test[feature_cols].to_numpy(np.float32)
    data = {
        "X": X,
        "Xt": Xt,
        "y_delta": train["delta_target"].to_numpy(np.float32),
        "y_abs": train["TVT"].to_numpy(np.float32),
        "last_known": train["last_known_tvt"].to_numpy(np.float32),
        "test_last_known": test["last_known_tvt"].to_numpy(np.float32),
        "is_lateral": train["is_lateral"].to_numpy(bool),
        "groups": train["well_id"].astype(str).to_numpy(),
        "test_groups": test["well_id"].astype(str).to_numpy(),
        "feature_cols": feature_cols,
        "test_files": test_files,
    }
    log(f"built features train={X.shape} test={Xt.shape}")
    return data


DATA = build_dataset()
X = DATA["X"]
Xt = DATA["Xt"]
y_delta = DATA["y_delta"]
y_abs = DATA["y_abs"]
last_known = DATA["last_known"]
test_last_known = DATA["test_last_known"]
is_lateral = DATA["is_lateral"]
groups = DATA["groups"]
test_groups = DATA["test_groups"]


def xgb_common_params() -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": XGB_DEVICE,
        "max_bin": 255,
        "verbosity": 0,
    }


def train_xgb_delta(tag: str, params: dict, num_boost_round: int, early_stopping_rounds: int):
    log(f"train {tag}")
    full_params = {**xgb_common_params(), **params}
    folds = GroupKFold(5).split(X, y_delta, groups)
    oof_delta = np.zeros(len(y_delta), dtype=np.float32)
    test_fold_preds = []
    dtest = xgb.DMatrix(Xt)

    for fold, (tr, va) in enumerate(folds):
        dtrain = xgb.DMatrix(X[tr], label=y_delta[tr])
        dvalid = xgb.DMatrix(X[va], label=y_delta[va])
        model = xgb.train(
            full_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        oof_delta[va] = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        test_fold_preds.append(model.predict(dtest, iteration_range=(0, model.best_iteration + 1)))
        fold_rmse = rmse(last_known[va][is_lateral[va]] + oof_delta[va][is_lateral[va]], y_abs[va][is_lateral[va]])
        log(f"{tag} fold={fold} best_iter={model.best_iteration} lat_rmse={fold_rmse:.6f}")

    oof_abs = (last_known + oof_delta).astype(np.float32)
    test_abs = (test_last_known + np.mean(np.column_stack(test_fold_preds), axis=1)).astype(np.float32)
    log(f"{tag} lateral_rmse={rmse(oof_abs[is_lateral], y_abs[is_lateral]):.6f}")
    return oof_abs, test_abs


def train_lgb_residual(tag: str, base_oof: np.ndarray, base_test: np.ndarray, params: dict, rounds: int):
    log(f"train residual {tag}")
    residual = (y_abs - base_oof).astype(np.float32)
    oof_resid = np.zeros(len(y_abs), dtype=np.float32)
    test_fold_preds = []
    folds = GroupKFold(5).split(X, residual, groups)

    for fold, (tr, va) in enumerate(folds):
        dtrain = lgb.Dataset(X[tr], label=residual[tr])
        dvalid = lgb.Dataset(X[va], label=residual[va], reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof_resid[va] = model.predict(X[va], num_iteration=model.best_iteration)
        test_fold_preds.append(model.predict(Xt, num_iteration=model.best_iteration))
        fold_score = rmse((base_oof[va] + oof_resid[va])[is_lateral[va]], y_abs[va][is_lateral[va]])
        log(f"{tag} fold={fold} best_iter={model.best_iteration} rmse_w1={fold_score:.6f}")

    test_resid = np.mean(np.column_stack(test_fold_preds), axis=1).astype(np.float32)
    return oof_resid, test_resid


def per_well_savgol(pred: np.ndarray, well_groups: np.ndarray, window: int, poly: int) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for well in np.unique(well_groups):
        sel = well_groups == well
        seg = out[sel]
        win = min(window, len(seg))
        if win % 2 == 0:
            win -= 1
        if win >= poly + 2:
            out[sel] = savgol_filter(seg, win, poly)
    return out


def per_well_median(pred: np.ndarray, well_groups: np.ndarray, kernel: int) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for well in np.unique(well_groups):
        sel = well_groups == well
        seg = out[sel]
        k = min(kernel, len(seg))
        if k % 2 == 0:
            k -= 1
        if k >= 3:
            out[sel] = medfilt(seg, k)
    return out


def load_meta(directory: Path, is_train: bool, use_md: bool = False) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("*__horizontal_well.csv")):
        cols = ["TVT_input"]
        if use_md:
            cols.insert(0, "MD")
        if is_train:
            cols.append("TVT")
        df = pd.read_csv(path, usecols=cols)
        df["well"] = path.stem.replace("__horizontal_well", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


TRAIN_META = load_meta(TRAIN_DIR, is_train=True)
TEST_META = load_meta(TEST_DIR, is_train=False)


def anchor_calibrate(pred: np.ndarray, meta: pd.DataFrame, mode: str, n_tail: int, shrink: float) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    cursor = 0
    for _, df in meta.groupby("well", sort=False):
        n = len(df)
        seg = out[cursor : cursor + n]
        known = df["TVT_input"].notna().to_numpy()
        if known.any():
            residual = df.loc[known, "TVT_input"].to_numpy(np.float32) - seg[known]
            tail = residual[-min(n_tail, len(residual)) :]
            if mode == "median":
                bias = float(np.median(tail))
            elif mode == "mean":
                bias = float(np.mean(tail))
            elif mode == "last":
                bias = float(residual[-1])
            else:
                raise ValueError(mode)
            out[cursor : cursor + n] = seg + shrink * bias
        cursor += n
    return out


# exp063 and exp051 are the selected base models behind exp050/exp070.
exp063_oof, exp063_test = train_xgb_delta(
    "exp063",
    {
        "max_depth": 5,
        "min_child_weight": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.65,
        "reg_lambda": 3.0,
        "learning_rate": 0.015,
        "seed": 101,
    },
    num_boost_round=8000,
    early_stopping_rounds=200,
)

exp051_oof, exp051_test = train_xgb_delta(
    "exp051",
    {
        "max_depth": 6,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.70,
        "reg_lambda": 2.0,
        "learning_rate": 0.03,
        "seed": 7,
    },
    num_boost_round=5000,
    early_stopping_rounds=200,
)


# exp050: hillclimb average selected [exp063, exp051, exp063] plus smoothing cascade.
exp050_oof_raw = ((2.0 * exp063_oof + exp051_oof) / 3.0).astype(np.float32)
exp050_test_raw = ((2.0 * exp063_test + exp051_test) / 3.0).astype(np.float32)
exp050_oof = per_well_savgol(exp050_oof_raw, groups, 201, 2)
exp050_test = per_well_savgol(exp050_test_raw, test_groups, 201, 2)
exp050_oof = per_well_median(exp050_oof, groups, 101)
exp050_test = per_well_median(exp050_test, test_groups, 101)
exp050_oof = per_well_savgol(exp050_oof, groups, 301, 2)
exp050_test = per_well_savgol(exp050_test, test_groups, 301, 2)
log(f"exp050 lateral_rmse={rmse(exp050_oof[is_lateral], y_abs[is_lateral]):.6f}")


exp056_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 2.0,
    "verbose": -1,
    "seed": 0,
    "num_threads": -1,
}
exp056_resid_oof, exp056_resid_test = train_lgb_residual("exp056", exp050_oof, exp050_test, exp056_params, 3000)
exp056_oof = (exp050_oof + exp056_resid_oof).astype(np.float32)
exp056_test = (exp050_test + exp056_resid_test).astype(np.float32)
log(f"exp056 lateral_rmse={rmse(exp056_oof[is_lateral], y_abs[is_lateral]):.6f}")


exp070_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 2.0,
    "lambda_l2": 4.0,
    "verbose": -1,
    "seed": 11,
    "num_threads": -1,
}
exp070_resid_oof, exp070_resid_test = train_lgb_residual("exp070_lgb", exp056_oof, exp056_test, exp070_params, 2000)
exp070_oof = (exp056_oof + 1.5 * exp070_resid_oof).astype(np.float32)
exp070_test = (exp056_test + 1.5 * exp070_resid_test).astype(np.float32)
exp070_oof = per_well_savgol(exp070_oof, groups, 201, 3)
exp070_test = per_well_savgol(exp070_test, test_groups, 201, 3)
log(f"exp070 lateral_rmse={rmse(exp070_oof[is_lateral], y_abs[is_lateral]):.6f}")


exp075_oof = anchor_calibrate(exp070_oof, TRAIN_META, mode="median", n_tail=160, shrink=0.36)
exp075_test = anchor_calibrate(exp070_test, TEST_META, mode="median", n_tail=160, shrink=0.36)
log(f"exp075 lateral_rmse={rmse(exp075_oof[is_lateral], y_abs[is_lateral]):.6f}")


# exp082 observed NNLS intercept winner: almost all exp075 plus a small exp070 term.
exp082_oof = (0.9891444583045232 * exp075_oof + 0.010860811502533917 * exp070_oof).astype(np.float32)
exp082_test = (0.9891444583045232 * exp075_test + 0.010860811502533917 * exp070_test).astype(np.float32)
log(f"exp082 lateral_rmse={rmse(exp082_oof[is_lateral], y_abs[is_lateral]):.6f}")


exp083_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 2.0,
    "lambda_l2": 4.0,
    "verbose": -1,
    "seed": 8301,
    "num_threads": -1,
}
exp083_resid_oof, exp083_resid_test = train_lgb_residual("exp083", exp082_oof, exp082_test, exp083_params, 3000)

exp084_oof = (exp082_oof + 2.6 * exp083_resid_oof).astype(np.float32)
exp084_test = (exp082_test + 2.6 * exp083_resid_test).astype(np.float32)
exp084_oof = per_well_savgol(exp084_oof, groups, 201, 3)
exp084_test = per_well_savgol(exp084_test, test_groups, 201, 3)
log(f"exp084 lateral_rmse={rmse(exp084_oof[is_lateral], y_abs[is_lateral]):.6f}")


late_resid_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.015,
    "num_leaves": 31,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.65,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 4.0,
    "lambda_l2": 8.0,
    "verbose": -1,
    "num_threads": -1,
}

exp085_params = {**late_resid_params, "seed": 8501}
exp085_resid_oof, exp085_resid_test = train_lgb_residual("exp085", exp084_oof, exp084_test, exp085_params, 2500)
exp086_oof = (exp084_oof - 3.12 * exp085_resid_oof).astype(np.float32)
exp086_test = (exp084_test - 3.12 * exp085_resid_test).astype(np.float32)
exp086_oof = per_well_savgol(exp086_oof, groups, 75, 2)
exp086_test = per_well_savgol(exp086_test, test_groups, 75, 2)
log(f"exp086 lateral_rmse={rmse(exp086_oof[is_lateral], y_abs[is_lateral]):.6f}")

exp087_params = {**late_resid_params, "seed": 8701}
exp087_resid_oof, exp087_resid_test = train_lgb_residual("exp087", exp086_oof, exp086_test, exp087_params, 2500)
exp087_oof = (exp086_oof + 0.83 * exp087_resid_oof).astype(np.float32)
exp087_test = (exp086_test + 0.83 * exp087_resid_test).astype(np.float32)
log(f"exp087 lateral_rmse={rmse(exp087_oof[is_lateral], y_abs[is_lateral]):.6f}")

exp089_params = {**late_resid_params, "seed": 8901}
exp089_resid_oof, exp089_resid_test = train_lgb_residual("exp089", exp087_oof, exp087_test, exp089_params, 2500)

final_oof = (exp087_oof - 3.81 * exp089_resid_oof).astype(np.float32)
final_test = (exp087_test - 3.81 * exp089_resid_test).astype(np.float32)
log(f"exp090 local lateral_rmse={rmse(final_oof[is_lateral], y_abs[is_lateral]):.6f}")


def write_submission(pred_all_test_rows: np.ndarray) -> pd.DataFrame:
    rows = []
    cursor = 0
    for path in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
        well = path.stem.replace("__horizontal_well", "")
        df = pd.read_csv(path, usecols=["TVT_input"])
        n = len(df)
        pred = pred_all_test_rows[cursor : cursor + n]
        cursor += n
        mask = df["TVT_input"].isna().to_numpy()
        ids = [f"{well}_{idx}" for idx in df.index[mask]]
        rows.append(pd.DataFrame({"id": ids, "tvt": pred[mask]}))

    assert cursor == len(pred_all_test_rows), (cursor, len(pred_all_test_rows))
    submission = pd.concat(rows, ignore_index=True)

    if SAMPLE_SUBMISSION.exists():
        sample = pd.read_csv(SAMPLE_SUBMISSION)
        submission = sample[["id"]].merge(submission, on="id", how="left")
        missing = int(submission["tvt"].isna().sum())
        assert missing == 0, f"Missing predictions for {missing} sample_submission ids"

    submission.to_csv(OUTPUT_PATH, index=False)
    log(f"wrote {OUTPUT_PATH} rows={len(submission)} tvt_min={submission['tvt'].min():.3f} tvt_max={submission['tvt'].max():.3f}")
    return submission


submission = write_submission(final_test)
submission.head()
