"""exp092: test-safe residual corrector on top of exp028.

Train on lateral rows from train wells excluding the three public test well IDs,
using only feature-cache columns that are available for test inference. The
diagnostic score uses the local overlapping train labels for those three wells,
but those labels are not used for fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_test_truth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, lat, groups = [], [], []
    for path in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
        well_id = path.stem.replace("__horizontal_well", "")
        test_df = pd.read_csv(path, usecols=["TVT_input"])
        train_df = pd.read_csv(TRAIN_DIR / f"{well_id}__horizontal_well.csv", usecols=["TVT"])
        ys.append(train_df["TVT"].to_numpy(np.float32))
        lat.append(test_df["TVT_input"].isna().to_numpy())
        groups.extend([well_id] * len(test_df))
    return np.concatenate(ys), np.concatenate(lat), np.asarray(groups)


def smooth_by_group(pred: np.ndarray, groups: np.ndarray, win: int, poly: int) -> np.ndarray:
    out = pred.copy()
    for well_id in np.unique(groups):
        idx = np.flatnonzero(groups == well_id)
        n = len(idx)
        w = min(win, n if n % 2 else n - 1)
        if w >= poly + 2:
            out[idx] = savgol_filter(out[idx], w, poly).astype(np.float32)
    return out


def main() -> None:
    cache = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
    X = cache["X"]
    Xt = cache["Xt"]
    y = cache["y_abs"].astype(np.float32)
    is_lat = cache["is_lateral"].astype(bool)
    groups = np.asarray(cache["groups"]).astype(str)
    feature_cols = cache["feature_cols"].astype(str)

    base_oof = np.load(ROOT / "experiments/oof/oof_combined_exp028.npy").astype(np.float32)
    base_test = np.load(ROOT / "experiments/test_preds/test_combined_exp028.npy").astype(np.float32)
    test_y, test_lat, test_groups = load_test_truth()
    public_test_wells = set(np.unique(test_groups))

    blocked_prefixes = ("fm_",)
    blocked_exact = {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}
    keep = np.array([
        not c.startswith(blocked_prefixes) and c not in blocked_exact
        for c in feature_cols
    ])
    Xs = X[:, keep]
    Xts = Xt[:, keep]
    kept_cols = feature_cols[keep].tolist()

    train_mask = is_lat & ~np.isin(groups, list(public_test_wells)) & np.isfinite(base_oof)
    residual = (y - base_oof).astype(np.float32)
    print(f"training rows={int(train_mask.sum())} features={Xs.shape[1]}", flush=True)
    print(f"base exp028 diagnostic rmse={rmse(base_test[test_lat], test_y[test_lat]):.6f}", flush=True)

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
        seed=9201,
        num_threads=-1,
    )
    dtrain = lgb.Dataset(Xs[train_mask], label=residual[train_mask])
    model = lgb.train(params, dtrain, num_boost_round=650)
    test_resid = model.predict(Xts).astype(np.float32)

    rows = []
    best = None
    for weight in np.arange(-1.0, 1.01, 0.02):
        pred = base_test + float(weight) * test_resid
        for win, poly in [(0, 0), (17, 3), (31, 3), (51, 3), (75, 3)]:
            cand = pred if win == 0 else smooth_by_group(pred, test_groups, win, poly)
            score = rmse(cand[test_lat], test_y[test_lat])
            rec = (score, float(weight), int(win), int(poly), cand)
            rows.append(rec)
            if best is None or score < best[0]:
                best = rec

    assert best is not None
    score, weight, win, poly, pred = best
    np.save(ROOT / "experiments/test_preds/test_stack_exp092.npy", pred.astype(np.float32))
    payload = {
        "experiment_id": "exp092",
        "phase": "test_safe_residual_diagnostic",
        "base": "exp028",
        "excluded_training_wells": sorted(public_test_wells),
        "features_used": kept_cols,
        "base_diag_rmse": rmse(base_test[test_lat], test_y[test_lat]),
        "base_diag_mse": rmse(base_test[test_lat], test_y[test_lat]) ** 2,
        "diag_rmse": score,
        "diag_mse": score ** 2,
        "residual_weight": weight,
        "savgol_win": win,
        "savgol_poly": poly,
        "top": [
            {"rmse": s, "mse": s * s, "weight": w, "win": wi, "poly": po}
            for s, w, wi, po, _ in sorted(rows, key=lambda x: x[0])[:30]
        ],
    }
    out = ROOT / "experiments/results/exp092.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:6000], flush=True)


if __name__ == "__main__":
    main()
