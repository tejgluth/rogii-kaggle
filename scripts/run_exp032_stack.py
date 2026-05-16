"""exp032: NNLS stack of exp026 (LGBM) + exp031 (CatBoost) + savgol smoothing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def load_targets():
    files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    ys, gs, lats = [], [], []
    for p in files:
        df = pd.read_csv(p, usecols=["TVT", "TVT_input"])
        ys.append(df["TVT"].values.astype(np.float32))
        gs.append(np.full(len(df), p.stem.replace("__horizontal_well", "")))
        lats.append(df["TVT_input"].isna().values)
    return np.concatenate(ys), np.concatenate(gs), np.concatenate(lats)


def load_test_groups():
    files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    gs, m = [], []
    for p in files:
        df = pd.read_csv(p, usecols=["TVT_input"])
        gs.append(np.full(len(df), p.stem.replace("__horizontal_well", "")))
        m.append(df["TVT_input"].isna().values)
    return np.concatenate(gs), np.concatenate(m)


def per_well_smooth(preds, groups, window, poly):
    out = preds.copy().astype(np.float32)
    for w in np.unique(groups):
        sel = groups == w
        seg = out[sel]; n = len(seg)
        if n < 5: continue
        win = min(window, n)
        if win % 2 == 0: win -= 1
        if win >= poly + 2:
            out[sel] = savgol_filter(seg, win, poly)
    return out


def main():
    y, groups, lateral = load_targets()
    tg, tmask = load_test_groups()
    members = []
    for tag, e in [("lightgbm","exp025"), ("lightgbm","exp026"), ("catboost","exp031")]:
        p = ROOT / f"experiments/oof/oof_{tag}_{e}.npy"
        if p.exists():
            members.append((e, tag, np.load(p)))
            print(f" {e}: lateral RMSE = {rmse(np.load(p)[lateral], y[lateral]):.4f}")
    keys = [m[0] for m in members]
    X = np.column_stack([m[2] for m in members]).astype(np.float64)
    test_arrays = [np.load(ROOT/f"experiments/test_preds/test_{tag}_{e}.npy") for e,tag,_ in members]
    Xt = np.column_stack(test_arrays).astype(np.float64)

    gkf = GroupKFold(5)
    stack_oof = np.zeros(len(y))
    stack_test_folds = []
    for fi, (tr, va) in enumerate(gkf.split(X, y, groups)):
        Xtr = np.column_stack([X[tr], np.ones(len(tr))])
        coef, _ = nnls(Xtr, y[tr])
        stack_oof[va] = np.column_stack([X[va], np.ones(len(va))]) @ coef
        stack_test_folds.append(np.column_stack([Xt, np.ones(len(Xt))]) @ coef)
        r = rmse(stack_oof[va][lateral[va]], y[va][lateral[va]])
        print(f" fold{fi}: lateral={r:.4f} coef={np.round(coef,3).tolist()}")
    stack_oof = stack_oof.astype(np.float32)
    stack_test = np.mean(np.column_stack(stack_test_folds), axis=1).astype(np.float32)
    raw_lat = rmse(stack_oof[lateral], y[lateral])
    print(f"NNLS stack: {raw_lat:.4f}")

    # Try simple averages too
    for w_cb in [0.5, 0.6, 0.7, 0.8]:
        # last member is exp031 catboost
        avg = X[:, -1] * w_cb + X[:, -2] * (1 - w_cb)
        r = rmse(avg[lateral], y[lateral])
        print(f" avg cb={w_cb} lgbm={1-w_cb:.2f}: lateral={r:.4f}")

    best = (raw_lat, None, None, stack_oof, stack_test)
    for win in [11, 17, 31, 51, 101, 201, 401]:
        for poly in [2, 3]:
            sm = per_well_smooth(stack_oof, groups, win, poly)
            sm_t = per_well_smooth(stack_test, tg, win, poly)
            r = rmse(sm[lateral], y[lateral])
            mark = "*" if r < best[0] else " "
            print(f" {mark} sav win={win} poly={poly}: {r:.4f}")
            if r < best[0]:
                best = (r, win, poly, sm, sm_t)

    best_r, best_win, best_poly, best_oof, best_test = best
    np.save(ROOT/"experiments/oof/oof_stack_exp032.npy", best_oof)
    np.save(ROOT/"experiments/test_preds/test_stack_exp032.npy", best_test)
    json.dump({
        "experiment_id":"exp032","model":"nnls_stack_delta_cb_lgb+savgol",
        "phase":"stacking",
        "cv_rmse_lateral":best_r,
        "members":keys,
        "savgol_window":best_win,"savgol_poly":best_poly,
        "oof_path":"experiments/oof/oof_stack_exp032.npy",
        "test_path":"experiments/test_preds/test_stack_exp032.npy",
        "notes":"NNLS stack of delta-target models incl exp031 CatBoost + per-well savgol.",
    }, open(ROOT/"experiments/results/exp032.json","w"), indent=2)
    print(f"\nexp032: {best_r:.4f} (win={best_win} poly={best_poly})")


if __name__ == "__main__":
    main()
