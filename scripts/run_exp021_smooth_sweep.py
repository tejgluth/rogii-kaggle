"""exp021: extend smoothing window sweep + try different filters."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"


def load_y_groups():
    files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    ys, gs = [], []
    for p in files:
        df = pd.read_csv(p, usecols=["TVT"])
        ys.append(df["TVT"].values.astype(np.float32))
        gs.append(np.full(len(df), p.stem.replace("__horizontal_well", "")))
    return np.concatenate(ys), np.concatenate(gs)


def load_test_groups():
    files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    gs = []
    for p in files:
        n = sum(1 for _ in open(p)) - 1
        gs.append(np.full(n, p.stem.replace("__horizontal_well", "")))
    return np.concatenate(gs)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def per_well(preds, groups, fn):
    out = preds.copy().astype(np.float64)
    for w in np.unique(groups):
        mask = groups == w
        seg = out[mask]
        if len(seg) < 5:
            continue
        out[mask] = fn(seg, len(seg))
    return out.astype(np.float32)


def make_savgol(window, poly):
    def f(seg, n):
        win = min(window, n - (1 - n % 2))
        if win % 2 == 0:
            win -= 1
        if win < poly + 1:
            return seg
        return savgol_filter(seg, window_length=win, polyorder=poly)
    return f


def make_gauss(sigma):
    def f(seg, n):
        return gaussian_filter1d(seg, sigma=sigma, mode="nearest")
    return f


def make_uniform(window):
    def f(seg, n):
        return uniform_filter1d(seg, size=min(window, n), mode="nearest")
    return f


def main():
    y, groups = load_y_groups()
    tgroups = load_test_groups()
    base = np.load(ROOT / "experiments/oof/oof_catboost_exp014.npy")
    base_test = np.load(ROOT / "experiments/test_preds/test_catboost_exp014.npy")
    stack = np.load(ROOT / "experiments/oof/oof_stack_exp020.npy")  # NNLS already smoothed
    # also load raw stack (re-derive)
    print(f"exp014: {rmse(base, y):.4f}")

    # Savgol sweep on exp014
    cfgs = []
    for win in [201, 251, 301, 401, 501, 701, 1001, 1501, 2001]:
        for poly in [2, 3, 4]:
            sm = per_well(base, groups, make_savgol(win, poly))
            r = rmse(sm, y)
            cfgs.append(("savgol", win, poly, r))
            print(f"savgol win={win} poly={poly}: {r:.4f}")
    for sigma in [10, 30, 60, 100, 200, 400, 800]:
        sm = per_well(base, groups, make_gauss(sigma))
        r = rmse(sm, y)
        cfgs.append(("gauss", sigma, None, r))
        print(f"gauss sigma={sigma}: {r:.4f}")
    for win in [50, 100, 200, 400, 800, 1600]:
        sm = per_well(base, groups, make_uniform(win))
        r = rmse(sm, y)
        cfgs.append(("uniform", win, None, r))
        print(f"uniform win={win}: {r:.4f}")

    cfgs.sort(key=lambda x: x[3])
    print("\nTop 5:", cfgs[:5])
    name, p1, p2, best = cfgs[0]
    print(f"\nBest: {name} p1={p1} p2={p2} -> {best:.4f}")

    # Build best smoother
    if name == "savgol":
        smooth_fn = make_savgol(p1, p2)
    elif name == "gauss":
        smooth_fn = make_gauss(p1)
    else:
        smooth_fn = make_uniform(p1)

    sm_oof = per_well(base, groups, smooth_fn)
    sm_test = per_well(base_test, tgroups, smooth_fn)
    np.save(ROOT / "experiments/oof/oof_catboost_exp021.npy", sm_oof)
    np.save(ROOT / "experiments/test_preds/test_catboost_exp021.npy", sm_test)
    json.dump({
        "experiment_id": "exp021",
        "model": f"exp014+{name}",
        "phase": "postprocessing",
        "base_experiment": "exp014",
        "cv_rmse": best,
        "filter": name, "p1": p1, "p2": p2,
        "all_results": [(c[0], c[1], c[2], c[3]) for c in cfgs[:20]],
        "notes": "Extended smoothing window sweep + Gaussian/uniform alternatives.",
        "oof_path": "experiments/oof/oof_catboost_exp021.npy",
        "test_path": "experiments/test_preds/test_catboost_exp021.npy",
    }, open(ROOT / "experiments/results/exp021.json", "w"), indent=2)
    print(f"\nexp021: {best:.4f}")

    # exp022: best smoothing applied to NNLS stack (re-derive stack to also smooth)
    # The exp020 NNLS stack was already smoothed with win=201. Try the new best smoother on raw NNLS.
    # Re-build raw NNLS stack:
    from sklearn.model_selection import GroupKFold
    from scipy.optimize import nnls
    members = {"exp014":"catboost","exp013":"lightgbm","exp012":"xgboost",
               "exp009":"lightgbm","exp011":"lightgbm","exp006":"lightgbm"}
    oofs = {e: np.load(ROOT/f"experiments/oof/oof_{m}_{e}.npy") for e,m in members.items()}
    tests= {e: np.load(ROOT/f"experiments/test_preds/test_{m}_{e}.npy") for e,m in members.items()}
    keys = list(members.keys())
    X = np.column_stack([oofs[k] for k in keys]).astype(np.float64)
    Xt= np.column_stack([tests[k] for k in keys]).astype(np.float64)
    gkf = GroupKFold(5)
    nnls_oof = np.zeros(len(y))
    nnls_test_folds = []
    for tr, va in gkf.split(X, y, groups):
        Xtr = np.column_stack([X[tr], np.ones(len(tr))])
        coef,_ = nnls(Xtr, y[tr])
        Xva = np.column_stack([X[va], np.ones(len(va))])
        nnls_oof[va] = Xva @ coef
        Xte = np.column_stack([Xt, np.ones(len(Xt))])
        nnls_test_folds.append(Xte @ coef)
    nnls_test = np.mean(np.column_stack(nnls_test_folds), axis=1)
    nnls_oof = nnls_oof.astype(np.float32); nnls_test = nnls_test.astype(np.float32)
    print(f"raw NNLS: {rmse(nnls_oof, y):.4f}")
    sm_stack = per_well(nnls_oof, groups, smooth_fn)
    sm_stack_test = per_well(nnls_test, tgroups, smooth_fn)
    r = rmse(sm_stack, y)
    print(f"NNLS + best smoother: {r:.4f}")
    np.save(ROOT/"experiments/oof/oof_stack_exp022.npy", sm_stack)
    np.save(ROOT/"experiments/test_preds/test_stack_exp022.npy", sm_stack_test)
    json.dump({
        "experiment_id": "exp022",
        "model": f"nnls_stack+{name}",
        "phase": "stacking+postprocessing",
        "cv_rmse": r,
        "filter": name, "p1": p1, "p2": p2,
        "members": keys,
        "notes": "NNLS stack + best smoothing filter from exp021 sweep.",
        "oof_path": "experiments/oof/oof_stack_exp022.npy",
        "test_path": "experiments/test_preds/test_stack_exp022.npy",
    }, open(ROOT/"experiments/results/exp022.json","w"), indent=2)
    print(f"exp022: {r:.4f}")

if __name__ == "__main__":
    main()
