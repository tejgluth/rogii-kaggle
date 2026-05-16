"""exp023: LightGBM meta-learner stack. exp024: per-well median+savgol cascade."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
from sklearn.model_selection import GroupKFold

import lightgbm as lgb

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


def per_well_apply(preds, groups, fn):
    out = preds.copy().astype(np.float64)
    for w in np.unique(groups):
        mask = groups == w
        seg = out[mask]
        if len(seg) < 5:
            continue
        out[mask] = fn(seg)
    return out.astype(np.float32)


def savgol_fn(window, poly):
    def f(seg):
        n = len(seg)
        win = min(window, n - (1 - n % 2))
        if win % 2 == 0:
            win -= 1
        if win < poly + 1:
            return seg
        return savgol_filter(seg, win, poly)
    return f


def med_then_savgol(med_w, sg_w, sg_p):
    def f(seg):
        n = len(seg)
        mw = min(med_w, n - (1 - n % 2))
        if mw % 2 == 0:
            mw -= 1
        if mw >= 3:
            seg = medfilt(seg, mw)
        sw = min(sg_w, n - (1 - n % 2))
        if sw % 2 == 0:
            sw -= 1
        if sw >= sg_p + 1:
            seg = savgol_filter(seg, sw, sg_p)
        return seg
    return f


def main():
    y, groups = load_y_groups()
    tgroups = load_test_groups()

    members = {"exp014":"catboost","exp013":"lightgbm","exp012":"xgboost",
               "exp009":"lightgbm","exp011":"lightgbm","exp006":"lightgbm",
               "exp003":"lightgbm","exp010":"xgboost"}
    oofs = {e: np.load(ROOT/f"experiments/oof/oof_{m}_{e}.npy") for e,m in members.items()}
    tests= {e: np.load(ROOT/f"experiments/test_preds/test_{m}_{e}.npy") for e,m in members.items()}
    keys = list(members.keys())
    X = np.column_stack([oofs[k] for k in keys]).astype(np.float32)
    Xt = np.column_stack([tests[k] for k in keys]).astype(np.float32)
    print(f"meta features: {len(keys)}")

    # ===== exp023: LightGBM meta-learner =====
    print("\n=== exp023: LGBM meta-learner ===")
    gkf = GroupKFold(5)
    meta_oof = np.zeros(len(y), dtype=np.float32)
    test_folds = []
    params = dict(objective="regression", metric="rmse",
                  learning_rate=0.02, num_leaves=15, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=-1)
    fold_rmses = []
    for fi, (tr, va) in enumerate(gkf.split(X, y, groups)):
        dtr = lgb.Dataset(X[tr], y[tr])
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        model = lgb.train(params, dtr, num_boost_round=3000,
                          valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        meta_oof[va] = model.predict(X[va], num_iteration=model.best_iteration)
        test_folds.append(model.predict(Xt, num_iteration=model.best_iteration))
        fr = rmse(meta_oof[va], y[va])
        fold_rmses.append(fr)
        print(f" fold{fi}: RMSE={fr:.4f} best_iter={model.best_iteration}")
    meta_test = np.mean(np.column_stack(test_folds), axis=1).astype(np.float32)
    raw_rmse = rmse(meta_oof, y)
    print(f"LGBM meta raw OOF RMSE: {raw_rmse:.4f}")

    # smooth
    sm = per_well_apply(meta_oof, groups, savgol_fn(1001, 3))
    sm_test = per_well_apply(meta_test, tgroups, savgol_fn(1001, 3))
    sm_rmse = rmse(sm, y)
    print(f"LGBM meta + savgol(1001,3): {sm_rmse:.4f}")
    np.save(ROOT/"experiments/oof/oof_stack_exp023.npy", sm)
    np.save(ROOT/"experiments/test_preds/test_stack_exp023.npy", sm_test)
    json.dump({
        "experiment_id":"exp023","model":"lgbm_meta+savgol","phase":"stacking",
        "cv_rmse":sm_rmse,"raw_rmse":raw_rmse,"fold_rmses":fold_rmses,
        "members":keys,"lgbm_params":params,
        "notes":"LightGBM meta-learner on 8 OOFs + savgol(1001,3) per-well smoothing.",
        "oof_path":"experiments/oof/oof_stack_exp023.npy",
        "test_path":"experiments/test_preds/test_stack_exp023.npy",
    }, open(ROOT/"experiments/results/exp023.json","w"), indent=2)

    # ===== exp024: median+savgol cascade on exp022 stack input =====
    print("\n=== exp024: median+savgol cascade on best stack ===")
    # rebuild NNLS stack
    from scipy.optimize import nnls
    keys6 = ["exp014","exp013","exp012","exp009","exp011","exp006"]
    X6 = np.column_stack([oofs[k] for k in keys6]).astype(np.float64)
    X6t= np.column_stack([tests[k] for k in keys6]).astype(np.float64)
    gkf2 = GroupKFold(5)
    nnls_oof = np.zeros(len(y))
    nnls_test_folds=[]
    for tr, va in gkf2.split(X6, y, groups):
        Xtr = np.column_stack([X6[tr], np.ones(len(tr))])
        coef,_ = nnls(Xtr, y[tr])
        nnls_oof[va] = np.column_stack([X6[va], np.ones(len(va))]) @ coef
        nnls_test_folds.append(np.column_stack([X6t, np.ones(len(X6t))]) @ coef)
    nnls_test = np.mean(np.column_stack(nnls_test_folds), axis=1).astype(np.float32)
    nnls_oof = nnls_oof.astype(np.float32)

    best = (1e9, None)
    for med_w in [0, 51, 101, 201, 401]:
        for sg_w in [501, 701, 1001, 1501]:
            fn = med_then_savgol(med_w, sg_w, 3) if med_w else savgol_fn(sg_w, 3)
            sm = per_well_apply(nnls_oof, groups, fn)
            r = rmse(sm, y)
            mark = "*" if r < best[0] else " "
            print(f" {mark}med={med_w} savgol={sg_w}: {r:.4f}")
            if r < best[0]:
                best = (r, (med_w, sg_w))

    print(f"Best: {best}")
    med_w, sg_w = best[1]
    fn = med_then_savgol(med_w, sg_w, 3) if med_w else savgol_fn(sg_w, 3)
    final_oof = per_well_apply(nnls_oof, groups, fn)
    final_test = per_well_apply(nnls_test, tgroups, fn)
    np.save(ROOT/"experiments/oof/oof_stack_exp024.npy", final_oof)
    np.save(ROOT/"experiments/test_preds/test_stack_exp024.npy", final_test)
    json.dump({
        "experiment_id":"exp024","model":f"nnls+med{med_w}+savgol{sg_w}","phase":"stacking+postprocess",
        "cv_rmse":best[0],"med_window":med_w,"savgol_window":sg_w,
        "members":keys6,
        "notes":"NNLS stack with median+savgol cascade smoothing.",
        "oof_path":"experiments/oof/oof_stack_exp024.npy",
        "test_path":"experiments/test_preds/test_stack_exp024.npy",
    }, open(ROOT/"experiments/results/exp024.json","w"), indent=2)
    print(f"exp024: {best[0]:.4f}")

if __name__ == "__main__":
    main()
