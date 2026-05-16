"""exp040: 3-seed XGBoost bag on exp026 features."""
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
from run_exp026_bwell_ncc import build_well
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def main():
    t0 = time.time()
    print("Loading...", flush=True)
    files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    dfs = []
    for i,p in enumerate(files):
        d = build_well(p, p.parent/(p.stem.replace("__horizontal_well","__typewell")+".csv"), True)
        if d is not None: dfs.append(d)
        if (i+1)%300==0: print(f" {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    train = pd.concat(dfs, ignore_index=True)
    tfiles = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    test = pd.concat([build_well(p, p.parent/(p.stem.replace("__horizontal_well","__typewell")+".csv"), False) for p in tfiles], ignore_index=True)

    skip = {"well_id","TVT","delta_target","is_lateral"}
    feat = [c for c in train.columns if c not in skip]
    X = train[feat].values.astype(np.float32)
    y_d = train["delta_target"].values.astype(np.float32)
    y_a = train["TVT"].values.astype(np.float32)
    lk = train["last_known_tvt"].values.astype(np.float32)
    is_lat = train["is_lateral"].values
    grp = train["well_id"].values
    Xt = test[feat].values.astype(np.float32)
    tlk = test["last_known_tvt"].values.astype(np.float32)

    base = dict(
        objective="reg:squarederror", eval_metric="rmse",
        learning_rate=0.03, max_depth=8, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        device="cuda", tree_method="hist", max_bin=255, verbosity=0,
    )
    seeds = [42, 1337, 2024]
    all_oof = []; all_test = []
    for seed in seeds:
        params = dict(base, seed=seed)
        gkf = GroupKFold(5)
        oof_d = np.zeros(len(y_d), np.float32)
        tp_folds = []
        for fi,(tr,va) in enumerate(gkf.split(X, y_d, grp)):
            dtr = xgb.DMatrix(X[tr], label=y_d[tr])
            dva = xgb.DMatrix(X[va], label=y_d[va])
            dte = xgb.DMatrix(Xt)
            m = xgb.train(params, dtr, num_boost_round=5000,
                          evals=[(dva,"v")], early_stopping_rounds=200, verbose_eval=0)
            oof_d[va] = m.predict(dva, iteration_range=(0, m.best_iteration+1))
            tp_folds.append(m.predict(dte, iteration_range=(0, m.best_iteration+1)))
            r = rmse(lk[va]+oof_d[va], y_a[va])
            print(f" seed{seed} fold{fi}: iter={m.best_iteration} full={r:.4f}", flush=True)
        all_oof.append(oof_d)
        all_test.append(np.mean(np.column_stack(tp_folds), axis=1))

    oof_avg = np.mean(np.column_stack(all_oof), axis=1)
    test_avg = np.mean(np.column_stack(all_test), axis=1)
    oof_abs = (lk + oof_avg).astype(np.float32)
    test_abs = (tlk + test_avg).astype(np.float32)
    r_lat = rmse(oof_abs[is_lat], y_a[is_lat])
    print(f"\nseedbag lateral RMSE: {r_lat:.4f}")
    np.save(ROOT/"experiments/oof/oof_xgboost_exp040.npy", oof_abs)
    np.save(ROOT/"experiments/test_preds/test_xgboost_exp040.npy", test_abs)
    json.dump({
        "experiment_id":"exp040","model":"xgboost_seedbag3",
        "phase":"feature_engineering","cv_rmse_lateral":r_lat,"seeds":seeds,
        "notes":"3-seed XGBoost bag on exp026 features.",
        "oof_path":"experiments/oof/oof_xgboost_exp040.npy",
        "test_path":"experiments/test_preds/test_xgboost_exp040.npy",
    }, open(ROOT/"experiments/results/exp040.json","w"), indent=2)
    print(f"exp040: {r_lat:.4f}")

if __name__ == "__main__":
    main()
