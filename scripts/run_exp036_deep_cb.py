"""exp036: Deeper CatBoost on exp026 features. depth=10, lr=0.03, iter=12000."""
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor, Pool

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
        tw = p.parent/(p.stem.replace("__horizontal_well","__typewell")+".csv")
        d = build_well(p, tw, True)
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
    print(f"features:{len(feat)}", flush=True)

    gkf = GroupKFold(5)
    oof_d = np.zeros(len(y_d), np.float32)
    tp_folds = []
    fold_rmses = []
    for fi,(tr,va) in enumerate(gkf.split(X, y_d, grp)):
        m = CatBoostRegressor(
            iterations=12000, learning_rate=0.03, depth=10, l2_leaf_reg=3.0,
            random_seed=42, od_type="Iter", od_wait=400, eval_metric="RMSE",
            loss_function="RMSE", task_type="GPU",
            allow_writing_files=False, verbose=400,
        )
        m.fit(Pool(X[tr], label=y_d[tr]), eval_set=Pool(X[va], label=y_d[va]),
              use_best_model=True)
        oof_d[va] = m.predict(X[va])
        tp_folds.append(m.predict(Xt))
        r = rmse(lk[va]+oof_d[va], y_a[va])
        fold_rmses.append(r)
        print(f" fold{fi}: best_iter={m.get_best_iteration()} full={r:.4f}", flush=True)

    oof_abs = (lk + oof_d).astype(np.float32)
    test_abs = (tlk + np.mean(np.column_stack(tp_folds), axis=1)).astype(np.float32)
    r_lat = rmse(oof_abs[is_lat], y_a[is_lat])
    print(f"\nlateral RMSE: {r_lat:.4f}")
    np.save(ROOT/"experiments/oof/oof_catboost_exp036.npy", oof_abs)
    np.save(ROOT/"experiments/test_preds/test_catboost_exp036.npy", test_abs)
    json.dump({
        "experiment_id":"exp036","model":"catboost_deep_d10_lr03",
        "phase":"feature_engineering",
        "cv_rmse_lateral":r_lat,
        "fold_rmses_full":fold_rmses,
        "notes":"Deeper CatBoost depth=10 lr=0.03 iter=12000 on exp026 features.",
        "oof_path":"experiments/oof/oof_catboost_exp036.npy",
        "test_path":"experiments/test_preds/test_catboost_exp036.npy",
    }, open(ROOT/"experiments/results/exp036.json","w"), indent=2)

if __name__ == "__main__":
    main()
