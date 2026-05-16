"""exp031: CatBoost on the exp026 feature set (delta target).

Replicates the strategy that won big in exp014 (CatBoost on exp009 LGBM features
gave -4 RMSE). Now applied to delta-target features which already give 4.16 RMSE.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor, Pool
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
from run_exp026_bwell_ncc import build_well, FORMATIONS, multi_scale_ncc_offsets  # reuse
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    t0 = time.time()
    print("Loading train...", flush=True)
    files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    dfs = []
    for i, p in enumerate(files):
        tw = p.parent / (p.stem.replace("__horizontal_well","__typewell")+".csv")
        d = build_well(p, tw, is_train=True)
        if d is not None: dfs.append(d)
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    train = pd.concat(dfs, ignore_index=True)
    print(f"train: {train.shape}")

    tfiles = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    tdfs = [build_well(p, p.parent/(p.stem.replace("__horizontal_well","__typewell")+".csv"), False) for p in tfiles]
    test = pd.concat([t for t in tdfs if t is not None], ignore_index=True)
    print(f"test: {test.shape}")

    skip = {"well_id", "TVT", "delta_target", "is_lateral"}
    feat_cols = [c for c in train.columns if c not in skip]
    print(f"#features: {len(feat_cols)}")
    X = train[feat_cols].values.astype(np.float32)
    y_delta = train["delta_target"].values.astype(np.float32)
    y_abs = train["TVT"].values.astype(np.float32)
    last_known = train["last_known_tvt"].values.astype(np.float32)
    is_lateral = train["is_lateral"].values
    groups = train["well_id"].values
    Xt = test[feat_cols].values.astype(np.float32)
    test_last_known = test["last_known_tvt"].values.astype(np.float32)

    params = dict(
        iterations=8000, learning_rate=0.05, depth=8, l2_leaf_reg=5.0,
        random_seed=42, od_type="Iter", od_wait=300, eval_metric="RMSE",
        loss_function="RMSE", task_type="GPU", allow_writing_files=False,
        verbose=400,
    )

    gkf = GroupKFold(5)
    oof_delta = np.zeros(len(y_delta), dtype=np.float32)
    tp_folds = []
    fold_rmses = []
    for fi, (tr, va) in enumerate(gkf.split(X, y_delta, groups)):
        m = CatBoostRegressor(**params)
        m.fit(Pool(X[tr], label=y_delta[tr]),
              eval_set=Pool(X[va], label=y_delta[va]),
              use_best_model=True)
        oof_delta[va] = m.predict(X[va])
        tp_folds.append(m.predict(Xt))
        abs_pred = last_known[va] + oof_delta[va]
        r = rmse(abs_pred[is_lateral[va]], y_abs[va][is_lateral[va]])
        fold_rmses.append(r)
        print(f" fold{fi}: best_iter={m.get_best_iteration()} lateral={r:.4f}", flush=True)

    oof_abs = last_known + oof_delta
    r_lat = rmse(oof_abs[is_lateral], y_abs[is_lateral])
    r_full = rmse(oof_abs, y_abs)
    test_abs = test_last_known + np.mean(np.column_stack(tp_folds), axis=1)
    np.save(ROOT/"experiments/oof/oof_catboost_exp031.npy", oof_abs.astype(np.float32))
    np.save(ROOT/"experiments/test_preds/test_catboost_exp031.npy", test_abs.astype(np.float32))
    json.dump({
        "experiment_id":"exp031","model":"catboost_delta_bwell_ncc",
        "phase":"feature_engineering",
        "cv_rmse":r_full,"cv_rmse_lateral":r_lat,
        "fold_rmses_lateral":fold_rmses,
        "n_features":len(feat_cols),
        "notes":"CatBoost (GPU) on exp026 feature set with delta target.",
        "oof_path":"experiments/oof/oof_catboost_exp031.npy",
        "test_path":"experiments/test_preds/test_catboost_exp031.npy",
    }, open(ROOT/"experiments/results/exp031.json","w"), indent=2)
    print(f"\nexp031: lateral={r_lat:.4f} full={r_full:.4f}")


if __name__ == "__main__":
    main()
