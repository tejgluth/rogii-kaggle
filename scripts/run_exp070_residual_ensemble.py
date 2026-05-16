"""exp070: residual ensemble — fit multiple model families on the same residual,
combine via NNLS. Each member captures different patterns in the bias.

Residual = y_true - exp056_stack (which already includes the LGBM residual).
We probe further: do XGB and CatBoost see additional structure?
"""
import json, time
from pathlib import Path
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent.parent
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

CACHE = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
X = CACHE["X"]; y_a = CACHE["y_abs"]; lk = CACHE["last_known"]
is_lat = CACHE["is_lateral"]; grp = np.asarray(CACHE["groups"])
Xt = CACHE["Xt"]; tlk = CACHE["test_last_known"]

oof_stack = np.load(ROOT / "experiments/oof/oof_stack_exp056.npy")
test_stack = np.load(ROOT / "experiments/test_preds/test_stack_exp056.npy")
residual = (y_a - oof_stack).astype(np.float32)
print(f"residual lat-RMSE (baseline = current stack RMSE): {rmse(residual[is_lat], 0*residual[is_lat]):.4f}")

gkf = GroupKFold(5)


def fit_oof(model_kind, params, nrounds=2000):
    oof = np.zeros(len(residual), np.float32)
    tps = []
    for fi, (tr, va) in enumerate(gkf.split(X, residual, grp)):
        if model_kind == "lgb":
            dtr = lgb.Dataset(X[tr], label=residual[tr])
            dva = lgb.Dataset(X[va], label=residual[va], reference=dtr)
            m = lgb.train(params, dtr, num_boost_round=nrounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
            tps.append(m.predict(Xt, num_iteration=m.best_iteration))
            print(f"  lgb fold{fi}: best={m.best_iteration}", flush=True)
        else:
            dtr = xgb.DMatrix(X[tr], label=residual[tr])
            dva = xgb.DMatrix(X[va], label=residual[va])
            dte = xgb.DMatrix(Xt)
            m = xgb.train(params, dtr, num_boost_round=nrounds,
                          evals=[(dva, "v")], early_stopping_rounds=150, verbose_eval=0)
            oof[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
            tps.append(m.predict(dte, iteration_range=(0, m.best_iteration + 1)))
            print(f"  xgb fold{fi}: best={m.best_iteration}", flush=True)
    return oof, np.mean(np.column_stack(tps), axis=1).astype(np.float32)


lgb_params = dict(objective="regression", metric="rmse", learning_rate=0.02,
                  num_leaves=31, min_data_in_leaf=300, feature_fraction=0.7,
                  bagging_fraction=0.85, bagging_freq=1,
                  lambda_l1=2.0, lambda_l2=4.0, verbose=-1, seed=11)
print("\n=== LGBM residual (more reg) ===")
oof_lgb, test_lgb = fit_oof("lgb", lgb_params)
print(f"  added lat: {rmse((oof_stack+oof_lgb)[is_lat], y_a[is_lat]):.4f}", flush=True)

xgb_params = dict(objective="reg:squarederror", eval_metric="rmse",
                  learning_rate=0.02, max_depth=5, min_child_weight=30,
                  subsample=0.85, colsample_bytree=0.65, reg_lambda=4.0,
                  device="cuda", tree_method="hist", max_bin=255, verbosity=0, seed=7)
print("\n=== XGB residual (depth=5, slow) ===")
oof_xgb, test_xgb = fit_oof("xgb", xgb_params)
print(f"  added lat: {rmse((oof_stack+oof_xgb)[is_lat], y_a[is_lat]):.4f}", flush=True)

# Now combine: stack_oof = oof_stack + w_lgb * oof_lgb + w_xgb * oof_xgb
# We can also include the existing exp056_resid as a third.
oof_resid56 = np.load(ROOT / "experiments/oof/oof_lgbm_exp056_resid.npy")  # already added; we omit it
# Actually exp056 oof is already inside oof_stack; treat oof_lgb and oof_xgb as supplements.

best = (rmse(oof_stack[is_lat], y_a[is_lat]), None)
for w1 in np.arange(0, 1.51, 0.05):
    for w2 in np.arange(0, 1.51, 0.05):
        cand = oof_stack + w1 * oof_lgb + w2 * oof_xgb
        r = rmse(cand[is_lat], y_a[is_lat])
        if r < best[0]:
            best = (r, (w1, w2))
print(f"\nbest residual combo: w_lgb={best[1][0]:.2f} w_xgb={best[1][1]:.2f} → lat={best[0]:.4f}")

final_oof = oof_stack + best[1][0] * oof_lgb + best[1][1] * oof_xgb
final_test = test_stack + best[1][0] * test_lgb + best[1][1] * test_xgb
final_oof = final_oof.astype(np.float32); final_test = final_test.astype(np.float32)

# Try savgol cascade on top
from scipy.signal import savgol_filter, medfilt
import pandas as pd
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"
def get_groups(d):
    gs = []
    for p in sorted(d.glob("*__horizontal_well.csv")):
        n = len(pd.read_csv(p, usecols=["MD"]))
        gs.append(np.full(n, p.stem.replace("__horizontal_well", "")))
    return np.concatenate(gs)
gtr = get_groups(TRAIN_DIR); gte = get_groups(TEST_DIR)
def per_well_smooth(preds, groups, fn):
    out = preds.copy().astype(np.float32)
    for w in np.unique(groups):
        sel = groups == w; seg = out[sel]
        if len(seg) > 5:
            out[sel] = fn(seg)
    return out

base = (best[0], final_oof, final_test, "raw")
for win in [101, 201, 301, 501]:
    for poly in [2, 3]:
        s = per_well_smooth(final_oof, gtr, lambda x, w=win, p=poly: savgol_filter(x, min(w, len(x))|1 if min(w,len(x))%2==0 else min(w,len(x)), p) if len(x) >= p+2 else x)
        st = per_well_smooth(final_test, gte, lambda x, w=win, p=poly: savgol_filter(x, min(w, len(x))|1 if min(w,len(x))%2==0 else min(w,len(x)), p) if len(x) >= p+2 else x)
        r = rmse(s[is_lat], y_a[is_lat])
        if r < base[0]:
            base = (r, s, st, f"savgol_w{win}_p{poly}")
print(f"\nbest with smoothing: {base[3]} lat={base[0]:.4f}")

np.save(ROOT / "experiments/oof/oof_stack_exp070.npy", base[1])
np.save(ROOT / "experiments/test_preds/test_stack_exp070.npy", base[2])
np.save(ROOT / "experiments/oof/oof_lgbm_exp070_resid.npy", oof_lgb)
np.save(ROOT / "experiments/test_preds/test_lgbm_exp070_resid.npy", test_lgb)
np.save(ROOT / "experiments/oof/oof_xgboost_exp070_resid.npy", oof_xgb)
np.save(ROOT / "experiments/test_preds/test_xgboost_exp070_resid.npy", test_xgb)
json.dump({"experiment_id":"exp070","phase":"stacking",
           "best_lat_rmse": base[0], "best_method": base[3],
           "w_lgb": best[1][0], "w_xgb": best[1][1]},
          open(ROOT / "experiments/results/exp070.json", "w"), indent=2)
print("\n=== done ===")
