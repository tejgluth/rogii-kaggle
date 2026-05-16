"""exp051+: XGB diversity bag on cached exp043 features.
Trains 4 XGB variants with different (seed, depth, objective) — each emits its own OOF/test
for the phase 4 stacking pool.
"""
import json, time, sys
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

CACHE = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
X = CACHE["X"]; y_d = CACHE["y_delta"]; y_a = CACHE["y_abs"]
lk = CACHE["last_known"]; is_lat = CACHE["is_lateral"]; grp = CACHE["groups"]
Xt = CACHE["Xt"]; tlk = CACHE["test_last_known"]
print(f"cached features: {X.shape}  test: {Xt.shape}")

VARIANTS = [
    ("exp051", dict(objective="reg:squarederror", learning_rate=0.03, max_depth=6,
                    min_child_weight=20, subsample=0.85, colsample_bytree=0.7,
                    reg_lambda=2.0, seed=7)),
    ("exp052", dict(objective="reg:squarederror", learning_rate=0.025, max_depth=10,
                    min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=3.0, seed=11)),
    ("exp053", dict(objective="reg:pseudohubererror", huber_slope=1.0,
                    learning_rate=0.03, max_depth=8, min_child_weight=20,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, seed=23)),
    ("exp054", dict(objective="reg:absoluteerror", learning_rate=0.03, max_depth=7,
                    min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=2.0, seed=31)),
]

common = dict(eval_metric="rmse", device="cuda", tree_method="hist",
              max_bin=255, verbosity=0)

gkf = GroupKFold(5)

results = []
for tag, p in VARIANTS:
    t0 = time.time()
    params = {**common, **p}
    print(f"\n=== {tag}  {params['objective']} depth={params['max_depth']} seed={params['seed']} ===")
    oof_d = np.zeros(len(y_d), np.float32)
    tp_folds = []
    fold_rmses = []
    for fi, (tr, va) in enumerate(gkf.split(X, y_d, grp)):
        dtr = xgb.DMatrix(X[tr], label=y_d[tr])
        dva = xgb.DMatrix(X[va], label=y_d[va])
        dte = xgb.DMatrix(Xt)
        m = xgb.train(params, dtr, num_boost_round=5000,
                      evals=[(dva, "v")], early_stopping_rounds=200, verbose_eval=0)
        oof_d[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        tp_folds.append(m.predict(dte, iteration_range=(0, m.best_iteration + 1)))
        fold_rmses.append(rmse(lk[va] + oof_d[va], y_a[va]))
        print(f"  fold{fi}: best_iter={m.best_iteration} full={fold_rmses[-1]:.4f}", flush=True)
    oof_abs = (lk + oof_d).astype(np.float32)
    test_abs = (tlk + np.mean(np.column_stack(tp_folds), axis=1)).astype(np.float32)
    r_lat = rmse(oof_abs[is_lat], y_a[is_lat])
    np.save(ROOT / f"experiments/oof/oof_xgboost_{tag}.npy", oof_abs)
    np.save(ROOT / f"experiments/test_preds/test_xgboost_{tag}.npy", test_abs)
    elapsed = time.time() - t0
    print(f"  lateral RMSE: {r_lat:.4f}   ({elapsed:.0f}s)")
    results.append({"exp": tag, "lat": r_lat, "params": {k: v for k, v in p.items()}, "elapsed": elapsed})

json.dump(results, open(ROOT / "experiments/results/exp051_054.json", "w"), indent=2)
print("\n=== summary ===")
for r in results: print(f"  {r['exp']}: lat={r['lat']:.4f}")
