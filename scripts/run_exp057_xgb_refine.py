"""exp057+: refine around exp051 (depth=6, seed=7, colsample=0.7 → 2.7546).
Test depth=5 and depth=7, lower/higher colsample, multi-seed bag.
"""
import json, time
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

# Anchor on exp051 (depth=6, lr=0.03, mcw=20, subs=0.85, cbt=0.7, lam=2, seed=7).
# Vary one knob at a time.
VARIANTS = [
    ("exp057", dict(max_depth=5, min_child_weight=25, subsample=0.85,
                    colsample_bytree=0.65, reg_lambda=3.0, learning_rate=0.025, seed=101)),
    ("exp058", dict(max_depth=7, min_child_weight=15, subsample=0.85,
                    colsample_bytree=0.7, reg_lambda=2.0, learning_rate=0.03, seed=51)),
    ("exp059", dict(max_depth=6, min_child_weight=20, subsample=0.85,
                    colsample_bytree=0.5, reg_lambda=2.0, learning_rate=0.03, seed=77)),
    ("exp060", dict(max_depth=6, min_child_weight=30, subsample=0.75,
                    colsample_bytree=0.7, reg_lambda=4.0, learning_rate=0.025, seed=131)),
]
common = dict(objective="reg:squarederror", eval_metric="rmse",
              device="cuda", tree_method="hist", max_bin=255, verbosity=0)
gkf = GroupKFold(5)

results = []
for tag, p in VARIANTS:
    t0 = time.time()
    params = {**common, **p}
    print(f"\n=== {tag}  depth={params['max_depth']} cbt={params['colsample_bytree']} seed={params['seed']} ===")
    oof_d = np.zeros(len(y_d), np.float32)
    tp_folds = []
    for fi, (tr, va) in enumerate(gkf.split(X, y_d, grp)):
        dtr = xgb.DMatrix(X[tr], label=y_d[tr])
        dva = xgb.DMatrix(X[va], label=y_d[va])
        dte = xgb.DMatrix(Xt)
        m = xgb.train(params, dtr, num_boost_round=5000,
                      evals=[(dva, "v")], early_stopping_rounds=200, verbose_eval=0)
        oof_d[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        tp_folds.append(m.predict(dte, iteration_range=(0, m.best_iteration + 1)))
        print(f"  fold{fi}: best_iter={m.best_iteration} full={rmse(lk[va]+oof_d[va], y_a[va]):.4f}", flush=True)
    oof_abs = (lk + oof_d).astype(np.float32)
    test_abs = (tlk + np.mean(np.column_stack(tp_folds), axis=1)).astype(np.float32)
    r_lat = rmse(oof_abs[is_lat], y_a[is_lat])
    np.save(ROOT / f"experiments/oof/oof_xgboost_{tag}.npy", oof_abs)
    np.save(ROOT / f"experiments/test_preds/test_xgboost_{tag}.npy", test_abs)
    elapsed = time.time() - t0
    print(f"  {tag} lateral RMSE: {r_lat:.4f}   ({elapsed:.0f}s)", flush=True)
    results.append({"exp": tag, "lat": r_lat, "params": p, "elapsed": elapsed})

json.dump(results, open(ROOT / "experiments/results/exp057_060.json", "w"), indent=2)
print("\n=== summary ===")
for r in results: print(f"  {r['exp']}: lat={r['lat']:.4f}")
