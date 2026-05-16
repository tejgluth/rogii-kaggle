"""exp066+: push the slow-lr regime further.

exp063 (depth=5, lr=0.015, 8000 rounds, seed=101) → 2.6928. fold2 best_iter=7992
(hit max), suggesting more iterations could help.
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

RUNS = [
    ("exp066", dict(max_depth=5, min_child_weight=25, subsample=0.85,
                    colsample_bytree=0.65, reg_lambda=3.0, learning_rate=0.012),
     15000, [101]),
    ("exp067", dict(max_depth=5, min_child_weight=25, subsample=0.85,
                    colsample_bytree=0.65, reg_lambda=3.0, learning_rate=0.015),
     8000, [101, 202, 303]),
    ("exp068", dict(max_depth=6, min_child_weight=25, subsample=0.85,
                    colsample_bytree=0.65, reg_lambda=3.0, learning_rate=0.015),
     8000, [101]),
]
common = dict(objective="reg:squarederror", eval_metric="rmse",
              device="cuda", tree_method="hist", max_bin=255, verbosity=0)
gkf = GroupKFold(5)

results = []
for tag, p, nrounds, seeds in RUNS:
    t0 = time.time()
    print(f"\n=== {tag}  depth={p['max_depth']} lr={p['learning_rate']} nrounds={nrounds} seeds={seeds} ===", flush=True)
    oof_d_bag = np.zeros(len(y_d), np.float32)
    tp_bag = np.zeros(len(Xt), np.float32)
    for sd in seeds:
        params = {**common, **p, "seed": sd}
        oof_d = np.zeros(len(y_d), np.float32)
        tp_folds = []
        for fi, (tr, va) in enumerate(gkf.split(X, y_d, grp)):
            dtr = xgb.DMatrix(X[tr], label=y_d[tr])
            dva = xgb.DMatrix(X[va], label=y_d[va])
            dte = xgb.DMatrix(Xt)
            m = xgb.train(params, dtr, num_boost_round=nrounds,
                          evals=[(dva, "v")], early_stopping_rounds=300, verbose_eval=0)
            oof_d[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
            tp_folds.append(m.predict(dte, iteration_range=(0, m.best_iteration + 1)))
            print(f"  seed{sd} fold{fi}: best={m.best_iteration} full={rmse(lk[va]+oof_d[va], y_a[va]):.4f}", flush=True)
        oof_d_bag += oof_d / len(seeds)
        tp_bag += np.mean(np.column_stack(tp_folds), axis=1).astype(np.float32) / len(seeds)
    oof_abs = (lk + oof_d_bag).astype(np.float32)
    test_abs = (tlk + tp_bag).astype(np.float32)
    r_lat = rmse(oof_abs[is_lat], y_a[is_lat])
    np.save(ROOT / f"experiments/oof/oof_xgboost_{tag}.npy", oof_abs)
    np.save(ROOT / f"experiments/test_preds/test_xgboost_{tag}.npy", test_abs)
    elapsed = time.time() - t0
    print(f"  {tag} lateral RMSE: {r_lat:.4f}   ({elapsed:.0f}s)", flush=True)
    results.append({"exp": tag, "lat": r_lat, "params": p, "elapsed": elapsed, "seeds": seeds})

json.dump(results, open(ROOT / "experiments/results/exp066_068.json", "w"), indent=2)
print("\n=== summary ===")
for r in results: print(f"  {r['exp']}: lat={r['lat']:.4f}")
