"""exp056: residual probe. Fit a LightGBM on (y_true - exp050_stack_oof) using cached
exp026 features. If the residual has signal, the model achieves CV RMSE less than the
random-baseline RMSE on the lateral set, and adding this to exp050 improves the stack.

This is a saturation test: it tells us whether the OOF pool truly has nothing left."""
import json, time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

CACHE = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
X = CACHE["X"]; y_a = CACHE["y_abs"]; lk = CACHE["last_known"]
is_lat = CACHE["is_lateral"]; grp = np.asarray(CACHE["groups"])
Xt = CACHE["Xt"]; tlk = CACHE["test_last_known"]

oof_stack = np.load(ROOT / "experiments/oof/oof_stack_exp050.npy")
test_stack = np.load(ROOT / "experiments/test_preds/test_stack_exp050.npy")
residual = (y_a - oof_stack).astype(np.float32)

print(f"residual stats: mean={residual.mean():.4f} std={residual.std():.4f}")
print(f"residual lateral RMSE (= current stack lateral RMSE): {rmse(residual[is_lat], 0*residual[is_lat]):.4f}")

gkf = GroupKFold(5)
oof_resid = np.zeros(len(residual), np.float32)
test_resid_folds = []
params = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, lambda_l1=1.0, lambda_l2=2.0, verbose=-1)
for fi, (tr, va) in enumerate(gkf.split(X, residual, grp)):
    dtr = lgb.Dataset(X[tr], label=residual[tr])
    dva = lgb.Dataset(X[va], label=residual[va], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_resid[va] = m.predict(X[va], num_iteration=m.best_iteration)
    test_resid_folds.append(m.predict(Xt, num_iteration=m.best_iteration))
    r_va = rmse(oof_resid[va][is_lat[va]], residual[va][is_lat[va]])
    print(f"  fold{fi}: best_iter={m.best_iteration} residual_va_lat_rmse={r_va:.4f}")

# Check whether subtracting predicted residual lowers stack
new_oof = oof_stack + oof_resid
new_test = test_stack + np.mean(np.column_stack(test_resid_folds), axis=1).astype(np.float32)
r_old = rmse(oof_stack[is_lat], y_a[is_lat])
r_new = rmse(new_oof[is_lat], y_a[is_lat])
print(f"\nstack lateral RMSE  old={r_old:.4f}  new={r_new:.4f}  delta={r_new-r_old:+.4f}")

# Try shrinkage
best = (r_new, 1.0)
for w in np.arange(0.05, 1.01, 0.05):
    cand = oof_stack + w * oof_resid
    r = rmse(cand[is_lat], y_a[is_lat])
    if r < best[0]:
        best = (r, w)
print(f"best shrinkage: w={best[1]:.2f}  lat={best[0]:.4f}")

np.save(ROOT / "experiments/oof/oof_lgbm_exp056_resid.npy", oof_resid.astype(np.float32))
np.save(ROOT / "experiments/test_preds/test_lgbm_exp056_resid.npy",
        np.mean(np.column_stack(test_resid_folds), axis=1).astype(np.float32))

# Also save the corrected stack with best shrinkage
final_oof = (oof_stack + best[1] * oof_resid).astype(np.float32)
final_test = (test_stack + best[1] * np.mean(np.column_stack(test_resid_folds), axis=1)).astype(np.float32)
np.save(ROOT / "experiments/oof/oof_stack_exp056.npy", final_oof)
np.save(ROOT / "experiments/test_preds/test_stack_exp056.npy", final_test)
json.dump({"experiment_id":"exp056","phase":"stacking","model":"residual_probe_lgbm",
           "stack_old": r_old, "stack_new_w1": r_new,
           "best_shrinkage_w": best[1], "best_lat_rmse": best[0]},
          open(ROOT / "experiments/results/exp056.json", "w"), indent=2)
