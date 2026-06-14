import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupKFold

tr = pd.read_parquet("local_runs/honest/train_feats.parquet")
te = pd.read_parquet("local_runs/honest/test_feats.parquet")

FEATS = ["MD","Z","X","Y","GR","dMD","dZ","dX","dY","dXY","lastTVT","lastZ","lastMD",
         "slope_Z","slope_MD","slope_Z_full","slope_MD_full",
         "geom_pred_Z","geom_pred_MD","geom_pred_negZ",
         "gr_roll_mean_25","gr_roll_std_25","gr_roll_mean_101","gr_d1",
         "n_known","frac_along"]

# Train only on target (unknown) rows -- mirror test exactly
trt = tr[tr.is_target==1].reset_index(drop=True)
X = trt[FEATS].values
y = trt["delta"].values            # predict TVT - lastTVT
base = trt["lastTVT"].values
ytrue = trt["TVT"].values
groups = trt["well"].values

def rmse(a,b): return np.sqrt(np.mean((a-b)**2))

params = dict(objective="regression", metric="rmse", learning_rate=0.03,
              num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, n_estimators=2000, verbose=-1)

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(trt))
test_pred = np.zeros(len(te))
for fold,(tri,vai) in enumerate(gkf.split(X,y,groups)):
    m = lgb.LGBMRegressor(**params)
    m.fit(X[tri],y[tri],eval_set=[(X[vai],y[vai])],
          callbacks=[lgb.early_stopping(100,verbose=False)])
    oof[vai] = m.predict(X[vai])
    test_pred += m.predict(te[FEATS].values)/5
    fr = rmse(base[vai]+oof[vai], ytrue[vai])
    print(f"fold{fold} pooled RMSE {fr:.4f}  best_iter {m.best_iteration_}")

pred = base + oof
print("="*40)
print(f"OOF pooled RMSE (delta model): {rmse(pred, ytrue):.4f}")
print(f"const baseline RMSE          : {rmse(base, ytrue):.4f}")
np.save("local_runs/honest/oof_lgbm.npy", oof)
np.save("local_runs/honest/test_lgbm.npy", test_pred)
# feature importance
imp = sorted(zip(FEATS, m.feature_importances_), key=lambda x:-x[1])
print("top feats:", imp[:12])
