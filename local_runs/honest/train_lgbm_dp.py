import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupKFold

tr = pd.read_parquet("local_runs/honest/train_feats.parquet")
dp = pd.read_parquet("local_runs/align/dp_feats.parquet")
trt = tr[tr.is_target==1].copy()
trt = trt.merge(dp, on=["well","MD"], how="left")
for c in ["dp1_delta","dp2_delta","dp_mean_delta"]:
    trt[c]=trt[c].fillna(0)

BASE=["MD","Z","X","Y","GR","dMD","dZ","dX","dY","dXY","lastTVT","lastZ","lastMD",
      "slope_Z","slope_MD","slope_Z_full","slope_MD_full",
      "geom_pred_Z","geom_pred_MD","geom_pred_negZ",
      "gr_roll_mean_25","gr_roll_std_25","gr_roll_mean_101","gr_d1","n_known","frac_along"]
DPF=["dp1_delta","dp2_delta","dp_mean_delta"]

def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
y=trt["delta"].values; base=trt["lastTVT"].values; ytrue=trt["TVT"].values; groups=trt["well"].values
params=dict(objective="regression",metric="rmse",learning_rate=0.03,num_leaves=63,
            min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,
            reg_lambda=5.0,n_estimators=1500,verbose=-1)
gkf=GroupKFold(5)
for feats,tag in [(BASE,"geom-only"),(BASE+DPF,"geom+DP")]:
    X=trt[feats].values
    oof=np.zeros(len(trt))
    for tri,vai in gkf.split(X,y,groups):
        m=lgb.LGBMRegressor(**params)
        m.fit(X[tri],y[tri],eval_set=[(X[vai],y[vai])],callbacks=[lgb.early_stopping(80,verbose=False)])
        oof[vai]=m.predict(X[vai])
    print(f"{tag:12s} OOF pooled RMSE = {rmse(base+oof,ytrue):.4f}")
print(f"const baseline = {rmse(base,ytrue):.4f}")
