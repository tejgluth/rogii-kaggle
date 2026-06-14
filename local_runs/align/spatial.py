import numpy as np, sys, time
sys.path.insert(0,"local_runs/align")
from harness import load_wells, pooled_rmse
from sklearn.neighbors import KNeighborsRegressor

wells=load_wells()
N=len(wells)
fold=np.array([i%5 for i in range(N)])

def build_knn(train_idx, k=30, max_per=80):
    XY=[]; S=[]
    for i in train_idx:
        w=wells[i]
        # sample rows across whole well (known+lateral), use TVT+Z structural elevation
        n=len(w["MD"])
        sel=np.linspace(0,n-1,min(max_per,n)).astype(int)
        XY.append(np.column_stack([w["MD"][sel]*0+w["MD"][sel],  # placeholder
                                   ]))
    # simpler: gather (X? we don't have X,Y in cache!) -> need X,Y
    return None

# X,Y not in cache; add them
import pandas as pd, glob, os
D="data/rogii-wellbore-geology-prediction"
xy_cache={}
for w in wells:
    f=f"{D}/train/{w['well']}__horizontal_well.csv"
    df=pd.read_csv(f, usecols=["MD","X","Y"]).sort_values("MD")
    xy_cache[w["well"]]=(df["X"].values.astype(float), df["Y"].values.astype(float))

# Build fold KNNs on (X,Y)->S=TVT+Z, predict held-out wells
preds={}
for fo in range(5):
    tr=[i for i in range(N) if fold[i]!=fo]
    XY=[]; S=[]
    for i in tr:
        w=wells[i]; X,Y=xy_cache[w["well"]]
        n=len(w["MD"]); sel=np.linspace(0,n-1,min(80,n)).astype(int)
        s=w["TVT"][sel]+w["Z"][sel]
        XY.append(np.column_stack([X[sel],Y[sel]])); S.append(s)
    XY=np.vstack(XY); S=np.concatenate(S)
    knn=KNeighborsRegressor(n_neighbors=30, weights="distance").fit(XY,S)
    for i in range(N):
        if fold[i]!=fo: continue
        w=wells[i]; X,Y=xy_cache[w["well"]]
        ev=~w["known"]
        Spred=knn.predict(np.column_stack([X[ev],Y[ev]]))
        # anchor: remove per-well bias using known zone S
        kn=w["known"]
        Sknown_pred=knn.predict(np.column_stack([X[kn],Y[kn]]))
        bias=np.median((w["TVT"][kn]+w["Z"][kn]) - Sknown_pred)
        preds[w["well"]]=( -w["Z"][ev]+Spred,           # raw
                           -w["Z"][ev]+Spred+bias )      # anchored

def pred_raw(w): return preds[w["well"]][0]
def pred_anch(w): return preds[w["well"]][1]
def const(w): return np.full((~w["known"]).sum(), w["TVT_input"][w["known"]][-1])

for nm,f in [("const",const),("spatial_raw",pred_raw),("spatial_anchored",pred_anch)]:
    r,per=pooled_rmse(wells,f)
    tw=[p for p in per if p[3]]
    twr=np.sqrt(np.sum([p[1]**2*p[2] for p in tw])/np.sum([p[2] for p in tw]))
    print(f"{nm:18s} pooled(all)={r:.3f}  test-dup={twr:.3f}")
