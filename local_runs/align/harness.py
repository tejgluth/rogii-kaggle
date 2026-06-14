"""Fast honest-CV harness for geosteering alignment R&D.
Caches per-well arrays; evaluates aligners on eval (TVT_input-NaN) rows.
"""
import pandas as pd, numpy as np, glob, os, pickle

D = "data/rogii-wellbore-geology-prediction"
TEST_WELLS = {"000d7d20","00bbac68","00e12e8b"}
CACHE = "local_runs/align/wells.pkl"

def load_wells():
    if os.path.exists(CACHE):
        with open(CACHE,"rb") as f: return pickle.load(f)
    wells=[]
    for hw in sorted(glob.glob(f"{D}/train/*__horizontal_well.csv")):
        pre=os.path.basename(hw).split("__")[0]
        df=pd.read_csv(hw).sort_values("MD").reset_index(drop=True)
        twf=hw.replace("__horizontal_well.csv","__typewell.csv")
        if not os.path.exists(twf): continue
        tw=pd.read_csv(twf).dropna(subset=["TVT","GR"]).sort_values("TVT")
        known=df["TVT_input"].notna().values
        if known.sum()<10 or (~known).sum()<5: continue
        # GR fill: interp along trace
        gr=pd.Series(df["GR"].values.astype(float)).interpolate(limit_direction="both").bfill().ffill().values
        gr=np.nan_to_num(gr, nan=float(np.nanmedian(tw["GR"].values)))
        wells.append(dict(
            well=pre, is_test=pre in TEST_WELLS,
            MD=df["MD"].values.astype(np.float64),
            Z=df["Z"].values.astype(np.float64),
            GR=gr.astype(np.float64),
            TVT=df["TVT"].values.astype(np.float64) if "TVT" in df else None,
            TVT_input=df["TVT_input"].values.astype(np.float64),
            known=known,
            tw_tvt=tw["TVT"].values.astype(np.float64),
            tw_gr=tw["GR"].values.astype(np.float64),
        ))
    with open(CACHE,"wb") as f: pickle.dump(wells,f)
    return wells

def pooled_rmse(wells, predict_fn, subset=None, weight_easy=False):
    """predict_fn(w)->array of TVT preds for eval rows. Returns pooled RMSE over eval rows."""
    se=0.0; n=0; per=[]
    for w in wells:
        if subset is not None and w["well"] not in subset: continue
        if w["TVT"] is None: continue
        ev=~w["known"]
        if ev.sum()==0: continue
        pred=predict_fn(w)
        ytrue=w["TVT"][ev]
        e=pred-ytrue
        se+=np.sum(e**2); n+=len(e)
        per.append((w["well"], np.sqrt(np.mean(e**2)), ev.sum(), w["is_test"]))
    return np.sqrt(se/n), per

if __name__=="__main__":
    wells=load_wells()
    print("wells", len(wells), "test-dup present:", sum(w["is_test"] for w in wells))
    # const baseline
    def const(w):
        lt=w["TVT_input"][w["known"]][-1]
        return np.full((~w["known"]).sum(), lt)
    r,per=pooled_rmse(wells, const)
    print(f"const baseline pooled RMSE (all): {r:.3f}")
    # per-well const RMSE distribution
    rs=np.array([p[1] for p in per])
    print(f"per-well const RMSE: median {np.median(rs):.2f} mean {np.mean(rs):.2f} p90 {np.percentile(rs,90):.2f}")
    # test-dup wells const (vs train-copy TVT, a proxy)
    for p in per:
        if p[3]: print(f"  TEST-DUP {p[0]}: const RMSE {p[1]:.3f} (n={p[2]})")
