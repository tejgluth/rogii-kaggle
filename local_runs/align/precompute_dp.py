import numpy as np, sys, time, pandas as pd
sys.path.insert(0,"local_runs/align")
from harness import load_wells
from dp_aligner import dp_align

wells=load_wells()
rows=[]
t0=time.time()
for i,w in enumerate(wells):
    ev=~w["known"]
    last=w["TVT_input"][w["known"]][-1]
    # two DP settings + NCC-free: gentle and stiffer
    dp1=dp_align(w, lam=8.0, sigma=10.0, margin=40.0, prior_w=0.05)
    dp2=dp_align(w, lam=16.0, sigma=14.0, margin=40.0, prior_w=0.1)
    md=w["MD"][ev]
    rows.append(pd.DataFrame({
        "well":w["well"],
        "MD":md,
        "dp1_delta":dp1-last,
        "dp2_delta":dp2-last,
        "dp_mean_delta":0.5*(dp1+dp2)-last,
    }))
    if (i+1)%150==0: print(f"  {i+1}/{len(wells)} {time.time()-t0:.0f}s")
out=pd.concat(rows,ignore_index=True)
out.to_parquet("local_runs/align/dp_feats.parquet")
print("saved", out.shape, f"{time.time()-t0:.0f}s")
