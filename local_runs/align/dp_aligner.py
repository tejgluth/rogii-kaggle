import numpy as np, sys, time
from numba import njit
sys.path.insert(0,"local_runs/align")
from harness import load_wells, pooled_rmse

# ---------- GR calibration diagnostic ----------
def calibrate(w):
    """Regress lateral GR onto typewell-GR-at-TVT over the KNOWN zone.
    Returns (a,b) so that calibrated_lat_gr = a*GR + b matches typewell scale."""
    kn=w["known"]
    if kn.sum()<20: return 1.0,0.0
    tvt_k=w["TVT_input"][kn]
    gr_k=w["GR"][kn]
    tw_at=np.interp(tvt_k, w["tw_tvt"], w["tw_gr"])
    # robust-ish least squares gr_k -> tw_at
    A=np.vstack([gr_k, np.ones_like(gr_k)]).T
    try:
        coef,_,_,_=np.linalg.lstsq(A, tw_at, rcond=None)
        a,b=coef
        if not (0.2<a<5): a,b=1.0,0.0
    except Exception:
        a,b=1.0,0.0
    return float(a),float(b)

@njit(cache=True, fastmath=True)
def _dp(emit, vmin, step, S, last_idx, lam, anchor_w, vmax_step):
    """DP min-cost smooth path. emit: (T,S) emission cost. Returns idx path (T,)."""
    T=emit.shape[0]
    INF=1e18
    cost=np.full(S, INF)
    # init: anchor pull to last_idx + emission[0]
    for s in range(S):
        cost[s]=emit[0,s]+anchor_w*((s-last_idx)*step)**2
    back=np.zeros((T,S), dtype=np.int32)
    prev=cost.copy()
    cur=np.empty(S)
    for t in range(1,T):
        for s in range(S):
            best=INF; bi=s
            lo=s-vmax_step; hi=s+vmax_step
            if lo<0: lo=0
            if hi>S-1: hi=S-1
            for ps in range(lo,hi+1):
                d=(s-ps)*step
                c=prev[ps]+lam*d*d
                if c<best:
                    best=c; bi=ps
            cur[s]=best+emit[t,s]
            back[t,s]=bi
        for s in range(S):
            prev[s]=cur[s]
    # terminal: pick min
    path=np.zeros(T, dtype=np.int32)
    bs=0; bc=prev[0]
    for s in range(1,S):
        if prev[s]<bc: bc=prev[s]; bs=s
    path[T-1]=bs
    for t in range(T-1,0,-1):
        path[t-1]=back[t,path[t]]
    return path

def dp_align(w, lam=8.0, sigma=10.0, margin=70.0, step=0.5, anchor_w=0.05,
             vmax_ftrow=1.0, calib=True, prior_w=0.0, prior_slope=False):
    ev=~w["known"]
    n=ev.sum()
    kn=w["known"]
    last_tvt=w["TVT_input"][kn][-1]
    gr=w["GR"][ev].copy()
    if calib:
        a,b=calibrate(w); gr=a*gr+b
    tw_tvt=w["tw_tvt"]; tw_gr=w["tw_gr"]
    # geometric prior path: const, or slope-extrapolated dip from known tail
    if prior_slope and kn.sum()>=30:
        mdk=w["MD"][kn]; tvtk=w["TVT_input"][kn]
        m=min(200,kn.sum())
        sl=np.polyfit(mdk[-m:], tvtk[-m:],1)[0]
        prior=last_tvt+sl*(w["MD"][ev]-mdk[-1])
    else:
        prior=np.full(n, last_tvt)
    # state grid centered on last_tvt
    vlo=last_tvt-margin; vhi=last_tvt+margin
    S=int((vhi-vlo)/step)+1
    grid=vlo+np.arange(S)*step
    tw_on_grid=np.interp(grid, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    diff=gr[:,None]-tw_on_grid[None,:]
    emit=(diff*diff)/(2*sigma*sigma)
    if prior_w>0:
        pd_=grid[None,:]-prior[:,None]
        emit=emit+prior_w*pd_*pd_
    last_idx=int(round((last_tvt-vlo)/step))
    vmax_step=max(1,int(round(vmax_ftrow/step)))
    path=_dp(emit.astype(np.float64), float(vlo), float(step), S, last_idx, float(lam), float(anchor_w), vmax_step)
    return grid[path]

if __name__=="__main__":
    wells=load_wells()
    # calibration stats
    aa=[];
    for w in wells:
        a,b=calibrate(w); aa.append(a)
    aa=np.array(aa)
    print(f"GR calib slope a: median {np.median(aa):.3f} p10 {np.percentile(aa,10):.3f} p90 {np.percentile(aa,90):.3f}")
    # quick eval on a subset for speed
    rng=np.random.default_rng(0)
    sub=set(rng.choice([w["well"] for w in wells], 120, replace=False)) | {"000d7d20","00bbac68","00e12e8b"}
    t0=time.time()
    def ev(f,tag):
        r,per=pooled_rmse(wells, f, subset=sub)
        tw=[p for p in per if p[3]]
        twr=np.sqrt(np.sum([p[1]**2*p[2] for p in tw])/np.sum([p[2] for p in tw]))
        print(f"{tag}: pooled(sub)={r:.3f}  test-dup={twr:.3f}  ({time.time()-t0:.0f}s)")
    ev(lambda w: np.full((~w["known"]).sum(), w["TVT_input"][w["known"]][-1]), "const           ")
    for pw in [0.02, 0.05, 0.1, 0.2, 0.5]:
        ev(lambda w,pw=pw: dp_align(w, lam=8.0, sigma=10.0, margin=40.0, prior_w=pw),
           f"prior_w={pw} m40    ")
    for pw in [0.1,0.2]:
        ev(lambda w,pw=pw: dp_align(w, lam=8.0, sigma=10.0, margin=40.0, prior_w=pw, prior_slope=True),
           f"prior_w={pw} slope  ")
