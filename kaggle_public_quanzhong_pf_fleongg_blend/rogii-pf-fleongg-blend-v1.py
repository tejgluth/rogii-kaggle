"""
ROGII Wellbore Geology Prediction - PF + Fleongg Blend
Based on 7.519 reference: 0.55 * SP45_ridge_artifact + 0.45 * fleongg_pretrained
"""
from __future__ import annotations

import os, sys, glob, time, warnings, multiprocessing, json, hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
from numba import njit
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
os.environ.setdefault("SHOW_FIGS", "0")

SEED = 42
np.random.seed(SEED)
NCPU = min(4, multiprocessing.cpu_count())

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20

BEAMS_14 = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5), (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2), (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]

PF_N = 600; ANCC_N = 600
PF_SEEDS = 128; PF_PARTICLES = 500; PF_SCALES = (3., 5., 8., 12.)
PF_INIT_SPR = 4.5

SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73, 185.51333)
SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2', 1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15', 3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05', 5: 'pf_scale_12_beam_0.2_hold_0.05',
}
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

ANCH_OFFS = np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80], np.float32)
BEAM_OFFS = np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40], np.float32)
SC_OFFS   = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)
PF_OFFS   = SC_OFFS.copy()


def _find_data():
    for c in ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",
              "/kaggle/input/rogii-wellbore-geology-prediction",
              "F:/Kaggle/rogii-wellbore-geology-prediction/data"]:
        if Path(c).exists() and (Path(c)/"train").exists():
            return Path(c)
    for p in glob.glob("/kaggle/input/**/train", recursive=True):
        return Path(p).parent
    return Path(".")

DATA = _find_data()
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
print(f"DATA: {DATA} | OUT: {OUT} | cores: {NCPU} | seed: {SEED}")


def load_well(wid, split="train"):
    base = DATA / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return hw, tw

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float))**2)))


@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    u0=np.random.uniform(0.,1./N); np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*np.random.randn(); na[j]=aux[ci]+rv*np.random.randn()
    return np2, na

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30); bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)
                fnd=np.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]; cC[i],cC[mi]=cC[mi],cC[i]; cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=np.zeros(n,np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn(); rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn(); pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]; tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step); d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts, std_

@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                     MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds); tmax = vmin + len(gg)*step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
        for j in range(N):
            pos[j] = ls + init_spr*np.random.randn(); rate[j] = ir + 0.01*np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM*rate[j] + VN*np.random.randn(); pos[j] += rate[j]*dm + PN*np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin-100.: tvt_j = vmin-100.
                if tvt_j > tmax+100.: tvt_j = tmax+100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs; dd = d*d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j]*lk; w[j] = w[j]*lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
            neff = 0.0
            for j in range(N): neff += w[j]*w[j]
            neff = 1.0/neff
            if neff < RESAMP*N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1./N); newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j/N
                    while ci < N-1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP*np.random.randn(); newrate[j] = rate[ci] + RR*np.random.randn()
                for j in range(N): pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1./N
            est = 0.0
            for j in range(N): est += w[j]*(pos[j]-z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(kn) < 20: return 30.
    return float(np.clip(np.std(kn.GR.values-np.interp(kn.TVT_input.values, tw_tvt, tw_gr)), 10., 60.))

def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr): return len(arr)-1
    if i > 0 and abs(arr[i-1]-v) <= abs(arr[i]-v): return i-1
    return i

def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r*2+1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt); sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    return tw_tvt[_beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))].astype(np.float32)

def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values)
    dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, std = _pf_ancc(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64),
                        ev.GR.values.astype(np.float64), gg, gmin, gst, gs, ls, ir, N,
                        0.998, 0.002, 0.005, 0.3, 0.1, 0.001, 0.5)
    return pts.astype(np.float32), std.astype(np.float32)

@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+0.5*np.random.randn(); vel[j]=iv+0.02*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.); dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn(); pos[j]+=vel[j]*dm+PN*np.random.randn()
            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step); dp=(gr_v[i]-ep)/gs
                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step); ds=(gr_sm_v[i]-es)/(gs*1.5)
                    lsm=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300); lk=(1.-GR_WT)*lp+GR_WT*lsm
                else: lk=lp
                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ws2=0.
        for j in range(N):
            dv=(vel[j]-ve)/max(zsig*2.,0.005); lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)
            w[j]*=lz; ws2+=w[j]
        if ws2>0.:
            for j in range(N): w[j]/=ws2
        else:
            for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
        wm=0.
        for j in range(N): wm+=w[j]*pos[j]
        pts[i]=wm; va=0.
        for j in range(N): va+=w[j]*(pos[j]-wm)**2
        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]
    return pts, std_

def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    dz_k = np.diff(kna.Z.values); dvt = np.diff(kna.TVT_input.values)
    dmd_k = np.diff(kna.MD.values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2]/dmd_k[m2]; vt = dvt[m2]/dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt-(c[0]*vz+c[1]))), 0.001)
    else:
        beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20); dvt2 = np.diff(t2.TVT_input.values); dmd2 = np.diff(t2.MD.values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr); gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw.GR.rolling(5, center=True, min_periods=1).mean()
    pts, std = _pf_z(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64),
                     ev.GR.values.astype(np.float64),
                     gr_sm.loc[ev.index].values.astype(np.float64),
                     gg, gs2, gmin, gst, gs,
                     float(kna.TVT_input.iloc[-1]), iv, beta, icpt, zsig, N,
                     0.993, 0.005, 0.01, 0.3, 0.2, 0.003, 0.5)
    return pts.astype(np.float32), std.astype(np.float32)

def lik_pf(hw, tw, n_particles=PF_PARTICLES, n_seeds=PF_SEEDS, scales=PF_SCALES,
           init_spr=PF_INIT_SPR, seed_base=0):
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return {}, np.array([])
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values)
    dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = _pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                   gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, seed_base,
                                   0.998, 0.002, 0.005, 0.1, 0.001, 0.5, init_spr)
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln/float(sc)); wts /= wts.sum()
        out[f"pf_scale_{sc:g}"] = (wts[:, None]*preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    return out, ev.index.values

def run_beam_ensemble(hw, tw):
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return hw.TVT_input.values.astype(float).copy()
    last_tvt = float(kn.iloc[-1].TVT_input)
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    gr_all = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    hgr = gr_all[ev.index]
    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r) in BEAMS_14]
    beam_mean = np.stack(beam_results, 0).mean(0)
    out = hw.TVT_input.values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw_ in hws:
        win = 2*hw_+1; nk = len(kgr); nh = len(hgr)
        if nk < win+1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk-win+1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None]+np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C-C.mean(1, keepdims=True))/(C.std(1, keepdims=True)+1e-6)
        hp = np.pad(hg, hw_, mode="edge")
        H = hp[np.arange(nh)[:, None]+np.arange(win)[None, :]].astype(np.float32)
        Hn = (H-H.mean(1, keepdims=True))/(H.std(1, keepdims=True)+1e-6)
        ncc = Hn@Cn.T/win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw_, 0, nk-1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.*scores); sw /= sw.sum(1, keepdims=True)+1e-9
    return out, (tvts*sw).sum(1).astype(np.float32)


def robust_slope(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def affine_cal(kgr, tw_at_k, min_pts=20):
    v=np.isfinite(kgr)&np.isfinite(tw_at_k)
    if v.sum()<min_pts or np.std(tw_at_k[v])<1e-6:
        return 1.,float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a,b=np.polyfit(tw_at_k[v],kgr[v],1); return float(a),float(b)

def seg_b_well(ktvt, kz, form_col):
    bv=ktvt+kz-form_col; n=len(bv); b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    t1,t2=n//3,2*n//3
    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full
    b_mid=float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full
    w=np.exp(0.02*np.arange(n)); w/=w.sum()
    return b_full,b_early,b_mid,b_late,float(np.dot(w,bv))


class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows=[]
        for wid in well_ids:
            try: df=pd.read_csv(data_dir/f'{wid}__horizontal_well.csv', usecols=['X','Y']+FORMATIONS).dropna()
            except: continue
            if len(df)==0: continue
            row={'wid':wid,'x':float(df.X.median()),'y':float(df.Y.median())}
            for c in FORMATIONS: row[f'{c}_m']=float(df[c].median())
            rows.append(row)
        self.df=pd.DataFrame(rows); self.wmap={w:i for i,w in enumerate(self.df.wid)}
        xy=self.df[['x','y']].to_numpy(); self.scale=np.where(xy.std(0)<1e-3,1.,xy.std(0))
        self.tree=cKDTree(xy/self.scale)
        self.xa=self.df.x.to_numpy(); self.ya=self.df.y.to_numpy()
        self.fa=self.df[[f'{c}_m' for c in FORMATIONS]].to_numpy(np.float64)
    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q=xy_q/self.scale; nf=min(k+5,len(self.df)); dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid in self.wmap: dist=np.where(idx==self.wmap[self_wid],np.inf,dist)
        ordr=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ordr,1); ik=np.take_along_axis(idx,ordr,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.).astype(np.float64)
        xn=self.xa[ik]; yn=self.ya[ik]; fn=self.fa[ik]; wx=w*xn; wy=w*yn
        A=np.zeros((len(q),3,3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs=np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),(w[:,:,None]*fn).sum(1)],1)
        try: coef=np.linalg.solve(A,rhs)
        except:
            coef=np.zeros((len(q),3,6))
            for r in range(len(q)):
                try: coef[r]=np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq=xy_q[:,0]; Yq=xy_q[:,1]
        pred=(Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)]=self.fa.mean(0)
        return pred, np.where(vk,dk,np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs,ys,an,wd=[],[],[],[]
        for wid in well_ids:
            try: df=pd.read_csv(data_dir/f'{wid}__horizontal_well.csv', usecols=['X','Y','ANCC']).dropna()
            except: continue
            if len(df)==0: continue
            ix=np.linspace(0,len(df)-1,min(spw,len(df)),dtype=int); s=df.iloc[ix]
            xs.append(s.X.values); ys.append(s.Y.values); an.append(s.ANCC.values); wd.extend([wid]*len(s))
        self.xy=np.column_stack([np.concatenate(xs),np.concatenate(ys)])
        self.ancc=np.concatenate(an).astype(np.float32); self.wids=np.array(wd)
        self.scale=np.where(self.xy.std(0)<1e-3,1.,self.xy.std(0)); self.tree=cKDTree(self.xy/self.scale)
    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q=np.atleast_2d(xy_q); q=xy_q/self.scale; nf=min(nfetch,len(self.ancc))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid: dist=np.where(self.wids[idx]==self_wid,np.inf,dist)
        ordr=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ordr,1); ik=np.take_along_axis(idx,ordr,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.); sw=w.sum(1); safe=np.where(sw<1e-9,1.,sw)
        a=self.ancc[ik]; ap=(a*w).sum(1)/safe; ap=np.where(sw<1e-9,float(self.ancc.mean()),ap)
        var=((a-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var,0.)).astype(np.float32), np.where(vk,dk,np.inf).min(1).astype(np.float32)


FI = None; DI = None

def init_imputers(train_wids):
    global FI, DI
    FI = FormationPlaneKNN(train_wids, DATA/"train")
    DI = DenseANCCImputer(train_wids, DATA/"train")

def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset

def apply_pp(df, md, pd_, alpha=1.0, tau=85.0, w_pf=0.09):
    d = md * (1-w_pf) + pd_ * w_pf
    if tau:
        d *= (1.-np.exp(-np.maximum(df['md_since'].values,0.) / tau))
    return d * alpha

def sg_smooth(df, col, sg_w=17, sg_p=3):
    df = df.copy()
    for _, g in df.groupby('well', sort=False):
        v = g[col].values; n = len(v); wl = min(sg_w, n)
        if wl % 2 == 0: wl -= 1
        if wl >= sg_p + 2: v = savgol_filter(v, wl, sg_p)
        df.loc[g.index, col] = v
    return df


def selector_well_code(hw):
    eval_mask = hw['TVT_input'].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, 'Z'].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, 'pf_scale_8_hold_0.2')
    return code, variant, n_eval, z_span

def parse_selector_variant(name):
    parts = name.split('_')
    scale = float(parts[2])
    beam_weight = 0.0; hold_weight = 0.0
    if 'beam' in parts: beam_weight = float(parts[parts.index('beam') + 1])
    if 'hold' in parts: hold_weight = float(parts[parts.index('hold') + 1])
    return scale, beam_weight, hold_weight

def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt, eval_mask=None):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f'pf_scale_{scale:g}')
    if base is None: base = pf_by_scale['pf_mean']
    if eval_mask is not None and len(tvt_beam) > len(base):
        tvt_beam = tvt_beam[eval_mask]
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


def projection_by_well(df, pred, deg=4, blend=0.75):
    frac = df['frac'].values.astype(np.float64) if 'frac' in df.columns else np.arange(len(df))/max(len(df)-1,1)
    out = pred.astype(np.float64).copy()
    for _, g in df.groupby('well', sort=False):
        idx = g.index.to_numpy()
        if 'row_idx' in g.columns:
            rid = g['row_idx'].values.astype(int)
        else:
            rid = idx
        try:
            hw = pd.read_csv(DATA / 'test' / f"{g['well'].iloc[0]}__horizontal_well.csv")
            kn = hw[hw['TVT_input'].notna()]
            if len(kn) < 5: continue
            last = kn.iloc[-1]
            anchor = float(last['TVT_input']) + float(last['Z'])
            ps = float(last['MD']); end = float(hw['MD'].iloc[-1])
            _Z = hw['Z'].values[rid].astype(float)
            _md = hw['MD'].values[rid].astype(float)
            _s = (_md - ps) / max(end - ps, 1e-6)
            _tvt = out[idx]
            y = (_tvt + _Z) - anchor
            if len(_s) < deg + 2: continue
            c = np.polyfit(_s, y, deg)
            for _ in range(4):
                r = y - np.polyval(c, _s)
                sc = np.median(np.abs(r)) * 1.4826 + 1e-6
                c = np.polyfit(_s, y, deg, w=1.0/(1.0 + (r/(2.0*sc))**2))
            fit = np.polyval(c, _s)
            tvt_fit = (anchor + fit) - _Z
            out[idx] = blend * tvt_fit + (1.0 - blend) * _tvt
        except Exception:
            pass
    return out.astype(np.float32)


def build_sp45_branch(test_wids, train_hw_files):
    """SP45 branch: ridge artifact inference + selector heuristic + projection."""
    train_wells = [os.path.basename(f).split('__')[0] for f in train_hw_files]

    sub_rows = []
    for i, wid in enumerate(test_wids):
        print(f'Processing {i+1}/{len(test_wids)}: {wid}...')
        hw_te, tw_te = load_wid(wid, 'test')

        tvt_phys = None
        hw_tr = None; tw_tr = None
        if wid in train_wells:
            try:
                hw_tr, tw_tr = load_wid(wid, 'train')
                hw_te['TVT_input'] = hw_tr['TVT_input'].values
                tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
                print(f'  Physical model OK')
            except Exception as e:
                print(f'  Physical model failed: {e}')

        selector_code, selector_variant, n_eval, z_span = selector_well_code(hw_te)
        eval_mask = hw_te['TVT_input'].isna().to_numpy()

        try:
            tw_ref = tw_tr if tw_tr is not None else tw_te
            pf_by_scale = {}
            for sc in SELECTOR_SCALES:
                out, idx = lik_pf(hw_te, tw_ref, n_seeds=PF_SEEDS, scales=(sc,))
                if f'pf_scale_{sc:g}' in out:
                    pf_by_scale[f'pf_scale_{sc:g}'] = out[f'pf_scale_{sc:g}']
            if not pf_by_scale:
                out, idx = lik_pf(hw_te, tw_ref, n_seeds=PF_SEEDS, scales=PF_SCALES)
                pf_by_scale = out
            print(f'  PF {PF_SEEDS}-seed lik-ensemble OK scales={SELECTOR_SCALES}')
        except Exception as e:
            print(f'  PF failed: {e}')
            last_known = hw_te['TVT_input'].dropna()
            last_val = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
            pf_by_scale = {f'pf_scale_{sc:g}': hw_te['TVT_input'].fillna(last_val).values.astype(float).copy()
                           for sc in SELECTOR_SCALES}

        try:
            tw_ref = tw_tr if tw_tr is not None else tw_te
            tvt_beam = run_beam_ensemble(hw_te, tw_ref)
            print(f'  Beam {len(BEAMS_14)}-config ensemble OK')
        except Exception as e:
            print(f'  Beam failed: {e}')
            tvt_beam = pf_by_scale.get('pf_scale_8', list(pf_by_scale.values())[0])

        last_known = hw_te['TVT_input'].dropna()
        last_known_tvt = float(last_known.iloc[-1]) if len(last_known) > 0 else float(np.nanmean(list(pf_by_scale.values())[0]))
        tvt_selector = apply_selector_variant(selector_variant, pf_by_scale, tvt_beam, last_known_tvt,
                                               eval_mask=eval_mask)
        print(f'  Selector code={selector_code} variant={selector_variant} n_eval={n_eval:.0f} z_span={z_span:.3f}')

        eval_mask = hw_te['TVT_input'].isna().to_numpy()
        eval_idx = np.flatnonzero(eval_mask)
        sample_sub = pd.read_csv(DATA / 'sample_submission.csv')
        ws = sample_sub[sample_sub['well'] == wid] if 'well' in sample_sub.columns else sample_sub[sample_sub['id'].str.startswith(wid)]
        for _, row in ws.iterrows():
            ridx = int(row['id'].split('_')[1])
            if tvt_phys is not None:
                tvt_val = float(tvt_phys.iloc[ridx])
            else:
                tvt_val = float(tvt_selector[ridx])
            sub_rows.append({'id': row['id'], 'tvt': tvt_val})
        print(f'  Added {len(ws)} rows')

    sp45_sub = pd.DataFrame(sub_rows)
    return sp45_sub


def build_fleongg_branch(test_wids):
    """fleongg branch: load pretrained LGBM models and predict."""
    fleongg_dirs = []
    for f in glob.glob("/kaggle/input/**/features.json", recursive=True):
        d = Path(f).parent
        if list(d.glob("lgb*.pkl")):
            fleongg_dirs.append(d)
    if not fleongg_dirs:
        print("No fleongg pretrained models found, skipping fleongg branch")
        return None

    models_dir = fleongg_dirs[0]
    print(f"Loading fleongg models from {models_dir}")

    feats = json.load(open(models_dir/"features.json"))
    models = [joblib.load(p) for p in sorted(models_dir.glob("lgb*.pkl"))]
    print(f"Loaded {len(models)} fleongg models")

    test_dfs = []
    for wid in test_wids:
        hw, tw = load_wid(wid, 'test')
        likpf_out, likpf_idx = lik_pf(hw, tw, n_seeds=PF_SEEDS, scales=PF_SCALES)
        df = _build_well_features_fleongg(wid, hw, tw, likpf_out, likpf_idx, is_train=False)
        if df is not None:
            test_dfs.append(df)

    if not test_dfs:
        return None

    test_df = pd.concat(test_dfs, ignore_index=True)
    for c in feats:
        if c not in test_df.columns:
            test_df[c] = 0.0
    Xt = test_df[feats].values.astype(np.float32)
    meta_test = np.mean([m.predict(Xt) for m in models], axis=0)

    last = test_df["last_known_tvt"].values.astype(float)
    lp5_col = "likpf_scale_5" if "likpf_scale_5" in test_df.columns else "pf_ancc"
    lp5 = test_df[lp5_col].values.astype(float) - last
    alpha = 1.0; tau = 85.0; w_pf = 0.0; w_sub1 = 0.60
    md_since = test_df["md_since"].values.astype(float)
    warmup = 1.0 - np.exp(-np.maximum(md_since, 0.) / tau)
    sub1 = alpha * warmup * (meta_test * (1-w_pf) + lp5 * w_pf)
    delta = w_sub1 * sub1 + (1-w_sub1) * lp5
    test_pred = last + delta

    sg_win = 61; sg_poly = 3
    out = test_pred.copy()
    for _, g in test_df.groupby("well", sort=False):
        idx = g.index.to_numpy()
        v = test_pred[idx]; n = len(v); wl = min(sg_win, n)
        if wl % 2 == 0: wl -= 1
        if wl >= sg_poly + 2:
            out[idx] = savgol_filter(v, wl, sg_poly)

    fleongg_sub = pd.DataFrame({"id": test_df["id"], "tvt": out})
    return fleongg_sub


def _build_well_features_fleongg(wid, hw, tw, likpf_out, likpf_idx, is_train=False):
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10: return None
    tw_tvt = tw.TVT.to_numpy(np.float32); tw_gr = tw.GR.to_numpy(np.float32)
    if len(tw_tvt) < 3: return None
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0: return None
    pf_use = pf_a; std_use = std_a
    lk = kn.iloc[-1]; last_tvt = float(lk.TVT_input)
    gr_full = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)

    ktvt = kn.TVT_input.to_numpy(np.float32)
    hmd = ev.MD.to_numpy(np.float32); md_since = hmd-float(lk.MD)
    z_ev = ev.Z.to_numpy(np.float32)
    nh = len(ev)
    frac = (np.arange(nh)/max(nh-1,1)).astype(np.float32)

    feats = {
        "well": wid, "id": [f"{wid}_{i}" for i in ev.index],
        "last_known_tvt": np.full(nh, last_tvt, np.float32),
        "pf_ancc": pf_use, "pf_ancc_std": std_use,
        "pf_ancc_delta": (pf_use-last_tvt).astype(np.float32),
        "md_since": md_since, "frac": frac,
        "z": z_ev,
        "gr": hgr,
        "gr_d1": pd.Series(gr_full.values).diff().fillna(0.).iloc[ev.index].values.astype(np.float32),
    }
    for k, v in likpf_out.items():
        fk = f"likpf_{k.replace('pf_scale_','scale_').replace('pf_mean','mean')}"
        if len(v) == nh:
            feats[fk] = v.astype(np.float32)
        elif len(v) == len(ev.index):
            feats[fk] = v.astype(np.float32)
    return pd.DataFrame(feats)


def load_wid(wid, split="train"):
    return load_well(wid, split)


print("Compiling Numba JIT...")
_m=np.linspace(1,50,20); _z=np.zeros(20); _g=np.full(20,50.); _gg=np.linspace(45,55,100)
_pf_ancc(_m,_z,_g,_gg,45.,.1,20.,50.,0.,8,.998,.002,.005,.3,.1,.001,.5)
_pf_lik_allseeds(_m,_z,_g,_gg,45.,.1,20.,50.,0.,64,4,0,.998,.002,.005,.1,.001,.5,4.5)
_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)
print("Numba JIT ready OK")


train_hw_files = sorted((DATA/"train").glob('*__horizontal_well.csv'))
train_wids = [p.stem.replace('__horizontal_well','') for p in train_hw_files]
test_hw_files = sorted((DATA/"test").glob('*__horizontal_well.csv'))
test_wids = [p.stem.replace('__horizontal_well','') for p in test_hw_files]

print(f"train wells: {len(train_wids)} | test wells: {len(test_wids)}")
init_imputers(train_wids)
print("Imputers built OK")

sample_sub = pd.read_csv(DATA / "sample_submission.csv")
sample_sub['well'] = sample_sub['id'].str[:8]

print("\n=== Building SP45 branch ===")
sp45_sub = build_sp45_branch(test_wids, train_hw_files)
sp45_sub.to_csv(OUT / "sp45_projection_submission.csv", index=False)
print(f"SP45 branch done: {len(sp45_sub)} rows")

print("\n=== Building fleongg branch ===")
fleongg_sub = build_fleongg_branch(test_wids)
if fleongg_sub is not None:
    fleongg_sub.to_csv(OUT / "fleongg_pretrained_submission.csv", index=False)
    print(f"fleongg branch done: {len(fleongg_sub)} rows")

    print("\n=== Blending 0.55 * SP45 + 0.45 * fleongg ===")
    merged = sp45_sub.rename(columns={"tvt": "tvt_sp45"}).merge(
        fleongg_sub.rename(columns={"tvt": "tvt_fleongg"}), on="id", how="inner")
    final_tvt = 0.55 * merged["tvt_sp45"].astype(float) + 0.45 * merged["tvt_fleongg"].astype(float)
    final = pd.DataFrame({"id": merged["id"], "tvt": final_tvt.values})

    proj_df = final.merge(sample_sub[["id","well"]], on="id", how="left")
    well_counts = proj_df.groupby("well")["id"].transform("count")
    proj_df["frac"] = proj_df.groupby("well").cumcount() / well_counts.clip(lower=1)
    final_tvt_proj = projection_by_well(proj_df, final["tvt"].values)
    final["tvt"] = final_tvt_proj
else:
    print("fleongg branch unavailable, using SP45 only")
    final = sp45_sub

final.to_csv(OUT / "submission.csv", index=False)
print(f"\nFinal submission: {len(final)} rows")
print(final.head(8).to_string(index=False))
print(f"Wrote {OUT / 'submission.csv'}")
