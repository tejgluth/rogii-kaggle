"""
ROGII Wellbore Geology Prediction - Gold-Level Solution (v8)
============================================================
V8 upgrades over V7:
  A. Particle Filter (Numba JIT) - tracks TVT in typewell GR space (biggest single signal)
  B. GR NaN interpolation from typewell BEFORE all signal processing (data quality fix)
  C. Dense ANCC spatial KNN proxy (predict ANCC from (X,Y) across all train wells)
  D. XGBoost + CatBoost added to LightGBM ensemble (model diversity)
  E. Hill-climbing blend over all 5 models

Inherited from V7:
  - Physical invariant TVT+Z-formation=const, Formation-datum KNN
  - Multi-scale NCC (8,15,25) softmax, Beam search (7 cfgs)
  - Slope extrapolation, GR features, typewell-diff offsets
  - Post-proc: alpha x tau grid + stratigraphic projection + Savitzky-Golay
  - Target: TVT delta from last anchor
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")

# Optional models - graceful fallback if unavailable
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except Exception:
    HAS_CB = False
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        # Support both @njit and @njit(cache=True, ...) usage
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(f):
            return f
        return deco

print(f"XGBoost={HAS_XGB}  CatBoost={HAS_CB}  Numba={HAS_NUMBA}")

# -- Paths --------------------------------------------------------------------
BASE_DIR = next(
    (p for p in [
        "data/rogii-wellbore-geology-prediction",
        "data/rogii-wellbore-geology-prediction",
    ] if os.path.exists(p) and os.listdir(p)),
    "/kaggle/input/rogii-wellbore-geology-prediction"
)
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
TEST_DIR   = os.path.join(BASE_DIR, "test")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

print(f"BASE_DIR: {BASE_DIR}")

# =============================================================================
# A. PARTICLE FILTER (Numba JIT)
# =============================================================================
@njit(cache=True, fastmath=True)
def _interp1(grid, x, vmin, step):
    """Linear interp of value on uniform grid starting at vmin with given step."""
    n = len(grid)
    pos = (x - vmin) / step
    if pos <= 0:
        return grid[0]
    if pos >= n - 1:
        return grid[n - 1]
    i = int(pos)
    frac = pos - i
    return grid[i] * (1.0 - frac) + grid[i + 1] * frac

@njit(cache=True, fastmath=True)
def _pf_run(q_gr, q_dmd, tw_grid, vmin, step, last_tvt,
            N, ALPHA, RN, PN, GR_SIG, RESAMP, seed):
    """
    Particle filter tracking TVT relative to last_tvt.
    State per particle: position (TVT delta) + rate (velocity per MD unit).
    Returns mean TVT-delta estimate per query step + std.
    """
    np.random.seed(seed)
    Nq = len(q_gr)
    out_mean = np.zeros(Nq)
    out_std  = np.zeros(Nq)

    pos  = np.random.randn(N) * 0.3            # initial TVT delta spread
    rate = np.random.randn(N) * 0.01           # initial velocity
    wts  = np.ones(N) / N

    for t in range(Nq):
        dmd = q_dmd[t]
        # motion update
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j]  = pos[j] + rate[j] * dmd + PN * np.random.randn()
        # likelihood update from GR observation
        obs = q_gr[t]
        wsum = 0.0
        for j in range(N):
            tvt_abs = last_tvt + pos[j]
            gr_exp = _interp1(tw_grid, tvt_abs, vmin, step)
            err = (obs - gr_exp) / GR_SIG
            wts[j] = wts[j] * np.exp(-0.5 * err * err) + 1e-12
            wsum += wts[j]
        # normalize
        for j in range(N):
            wts[j] /= wsum
        # weighted estimate
        m = 0.0
        for j in range(N):
            m += wts[j] * pos[j]
        out_mean[t] = m
        v = 0.0
        for j in range(N):
            v += wts[j] * (pos[j] - m) ** 2
        out_std[t] = np.sqrt(v)
        # resample if effective sample size too low
        neff = 0.0
        for j in range(N):
            neff += wts[j] * wts[j]
        neff = 1.0 / neff
        if neff < RESAMP * N:
            # systematic resampling
            cum = np.cumsum(wts)
            u0 = np.random.random() / N
            new_pos  = np.zeros(N)
            new_rate = np.zeros(N)
            k = 0
            for j in range(N):
                u = u0 + j / N
                while k < N - 1 and u > cum[k]:
                    k += 1
                new_pos[j]  = pos[k]
                new_rate[j] = rate[k]
            for j in range(N):
                pos[j]  = new_pos[j]
                rate[j] = new_rate[j]
                wts[j]  = 1.0 / N
    return out_mean, out_std


def particle_filter(q_gr, q_dmd, tw_tvt, tw_gr, last_tvt, N=200, seed=42):
    """Wrapper: build uniform typewell grid, run PF."""
    if len(q_gr) == 0 or tw_tvt is None or len(tw_tvt) < 5:
        return np.zeros(len(q_gr)), np.zeros(len(q_gr))
    vmin, vmax = tw_tvt.min(), tw_tvt.max()
    step = 0.2
    n_grid = int((vmax - vmin) / step) + 2
    grid_tvt = vmin + np.arange(n_grid) * step
    tw_grid = np.interp(grid_tvt, tw_tvt, tw_gr).astype(np.float64)
    return _pf_run(
        q_gr.astype(np.float64), q_dmd.astype(np.float64),
        tw_grid, float(vmin), float(step), float(last_tvt),
        N, 0.995, 0.003, 0.01, 8.0, 0.5, seed
    )

# =============================================================================
# B. GR NaN INTERPOLATION (from typewell)
# =============================================================================
def interp_gr_nan(df, tw_tvt, tw_gr):
    """Fill NaN GR. In known zone use typewell @ TVT_input; elsewhere linear."""
    gr = df['GR'].values.astype(float)
    nan_mask = np.isnan(gr)
    if not nan_mask.any():
        return gr
    known = df['TVT_input'].notna().values
    # known zone: interp from typewell at TVT_input
    if tw_tvt is not None:
        fillable = nan_mask & known
        if fillable.any():
            gr[fillable] = np.interp(df['TVT_input'].values[fillable], tw_tvt, tw_gr)
    # remaining NaN: linear interpolation along the trace
    s = pd.Series(gr)
    gr = s.interpolate(method='linear', limit_direction='both').values
    gr = pd.Series(gr).ffill().bfill().values
    if np.isnan(gr).any():
        gr = np.nan_to_num(gr, nan=np.nanmean(tw_gr) if tw_gr is not None else 100.0)
    return gr

# =============================================================================
# C. DENSE FORMATIONS - SPATIAL KNN (leak-free via 5-fold leave-one-out)
# =============================================================================
# Each formation marker (ANCC, ASTNU, ...) is an areal surface ~ f(X, Y).
# We fit a spatial KNN  (X,Y) -> formation_depth  from training wells' samples.
# For a target well we predict the formation depth at each (X,Y), then anchor
# via the physical invariant  TVT + Z - formation = b_well (const per well):
#     b_well = median over known zone of (TVT_input + Z - formation_pred)
#     TVT_est = b_well - Z + formation_pred
# CRITICAL: for TRAIN wells we must use a KNN that EXCLUDES the well itself,
# otherwise the well's own samples leak its target. We do this with 5 fold
# models (each fit on 4/5 of wells) + 1 full model (for test).
class DenseFormations:
    def __init__(self, k=20, max_per_well=100):
        self.k = k
        self.max_per_well = max_per_well
        self.knn = {}
        self.fitted = False

    def fit(self, well_list):
        samp = {f: ([], []) for f in FORMATIONS}
        for wd in well_list:
            hw = wd['hw']
            if hw is None:
                continue
            for f in FORMATIONS:
                if f not in hw.columns:
                    continue
                sub = hw.dropna(subset=[f, 'X', 'Y', 'Z'])
                if len(sub) == 0:
                    continue
                if len(sub) > self.max_per_well:
                    sub = sub.iloc[np.linspace(0, len(sub)-1, self.max_per_well).astype(int)]
                samp[f][0].append(sub[['X', 'Y']].values)
                samp[f][1].append(sub[f].values)   # areal formation depth
        for f in FORMATIONS:
            if not samp[f][0]:
                continue
            Xf = np.vstack(samp[f][0]); yf = np.concatenate(samp[f][1])
            self.knn[f] = KNeighborsRegressor(n_neighbors=min(self.k, len(Xf)), weights='distance').fit(Xf, yf)
        self.fitted = len(self.knn) > 0
        return self

    def predict(self, X, Y):
        """dict: formation -> predicted areal depth at each (X,Y)."""
        out = {}
        if not self.fitted:
            return out
        XY = np.column_stack([X, Y])
        for f, m in self.knn.items():
            out[f] = m.predict(XY)
        return out

# =============================================================================
# Multi-Scale NCC - VECTORIZED (numpy sliding windows + matrix multiply)
# ~1000x faster than the double-loop version, same template-matching semantics.
# =============================================================================
def ncc_match(query_gr, ref_gr, ref_tvt, half_wins=(8, 15, 25)):
    N, M = len(query_gr), len(ref_gr)
    if M < 10 or N == 0:
        return np.zeros(N), np.zeros(N)
    ref_tvt_delta = (ref_tvt - ref_tvt[-1]).astype(np.float64)
    q = np.nan_to_num(query_gr.astype(np.float64), nan=np.nanmean(query_gr) if np.isfinite(query_gr).any() else 0.0)
    r = np.nan_to_num(ref_gr.astype(np.float64), nan=np.nanmean(ref_gr) if np.isfinite(ref_gr).any() else 0.0)
    scores_all, tvts_all = [], []
    for hw in half_wins:
        w = 2 * hw + 1
        if M < w or N < 1:
            scores_all.append(np.zeros(N)); tvts_all.append(np.zeros(N)); continue
        R = M - w + 1
        # reference windows (R, w)
        ref_win = np.lib.stride_tricks.sliding_window_view(r, w)
        ref_c   = ref_win - ref_win.mean(axis=1, keepdims=True)
        ref_norm = np.sqrt((ref_c ** 2).sum(axis=1)) + 1e-9            # (R,)
        ref_center_tvt = ref_tvt_delta[hw:hw + R]                       # (R,) center of each window
        # query windows (centered via edge padding) -> (N, w)
        q_pad = np.pad(q, (hw, hw), mode='edge')
        q_win = np.lib.stride_tricks.sliding_window_view(q_pad, w)[:N]
        q_c   = q_win - q_win.mean(axis=1, keepdims=True)
        q_norm = np.sqrt((q_c ** 2).sum(axis=1)) + 1e-9                 # (N,)
        # correlation matrix (N, R) = (N,w) @ (w,R), normalized
        corr = (q_c @ ref_c.T) / (q_norm[:, None] * ref_norm[None, :])
        best = corr.argmax(axis=1)                                      # (N,)
        sc   = corr[np.arange(N), best]
        tv   = ref_center_tvt[best]
        scores_all.append(np.maximum(sc, 0.0)); tvts_all.append(tv)
    scores_arr = np.stack(scores_all); tvts_arr = np.stack(tvts_all)
    exp_sc = np.exp(3.0 * scores_arr)
    weights = exp_sc / (exp_sc.sum(axis=0, keepdims=True) + 1e-9)
    return (weights * tvts_arr).sum(axis=0), scores_arr.max(axis=0)

# =============================================================================
# Beam Search (from V7)
# =============================================================================
BEAM_CFGS = [
    dict(BS=10, mc=20.0, es=144.0, sm=2, tag='cons'),
    dict(BS=10, mc=8.0,  es=64.0,  sm=2, tag='loose'),
    dict(BS=8,  mc=35.0, es=220.0, sm=1, tag='vcons'),
    dict(BS=10, mc=14.0, es=90.0,  sm=5, tag='sm5'),
    dict(BS=20, mc=4.0,  es=36.0,  sm=3, tag='vloose'),
    dict(BS=12, mc=12.0, es=100.0, sm=3, tag='mid'),
    dict(BS=15, mc=25.0, es=180.0, sm=2, tag='stiff'),
]

@njit(cache=True, fastmath=True)
def _beam_run(q_gr, ref_sm, ref_tvt, BS, mc, es, start, anchor):
    """Array-based beam search (numba-friendly, no dict). Returns TVT delta from anchor."""
    N = len(q_gr); M = len(ref_sm)
    beam_idx  = np.full(BS, start, dtype=np.int64)
    beam_cost = np.zeros(BS, dtype=np.float64)
    nb = 1
    out = np.zeros(N)
    deltas = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    for t in range(N):
        nc = nb * 5
        cand_idx  = np.empty(nc, dtype=np.int64)
        cand_cost = np.empty(nc, dtype=np.float64)
        c = 0
        for b in range(nb):
            for dd in range(5):
                ni = beam_idx[b] + deltas[dd]
                if ni < 0: ni = 0
                if ni > M - 1: ni = M - 1
                diff = q_gr[t] - ref_sm[ni]
                cand_idx[c]  = ni
                cand_cost[c] = beam_cost[b] + diff * diff / es + (deltas[dd] if deltas[dd] >= 0 else -deltas[dd]) * mc / 100.0
                c += 1
        order = np.argsort(cand_cost)
        new_nb = BS if BS < nc else nc
        mincost = cand_cost[order[0]]
        for b in range(new_nb):
            beam_idx[b]  = cand_idx[order[b]]
            beam_cost[b] = cand_cost[order[b]] - mincost   # renormalize to avoid overflow
        nb = new_nb
        out[t] = ref_tvt[beam_idx[0]] - anchor
    return out

def beam_search(q_gr, ref_gr, ref_tvt, cfg, last_tvt):
    """ref_gr/ref_tvt = typewell GR/TVT. Beam starts at typewell index nearest last_tvt,
    output = typewell_TVT[best] - last_tvt (delta from the well's last known anchor)."""
    BS, mc, es, sm = cfg['BS'], cfg['mc'], cfg['es'], cfg['sm']
    N, M = len(q_gr), len(ref_gr)
    if M < 5 or N == 0:
        return np.zeros(N)
    ref_sm = pd.Series(ref_gr).rolling(sm, center=True, min_periods=1).mean().values if sm > 1 else ref_gr.copy()
    start = int(np.clip(np.argmin(np.abs(ref_tvt - last_tvt)), 0, M - 1))
    try:
        return _beam_run(q_gr.astype(np.float64), ref_sm.astype(np.float64),
                         ref_tvt.astype(np.float64), int(BS), float(mc), float(es) + 1e-9,
                         start, float(last_tvt))
    except Exception:
        beam = [(start, 0.0)]; out = np.zeros(N)
        for i in range(N):
            nbd = {}
            for idx, cost in beam:
                for d in [-2,-1,0,1,2]:
                    ni = int(np.clip(idx+d, 0, M-1))
                    c = cost + (q_gr[i]-ref_sm[ni])**2/(es+1e-9) + abs(d)*mc/100.0
                    if ni not in nbd or nbd[ni] > c:
                        nbd[ni] = c
            srt = sorted(nbd.items(), key=lambda x: x[1])[:BS]
            beam = [(k, v) for k, v in srt]
            out[i] = ref_tvt[beam[0][0]] - last_tvt
        return out

# =============================================================================
# Formation Datum KNN (from V7)
# =============================================================================
class FormationKNN:
    def __init__(self, k=10):
        self.k = k; self.knn = {}; self.fitted = False
    def fit(self, well_list):
        rows = []
        for wd in well_list:
            hw = wd['hw']
            if not all(f in hw.columns for f in FORMATIONS):
                continue
            kn = hw[hw['TVT_input'].notna()]
            if len(kn) < 5:
                continue
            row = {'cx': hw['X'].mean(), 'cy': hw['Y'].mean()}
            for f in FORMATIONS:
                b = kn['TVT'] + kn['Z'] - kn[f]; n = len(b)
                row[f'bf_{f}'] = b.median()
                row[f'be_{f}'] = b.iloc[:n//3].median() if n>3 else b.median()
                row[f'bm_{f}'] = b.iloc[n//3:2*n//3].median() if n>3 else b.median()
                row[f'bl_{f}'] = b.iloc[-n//3:].median() if n>3 else b.median()
                w = np.exp(0.02*np.arange(n))
                row[f'bw_{f}'] = np.dot(w, b.values)/(w.sum()+1e-9)
            rows.append(row)
        if not rows:
            return self
        df = pd.DataFrame(rows); XY = df[['cx','cy']].values
        for f in FORMATIONS:
            cols = [c for c in df.columns if f in c]
            Y = df[cols].fillna(df[cols].median()).values
            self.knn[f] = KNeighborsRegressor(n_neighbors=min(self.k,len(df)), weights='distance')
            self.knn[f].fit(XY, Y)
        self.fitted = True
        return self
    def predict(self, cx, cy, Z, form_vals):
        out = {}
        if not self.fitted:
            return out
        q = np.array([[cx, cy]])
        for f in FORMATIONS:
            if f not in self.knn:
                continue
            b_pred = self.knn[f].predict(q)[0]
            for seg, bp in zip(['f','e','m','l','w'], b_pred):
                if f in form_vals:
                    out[f'tvtF_{f}_{seg}'] = bp + form_vals[f] - Z
                else:
                    out[f'tvtF_{f}_{seg}'] = np.full(len(Z), np.nan)
        return out

# -- Typewell diff offsets --
ANCH_OFFS = [-80,-40,-20,-10,-5, 0, 5,10,20,40, 80]
BEAM_OFFS = [-40,-20,-10,-5, -3, 0, 3, 5,10,20, 40]
SC_OFFS   = [-30,-15,-8, -4, -2, 0, 2, 4, 8,15, 30]

# =============================================================================
# Feature engineering per well
# =============================================================================
def engineer(hw, tw_tvt, tw_gr, name, fknn=None, dense=None):
    df = hw.copy().reset_index(drop=True)
    df['well'] = name; df['row_idx'] = np.arange(len(df))
    known_mask = df['TVT_input'].notna(); eval_mask = ~known_mask
    n_kn = known_mask.sum()
    last_tvt = df.loc[known_mask,'TVT_input'].iloc[-1] if n_kn>0 else 0.0
    df['last_tvt'] = last_tvt
    df['tvt_ffill'] = df['TVT_input'].ffill().bfill()
    df['is_eval'] = eval_mask.astype(int)

    eval_start = eval_mask.idxmax() if eval_mask.any() else len(df)
    df['md_since'] = np.maximum(df['MD'] - (df.loc[known_mask,'MD'].iloc[-1] if n_kn>0 else 0.0), 0)
    df['rows_into'] = np.maximum(df.index - eval_start + 1, 0)
    df['frac'] = df['rows_into']/(eval_mask.sum()+1)
    df['sqrt_frac'] = np.sqrt(df['frac'])
    df['known_len'] = n_kn; df['eval_len'] = eval_mask.sum()

    # === B. GR NaN interpolation FIRST ===
    gr = interp_gr_nan(df, tw_tvt, tw_gr)
    df['GR'] = gr

    # GR features
    for w in [5,21,51,101]:
        s = pd.Series(gr)
        df[f'grm{w}'] = s.rolling(w,center=True,min_periods=1).mean().values
        df[f'grs{w}'] = s.rolling(w,center=True,min_periods=1).std().fillna(0).values
    for lag in [1,5,15,30]:
        df[f'glag{lag}'] = pd.Series(gr).shift(lag).bfill().values
        df[f'glead{lag}'] = pd.Series(gr).shift(-lag).ffill().values
    df['gr_d1'] = np.gradient(gr); df['gr_d2'] = np.gradient(df['gr_d1'].values)
    df['gr_env'] = pd.Series(gr).rolling(21,center=True,min_periods=1).max().values
    df['gr_nrg'] = pd.Series(gr**2).rolling(21,center=True,min_periods=1).mean().apply(np.sqrt).values
    if n_kn >= 2:
        sg = np.polyfit(df.loc[known_mask,'MD'].values, df.loc[known_mask,'GR'].values, 1)
        df['gr_detr'] = gr - np.polyval(sg, df['MD'].values)
    else:
        df['gr_detr'] = gr - gr.mean()

    # Spatial
    for c in ['X','Y','Z']:
        df[f'd{c}'] = df[c].diff().fillna(0)
    df['dH'] = np.sqrt(df['dX']**2 + df['dY']**2)
    df['MD_norm'] = (df['MD']-df['MD'].min())/(df['MD'].max()-df['MD'].min()+1e-9)
    if n_kn > 0:
        ax, ay, az = df.loc[known_mask,['X','Y','Z']].iloc[-1]
        df['dx_anch'] = df['X']-ax; df['dy_anch'] = df['Y']-ay; df['dz_anch'] = df['Z']-az
        df['dxy_anch'] = np.sqrt(df['dx_anch']**2 + df['dy_anch']**2)

    # === C. Dense Formations spatial datum (LEAK-FREE) ===
    # `dense` must be a KNN that EXCLUDES this well (5-fold LOO for train, full for test).
    if dense is not None and dense.fitted:
        Zv = df['Z'].values
        fpred = dense.predict(df['X'].values, df['Y'].values)  # f -> areal formation depth
        delta_cols = []
        for f, fdep in fpred.items():
            # physical invariant: TVT + Z - formation = b_well (const per well)
            if n_kn > 0:
                b_known = (df.loc[known_mask,'TVT_input'].values
                           + Zv[known_mask.values] - fdep[known_mask.values])
                b_const = np.median(b_known)
            else:
                b_const = last_tvt
            tvt_f = b_const - Zv + fdep        # TVT = b - Z + formation
            df[f'tvtf_{f}'] = tvt_f
            df[f'tvtf_{f}_delta'] = tvt_f - last_tvt
            delta_cols.append(f'tvtf_{f}_delta')
        if delta_cols:
            df['form_mean'] = df[delta_cols].mean(axis=1)
            df['form_std']  = df[delta_cols].std(axis=1).fillna(0)
            df['form_med']  = df[delta_cols].median(axis=1)

    # Typewell GR interp
    if tw_tvt is not None and n_kn > 0:
        df.loc[known_mask,'tw_GR'] = np.interp(df.loc[known_mask,'TVT_input'].values, tw_tvt, tw_gr)
    df['tw_GR'] = df['tw_GR'].ffill().bfill() if 'tw_GR' in df.columns else df['GR']
    df['gr_vs_tw'] = df['GR'] - df['tw_GR']

    # Slope extrapolation
    if n_kn >= 2:
        k_md = df.loc[known_mask,'MD'].values
        k_tvt = df.loc[known_mask,'TVT_input'].values
        k_z = df.loc[known_mask,'Z'].values
        slp_all = np.polyfit(k_md, k_tvt, 1)[0]
        tail = min(50, n_kn)
        slp_50 = np.polyfit(k_md[-tail:], k_tvt[-tail:], 1)[0]
        slp_z = np.polyfit(k_z, k_tvt, 1)[0] if k_z.std()>1e-6 else 0.0
        df['slp_all'] = slp_all; df['slp_50'] = slp_50; df['slp_z'] = slp_z
        df['tvt_proj_all'] = slp_all*df['md_since']; df['tvt_proj_50'] = slp_50*df['md_since']
        df['tvt_proj_z'] = slp_z*df['dz_anch'] if 'dz_anch' in df.columns else 0.0
        df['ktvt_range'] = k_tvt.max()-k_tvt.min(); df['ktvt_std'] = k_tvt.std()
        df['pfx_gr_slope'] = np.polyfit(k_md, df.loc[known_mask,'GR'].values, 1)[0]
    for c in ['slp_all','slp_50','slp_z','tvt_proj_all','tvt_proj_50','tvt_proj_z',
              'ktvt_range','ktvt_std','pfx_gr_slope']:
        if c not in df.columns: df[c] = 0.0

    # (Formation features now handled leak-free by DenseFormations above.)

    # === A. Particle Filter ===
    if eval_mask.any() and tw_tvt is not None and n_kn >= 5:
        q_gr = df.loc[eval_mask,'GR'].values
        q_dmd = df.loc[eval_mask,'MD'].diff().fillna(1.0).values
        pf_m, pf_s = particle_filter(q_gr, q_dmd, tw_tvt, tw_gr, last_tvt, N=200, seed=42)
        df.loc[eval_mask,'pf_delta'] = pf_m
        df.loc[eval_mask,'pf_std'] = pf_s
    df['pf_delta'] = df.get('pf_delta', pd.Series(0.0,index=df.index)).fillna(0)
    df['pf_std'] = df.get('pf_std', pd.Series(0.0,index=df.index)).fillna(0)

    # NCC
    if eval_mask.any() and n_kn >= 20:
        ref_gr = df.loc[known_mask,'GR'].values
        ref_tvt_kn = df.loc[known_mask,'TVT_input'].values
        ncc_d, ncc_sc = ncc_match(df.loc[eval_mask,'GR'].values, ref_gr, ref_tvt_kn, (8,15,25))
        df.loc[eval_mask,'ncc_delta'] = ncc_d; df.loc[eval_mask,'ncc_score'] = ncc_sc
        df['sc_trust'] = np.clip(n_kn/200.0, 0, 0.6)
    df['ncc_delta'] = df.get('ncc_delta', pd.Series(0.0,index=df.index)).fillna(0)
    df['ncc_score'] = df.get('ncc_score', pd.Series(0.0,index=df.index)).fillna(0)
    df['sc_trust'] = df.get('sc_trust', pd.Series(0.0,index=df.index)).fillna(0)

    # Beam
    if eval_mask.any() and tw_tvt is not None and n_kn >= 5:
        q_gr = df.loc[eval_mask,'GR'].values
        for cfg in BEAM_CFGS:
            df.loc[eval_mask, f'beam_{cfg["tag"]}'] = beam_search(q_gr, tw_gr, tw_tvt, cfg, last_tvt)
    for cfg in BEAM_CFGS:
        c = f'beam_{cfg["tag"]}'
        df[c] = df.get(c, pd.Series(0.0,index=df.index)).fillna(0)
    bc = [f'beam_{c["tag"]}' for c in BEAM_CFGS]
    df['beam_mean'] = df[bc].mean(axis=1); df['beam_std'] = df[bc].std(axis=1).fillna(0)
    df['beam_med'] = df[bc].median(axis=1)
    df['hyb_delta'] = (1-df['sc_trust'])*df['beam_mean'] + df['sc_trust']*df['ncc_delta']

    # Cross-signal agreement
    df['pf_vs_ncc'] = df['pf_delta'] - df['ncc_delta']
    df['pf_vs_beam'] = df['pf_delta'] - df['beam_mean']
    df['ncc_vs_beam'] = df['ncc_delta'] - df['beam_mean']
    df['pf_vs_form'] = df['pf_delta'] - df.get('form_mean', 0.0)
    sig_cols = ['pf_delta','ncc_delta','beam_mean','hyb_delta']
    if 'form_mean' in df.columns:
        sig_cols.append('form_mean')
    df['signal_mean'] = df[sig_cols].mean(axis=1)
    df['signal_std'] = df[sig_cols].std(axis=1).fillna(0)

    # Typewell diff offsets
    if tw_tvt is not None and n_kn > 0:
        tvt_a = df['tvt_ffill'].values
        for o in ANCH_OFFS:
            df[f'tda{o:+d}'] = gr - np.interp(tvt_a+o, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
        beam_ref = (df['beam_cons'].values + df['beam_sm5'].values)/2.0 if 'beam_cons' in df.columns else np.zeros(len(df))
        for o in BEAM_OFFS:
            df[f'tdbc{o:+d}'] = gr - np.interp(tvt_a+beam_ref+o, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
        for o in SC_OFFS:
            df[f'tdsc{o:+d}'] = gr - np.interp(tvt_a+df['ncc_delta'].values+o, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])

    return df

# =============================================================================
# MAIN
# =============================================================================
def load_tw(path):
    if not os.path.exists(path):
        return None, None
    t = pd.read_csv(path).dropna(subset=['TVT','GR']).sort_values('TVT')
    return t['TVT'].values, t['GR'].values

print("Loading train wells...")
hw_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*__horizontal_well.csv")))
print(f"  {len(hw_files)} wells")
train_wds = []
for path in hw_files:
    nm = os.path.basename(path).replace("__horizontal_well.csv","")
    hw = pd.read_csv(path)
    tw_tvt, tw_gr = load_tw(path.replace("__horizontal_well.csv","__typewell.csv"))
    train_wds.append({'name':nm,'hw':hw,'tw_tvt':tw_tvt,'tw_gr':tw_gr})

print("Fitting DenseFormations (5-fold leave-one-out + full, leak-free)...")
N_FOLD = 5
well_fold = np.array([i % N_FOLD for i in range(len(train_wds))])
dense_loo = []
for f in range(N_FOLD):
    sub = [train_wds[i] for i in range(len(train_wds)) if well_fold[i] != f]
    dense_loo.append(DenseFormations(k=20, max_per_well=100).fit(sub))
dense_full = DenseFormations(k=20, max_per_well=100).fit(train_wds)
print(f"  LOO models: {[d.fitted for d in dense_loo]}, full: {dense_full.fitted}")

# Warm up Numba JIT once
if HAS_NUMBA:
    print("Warming up Numba JIT...")
    _ = particle_filter(np.random.randn(10).astype(float), np.ones(10),
                        np.linspace(0,100,50), np.random.randn(50), 50.0, N=50, seed=0)
    print("  JIT compiled.")

import gc
TARGET = 'tvt_delta'
DROP = ['well','row_idx','TVT_input','TVT','GR','tw_GR',
        'ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA',
        'last_tvt','tvt_ffill','is_eval',TARGET]

print("Engineering train features (eval-rows only, float32 to save RAM)...")
train_frames = []
for i, wd in enumerate(train_wds):
    # use the LOO formation model that EXCLUDES this well's fold (no target leak)
    df = engineer(wd['hw'], wd['tw_tvt'], wd['tw_gr'], wd['name'], None, dense_loo[well_fold[i]])
    # Keep only eval rows (what we train on) -> drops ~25% of rows immediately
    df = df[df['is_eval'] == 1].copy()
    if len(df):
        df[TARGET] = (df['TVT'] - df['last_tvt']).astype('float32')
        # keep group + md_since + features + target; downcast floats
        df['md_since'] = df['md_since'].astype('float32')
        keep = ['well', TARGET, 'md_since'] + \
               [c for c in df.columns if c not in DROP and c not in ('md_since',) and df[c].dtype != object]
        df = df[list(dict.fromkeys(keep))]
        for c in df.select_dtypes(include=['float64']).columns:
            df[c] = df[c].astype('float32')
        train_frames.append(df)
    # free the raw well immediately
    train_wds[i]['hw'] = None
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(train_wds)}")

train_eval = pd.concat(train_frames, ignore_index=True)
del train_frames; gc.collect()
print(f"  Train eval rows: {len(train_eval)}, mem={train_eval.memory_usage(deep=True).sum()/1e9:.2f}GB")

feature_cols = [c for c in train_eval.columns if c not in DROP and c != 'md_since' and train_eval[c].dtype != object]
print(f"  Features: {len(feature_cols)}")

X = train_eval[feature_cols].fillna(0)
y = train_eval[TARGET].astype('float32')
groups = train_eval['well']
md_since_oof = train_eval['md_since'].values
del train_eval; gc.collect()

# === D. Train LGB x3 + XGB + CatBoost (GPU with CPU fallback) ===
# Detect GPU availability
USE_GPU = False
try:
    import subprocess
    r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=10)
    USE_GPU = (r.returncode == 0)
except Exception:
    USE_GPU = False
print(f"GPU available: {USE_GPU}")

gkf = GroupKFold(n_splits=5)
model_specs = []
LGB_BASE = dict(objective='regression', metric='rmse', num_leaves=255, min_child_samples=15,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05,
                n_estimators=3000, verbosity=-1, n_jobs=-1)
for si, (seed, lr) in enumerate(zip([42,7,123],[0.025,0.020,0.030])):
    model_specs.append(('lgb', seed, {**LGB_BASE,'random_state':seed,'learning_rate':lr}))
if HAS_XGB:
    model_specs.append(('xgb', 99, dict(n_estimators=3000, learning_rate=0.025, max_depth=8,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                        n_jobs=-1, random_state=99)))
if HAS_CB:
    model_specs.append(('cb', 77, dict(iterations=3000, learning_rate=0.03, depth=8,
                        l2_leaf_reg=3.0, loss_function='RMSE', random_seed=77, verbose=0)))

def fit_lgb(params, Xtr, ytr, Xva, yva, gpu):
    p = dict(params)
    if gpu:
        p['device_type'] = 'gpu'
    m = lgb.LGBMRegressor(**p)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    return m

def fit_xgb(params, Xtr, ytr, Xva, yva, gpu):
    p = dict(params)
    p['tree_method'] = 'hist'
    if gpu:
        p['device'] = 'cuda'
    m = xgb.XGBRegressor(**p, early_stopping_rounds=150)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m

def fit_cb(params, Xtr, ytr, Xva, yva, gpu):
    p = dict(params)
    if gpu:
        p['task_type'] = 'GPU'; p['devices'] = '0'
    m = CatBoostRegressor(**p, early_stopping_rounds=150)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    return m

FITTERS = {'lgb': fit_lgb, 'xgb': fit_xgb, 'cb': fit_cb}

print(f"Training {len(model_specs)} models (GPU={USE_GPU})...")
all_oof = np.zeros((len(X), len(model_specs)))
all_models = []
for mi, (mtype, seed, params) in enumerate(model_specs):
    oof = np.zeros(len(X)); mdls = []
    fitter = FITTERS[mtype]
    for fold, (tr_i, va_i) in enumerate(gkf.split(X, y, groups)):
        Xtr, ytr = X.iloc[tr_i], y.iloc[tr_i]
        Xva, yva = X.iloc[va_i], y.iloc[va_i]
        try:
            m = fitter(params, Xtr, ytr, Xva, yva, USE_GPU)
        except Exception as e:
            print(f"    [GPU fit failed for {mtype}, fallback CPU] {str(e)[:80]}")
            m = fitter(params, Xtr, ytr, Xva, yva, False)
        oof[va_i] = m.predict(Xva); mdls.append(m)
    rmse = np.sqrt(np.mean((y-oof)**2))
    print(f"  [{mtype} seed={seed}] OOF RMSE: {rmse:.4f}")
    all_oof[:,mi] = oof; all_models.append((mtype, mdls))

# Hill climbing over all models
print("Hill climbing blend...")
best_w, best_rmse = np.ones(len(model_specs))/len(model_specs), 1e9
rng = np.random.default_rng(0)
for _ in range(5000):
    w = rng.dirichlet(np.ones(len(model_specs)))
    r = np.sqrt(np.mean((y-(all_oof*w).sum(1))**2))
    if r < best_rmse:
        best_rmse, best_w = r, w
print(f"  Blend RMSE: {best_rmse:.4f}, weights: {best_w.round(3)}")

# Inference
print("Inferencing test wells...")
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub[['wellname','ridx']] = sample_sub['id'].str.rsplit('_',n=1,expand=True)
sample_sub['ridx'] = sample_sub['ridx'].astype(int)
test_frames = []
for path in sorted(glob.glob(os.path.join(TEST_DIR,"*__horizontal_well.csv"))):
    nm = os.path.basename(path).replace("__horizontal_well.csv","")
    hw = pd.read_csv(path)
    tw_tvt, tw_gr = load_tw(path.replace("__horizontal_well.csv","__typewell.csv"))
    df = engineer(hw, tw_tvt, tw_gr, nm, None, dense_full)
    sub_ridx = sample_sub.loc[sample_sub['wellname']==nm,'ridx'].values
    test_frames.append(df[df['row_idx'].isin(sub_ridx)].copy())
test_df = pd.concat(test_frames, ignore_index=True)
print(f"  Test rows: {len(test_df)}")
for c in feature_cols:
    if c not in test_df.columns: test_df[c] = 0.0
X_test = test_df[feature_cols].fillna(0)

preds_per_model = []
for mtype, mdls in all_models:
    preds_per_model.append(np.mean([m.predict(X_test) for m in mdls], axis=0))
test_delta = (np.column_stack(preds_per_model)*best_w).sum(axis=1)

# Post-processing: alpha x tau grid on OOF
hc_oof = (all_oof*best_w).sum(1)
best_pp, best_pp_rmse = {'alpha':1.0,'tau':None}, 1e9
for alpha in np.arange(0.70, 1.01, 0.05):
    for tau in [None,25,50,100,200]:
        d = hc_oof.copy()
        if tau is not None:
            d = d*(1.0-np.exp(-np.maximum(md_since_oof,0)/tau))
        d = d*alpha
        r = np.sqrt(np.mean((y-d)**2))
        if r < best_pp_rmse:
            best_pp_rmse, best_pp = r, {'alpha':alpha,'tau':tau}
print(f"  Best PP: {best_pp}, RMSE={best_pp_rmse:.4f}")

alpha, tau = best_pp['alpha'], best_pp['tau']
if tau is not None:
    test_delta_pp = test_delta*(1.0-np.exp(-np.maximum(test_df['md_since'].values,0)/tau))*alpha
else:
    test_delta_pp = test_delta*alpha
test_df['pred_tvt_pp'] = test_df['last_tvt'].values + test_delta_pp

# Stratigraphic projection + Savitzky-Golay
for nm in test_df['well'].unique():
    mask = (test_df['well']==nm).values
    sub = test_df[mask]
    md_vals = sub['MD'].values; tvt_pred = sub['pred_tvt_pp'].values; z_vals = sub['Z'].values
    if len(md_vals) >= 5:
        try:
            mn = (md_vals-md_vals.min())/(md_vals.max()-md_vals.min()+1e-9)
            U = np.polyval(np.polyfit(mn, tvt_pred+z_vals, min(4,len(md_vals)-1)), mn)
            tvt_pred = 0.5*tvt_pred + 0.5*(U - z_vals)
        except Exception:
            pass
        wl = min(17, len(tvt_pred) if len(tvt_pred)%2==1 else len(tvt_pred)-1); wl = max(wl,3)
        if wl >= 5:
            try:
                tvt_pred = savgol_filter(tvt_pred, wl, 3)
            except Exception:
                pass
        test_df.loc[mask, 'pred_tvt_pp'] = tvt_pred

test_df['id'] = test_df['well'] + '_' + test_df['row_idx'].astype(str)
pred_map = dict(zip(test_df['id'], test_df['pred_tvt_pp']))
sample_sub['tvt'] = sample_sub['id'].map(pred_map)
miss = sample_sub['tvt'].isna().sum()
if miss:
    print(f"WARNING: {miss} missing, ffill")
    sample_sub['tvt'] = sample_sub['tvt'].ffill().fillna(0)
submission = sample_sub[['id','tvt']]
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission: {len(submission)} rows")
print(submission.head(10))
print(f"TVT: mean={submission['tvt'].mean():.2f} std={submission['tvt'].std():.2f}")
