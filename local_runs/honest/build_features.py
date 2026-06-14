"""Honest feature builder mirroring the test task.

For each well we predict TVT on the 'unknown' lateral rows (TVT_input is NaN).
Only test-available columns are used: MD,X,Y,Z,GR,TVT_input + typewell GR/Geology.
Target = TVT - last_known_TVT (delta).
"""
import pandas as pd, numpy as np, glob, os
from scipy.signal import savgol_filter

D = "data/rogii-wellbore-geology-prediction"

def smooth_gr(gr):
    g = pd.Series(gr).interpolate(limit_direction="both").bfill().ffill().values
    if np.all(~np.isfinite(g)):
        g = np.zeros_like(g)
    g = np.nan_to_num(g, nan=np.nanmedian(g) if np.isfinite(np.nanmedian(g)) else 0.0)
    return g

def load_typewell(prefix, split):
    f = f"{D}/{split}/{prefix}__typewell.csv"
    if not os.path.exists(f):
        return None
    tw = pd.read_csv(f)
    return tw

def build_well(hw_path, split):
    prefix = os.path.basename(hw_path).split("__")[0]
    df = pd.read_csv(hw_path)
    df = df.sort_values("MD").reset_index(drop=True)
    known = df["TVT_input"].notna().values
    n = len(df)
    # last known anchor
    kidx = np.where(known)[0]
    if len(kidx) == 0:
        return None
    last_k = kidx[-1]
    lastTVT = df["TVT_input"].values[last_k]
    lastMD = df["MD"].values[last_k]
    lastZ = df["Z"].values[last_k]
    lastX = df["X"].values[last_k]
    lastY = df["Y"].values[last_k]

    Z = df["Z"].values.astype(float)
    MD = df["MD"].values.astype(float)
    X = df["X"].values.astype(float); Y = df["Y"].values.astype(float)
    grs = smooth_gr(df["GR"].values.astype(float))

    # slope of TVT_input wrt Z and MD on the known heel (apparent dip near anchor)
    kk = known
    tvtk = df["TVT_input"].values[kk]
    Zk = Z[kk]; MDk = MD[kk]
    # use last portion of heel for local slope
    m = min(300, kk.sum())
    sl = slice(kk.sum()-m, kk.sum())
    def safe_slope(x, y):
        if len(x) < 3 or np.ptp(x) < 1e-6:
            return 0.0
        return np.polyfit(x, y, 1)[0]
    slope_Z = safe_slope(Zk[sl], tvtk[sl])
    slope_MD = safe_slope(MDk[sl], tvtk[sl])
    slope_Z_full = safe_slope(Zk, tvtk)
    slope_MD_full = safe_slope(MDk, tvtk)

    # rolling GR stats
    grS = pd.Series(grs)
    feat = pd.DataFrame({
        "well": prefix,
        "MD": MD, "Z": Z, "X": X, "Y": Y,
        "GR": grs,
        "dMD": MD - lastMD,
        "dZ": Z - lastZ,
        "dX": X - lastX, "dY": Y - lastY,
        "dXY": np.hypot(X-lastX, Y-lastY),
        "lastTVT": lastTVT, "lastZ": lastZ, "lastMD": lastMD,
        "slope_Z": slope_Z, "slope_MD": slope_MD,
        "slope_Z_full": slope_Z_full, "slope_MD_full": slope_MD_full,
        "geom_pred_Z": lastTVT + slope_Z*(Z-lastZ),
        "geom_pred_MD": lastTVT + slope_MD*(MD-lastMD),
        "geom_pred_negZ": lastTVT - (Z-lastZ),
        "gr_roll_mean_25": grS.rolling(25,center=True,min_periods=1).mean().values,
        "gr_roll_std_25": grS.rolling(25,center=True,min_periods=1).std().fillna(0).values,
        "gr_roll_mean_101": grS.rolling(101,center=True,min_periods=1).mean().values,
        "gr_d1": np.gradient(savgol_filter(grs, min(51, n if n%2 else n-1) | 1, 2) if n>5 else grs),
        "known": known.astype(int),
        "n_known": kk.sum(),
        "frac_along": (np.arange(n) - last_k) / max(1, n - last_k),
    })
    if "TVT" in df.columns:
        feat["TVT"] = df["TVT"].values
        feat["delta"] = df["TVT"].values - lastTVT
    feat["row_in_well"] = np.arange(n)
    feat["is_target"] = (~known).astype(int)
    return feat

def build_split(split, limit=None):
    hws = sorted(glob.glob(f"{D}/{split}/*__horizontal_well.csv"))
    if limit: hws = hws[:limit]
    out = []
    for i, f in enumerate(hws):
        try:
            w = build_well(f, split)
            if w is not None:
                out.append(w)
        except Exception as e:
            print("ERR", f, e)
        if (i+1) % 100 == 0:
            print(f"  {split} {i+1}/{len(hws)}")
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    import sys
    tr = build_split("train")
    tr.to_parquet("local_runs/honest/train_feats.parquet")
    print("train feats", tr.shape, "target rows", tr.is_target.sum())
    te = build_split("test")
    te.to_parquet("local_runs/honest/test_feats.parquet")
    print("test feats", te.shape, "target rows", te.is_target.sum())
