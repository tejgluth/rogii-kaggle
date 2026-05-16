"""exp030: exp026 features + extended GR features.

Adds to exp026:
- tda{offset}: GR(row) - typewell_GR(at last_known_TVT + offset) for 11 offsets.
- gr lag/lead at 1/5/15/30 rows.
- gr_d1, gr_d2 diffs.
- gr_env (rolling-max envelope), gr_nrg (rolling-RMS energy).
- Multi-window rolling means/stds at w=5, 21, 51, 101.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
TDA_OFFS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out_tvt, out_score = [], []
    for hw in hws:
        win = 2 * hw + 1; nk = len(kgr); nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out_tvt.append(np.full(nh, ktvt[-1] if len(ktvt) else 0, np.float32))
            out_score.append(np.zeros(nh, np.float32)); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32); M = len(sts)
        if M == 0:
            out_tvt.append(np.full(nh, ktvt[-1] if len(ktvt) else 0, np.float32))
            out_score.append(np.zeros(nh, np.float32)); continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode='edge')
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out_tvt.append(ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32))
        out_score.append(score)
    return out_tvt, out_score


def build_well(hw_path, tw_path, is_train):
    wid = hw_path.stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path)
    if not tw_path.exists():
        return None
    tw = pd.read_csv(tw_path).sort_values("TVT")
    kn_mask = hw["TVT_input"].notna()
    if not kn_mask.any(): return None
    if is_train and "TVT" not in hw.columns: return None
    kn = hw[kn_mask]; ev = hw[~kn_mask]
    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    last_md = float(lk["MD"])
    last_x, last_y, last_z = float(lk["X"]), float(lk["Y"]), float(lk["Z"])
    last_gr = float(lk["GR"]) if pd.notna(lk["GR"]) else float(kn["GR"].mean())

    out = pd.DataFrame(index=hw.index)
    out["well_id"] = wid
    out["MD"] = hw["MD"]; out["X"] = hw["X"]; out["Y"] = hw["Y"]; out["Z"] = hw["Z"]
    out["GR"] = hw["GR"]; out["TVT_input"] = hw["TVT_input"]
    out["is_lateral"] = (~kn_mask).values
    out["last_known_tvt"] = last_tvt
    out["last_known_md"] = last_md
    out["last_known_gr"] = last_gr
    out["md_since_anchor"] = hw["MD"] - last_md
    out["dx_since_anchor"] = hw["X"] - last_x
    out["dy_since_anchor"] = hw["Y"] - last_y
    out["dz_since_anchor"] = hw["Z"] - last_z
    out["xy_dist"] = np.sqrt(out["dx_since_anchor"]**2 + out["dy_since_anchor"]**2)

    # GR rich features
    gr = pd.Series(hw["GR"]).interpolate(limit_direction="both").fillna(0).astype(np.float32)
    out["gr_filled"] = gr.values
    for w in [5, 21, 51, 101]:
        out[f"grm{w}"] = gr.rolling(w, center=True, min_periods=1).mean().values.astype(np.float32)
        out[f"grs{w}"] = gr.rolling(w, center=True, min_periods=1).std().fillna(0).values.astype(np.float32)
    out["gr_d1"] = gr.diff().fillna(0).values.astype(np.float32)
    out["gr_d2"] = gr.diff().diff().fillna(0).values.astype(np.float32)
    out["gr_env"] = gr.rolling(21, center=True, min_periods=1).max().values.astype(np.float32)
    out["gr_nrg"] = np.sqrt(np.maximum((gr**2).rolling(21, center=True, min_periods=1).mean().values, 0.)).astype(np.float32)
    for lag in [1, 5, 15, 30]:
        out[f"glag{lag}"] = gr.shift(lag).bfill().values.astype(np.float32)
        out[f"glead{lag}"] = gr.shift(-lag).ffill().values.astype(np.float32)

    # b_well anchoring per formation
    ktvt = kn["TVT_input"].values.astype(np.float64)
    kz = kn["Z"].values.astype(np.float64)
    z_all = hw["Z"].values.astype(np.float64)
    for fm in FORMATIONS:
        if fm in hw.columns and hw[fm].notna().any():
            f_all = hw[fm].values.astype(np.float64)
            f_kn = f_all[kn_mask.values]
            bv = ktvt + kz - f_kn
            b_full = float(np.median(bv)); n = len(bv)
            b_late = float(np.median(bv[-50:])) if n >= 50 else b_full
            w_ = np.exp(0.02 * np.arange(n)); w_ /= w_.sum()
            b_wls = float(np.dot(w_, bv))
            out[f"fm_{fm}_tvt"] = (-z_all + f_all + b_full).astype(np.float32)
            out[f"fm_{fm}_tvt_late"] = (-z_all + f_all + b_late).astype(np.float32)
            out[f"fm_{fm}_tvt_wls"] = (-z_all + f_all + b_wls).astype(np.float32)
            out[f"fm_{fm}_bwell"] = b_full
            out[f"fm_{fm}_bwell_late"] = b_late
            out[f"fm_{fm}_minus_z"] = (f_all - z_all).astype(np.float32)
        else:
            for c in [f"fm_{fm}_tvt", f"fm_{fm}_tvt_late", f"fm_{fm}_tvt_wls", f"fm_{fm}_minus_z"]:
                out[c] = 0.0
            for c in [f"fm_{fm}_bwell", f"fm_{fm}_bwell_late"]:
                out[c] = 0.0

    fm_preds = np.column_stack([out[f"fm_{fm}_tvt"].values for fm in FORMATIONS])
    out["fm_mean_tvt"] = fm_preds.mean(1)
    out["fm_std_tvt"] = fm_preds.std(1)
    out["fm_min_tvt"] = fm_preds.min(1)
    out["fm_max_tvt"] = fm_preds.max(1)

    # NCC
    tw_tvt = tw["TVT"].values.astype(np.float32)
    tw_gr = tw["GR"].values.astype(np.float32)
    kgr = kn["GR"].interpolate(limit_direction="both").fillna(0).values.astype(np.float32)
    hgr_all = gr.values.astype(np.float32)
    ncc_tvts, ncc_scores = multi_scale_ncc(kgr, ktvt.astype(np.float32), hgr_all, hws=(8, 15, 25))
    for i, hw_n in enumerate([8, 15, 25]):
        out[f"ncc{hw_n}_tvt"] = ncc_tvts[i]
        out[f"ncc{hw_n}_score"] = ncc_scores[i]
    tvts = np.stack(ncc_tvts, 1); scores = np.stack(ncc_scores, 1)
    sw = np.exp(3. * scores); sw /= sw.sum(1, keepdims=True) + 1e-9
    out["ncc_ens_tvt"] = (tvts * sw).sum(1).astype(np.float32)
    out["ncc_max_score"] = scores.max(1)

    # tda features: GR - typewell_GR(at last_known_tvt + offset)
    for o in TDA_OFFS:
        tw_at = np.interp(last_tvt + o, tw_tvt, tw_gr).astype(np.float32)
        out[f"tda{int(o)}"] = (hgr_all - tw_at).astype(np.float32)

    # tw_anc: typewell GR at the anchor (last_known)
    out["tw_gr_at_anchor"] = float(np.interp(last_tvt, tw_tvt, tw_gr))
    out["tw_gr_mean"] = float(tw_gr.mean())
    out["pfx_rmse"] = float(np.sqrt(np.mean((kgr - np.interp(ktvt, tw_tvt, tw_gr))**2)))

    # Slope features (TVT vs MD slope on heel)
    kmd = kn["MD"].values.astype(np.float64)
    if len(kmd) >= 2 and kmd.std() > 1e-6:
        slp_all = float(np.polyfit(kmd, ktvt, 1)[0])
        slp_50 = float(np.polyfit(kmd[-50:], ktvt[-50:], 1)[0]) if len(kmd) >= 50 else slp_all
    else:
        slp_all = slp_50 = 0.0
    out["slp_all"] = slp_all
    out["slp_50"] = slp_50
    out["slope_proj"] = (last_tvt + slp_all * out["md_since_anchor"]).astype(np.float32)
    out["slope50_proj"] = (last_tvt + slp_50 * out["md_since_anchor"]).astype(np.float32)

    if is_train:
        out["TVT"] = hw["TVT"]
        out["delta_target"] = hw["TVT"].values - last_tvt
    return out


def main():
    t0 = time.time()
    print("Loading train...", flush=True)
    files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    dfs = []
    for i, p in enumerate(files):
        tw_path = p.parent / (p.stem.replace("__horizontal_well", "__typewell") + ".csv")
        d = build_well(p, tw_path, is_train=True)
        if d is not None:
            dfs.append(d)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    train = pd.concat(dfs, ignore_index=True)
    print(f"train: {train.shape} ({time.time()-t0:.0f}s)")

    print("Loading test...", flush=True)
    tfiles = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    tdfs = [build_well(p, p.parent / (p.stem.replace("__horizontal_well", "__typewell") + ".csv"), False) for p in tfiles]
    test = pd.concat([t for t in tdfs if t is not None], ignore_index=True)
    print(f"test: {test.shape}")

    skip = {"well_id", "TVT", "delta_target", "is_lateral"}
    feat_cols = [c for c in train.columns if c not in skip]
    print(f"#features: {len(feat_cols)}")

    X = train[feat_cols].values.astype(np.float32)
    y_delta = train["delta_target"].values.astype(np.float32)
    y_abs = train["TVT"].values.astype(np.float32)
    last_known = train["last_known_tvt"].values.astype(np.float32)
    is_lateral = train["is_lateral"].values
    groups = train["well_id"].values
    Xt = test[feat_cols].values.astype(np.float32)
    test_last_known = test["last_known_tvt"].values.astype(np.float32)

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.025, num_leaves=127, min_data_in_leaf=30,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=2.0, verbose=-1, num_threads=-1, max_bin=255,
    )

    gkf = GroupKFold(5)
    oof_delta = np.zeros(len(y_delta), dtype=np.float32)
    tp_folds = []
    fold_rmses = []
    for fi, (tr, va) in enumerate(gkf.split(X, y_delta, groups)):
        dtr = lgb.Dataset(X[tr], y_delta[tr])
        dva = lgb.Dataset(X[va], y_delta[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        oof_delta[va] = m.predict(X[va], num_iteration=m.best_iteration)
        tp_folds.append(m.predict(Xt, num_iteration=m.best_iteration))
        abs_pred = last_known[va] + oof_delta[va]
        r = rmse(abs_pred[is_lateral[va]], y_abs[va][is_lateral[va]])
        fold_rmses.append(r)
        print(f" fold{fi}: best_iter={m.best_iteration} lateral={r:.4f}", flush=True)

    oof_abs = last_known + oof_delta
    r_lat = rmse(oof_abs[is_lateral], y_abs[is_lateral])
    r_full = rmse(oof_abs, y_abs)
    test_abs = test_last_known + np.mean(np.column_stack(tp_folds), axis=1)
    print(f"\nlateral={r_lat:.4f} full={r_full:.4f}")
    np.save(ROOT/"experiments/oof/oof_lightgbm_exp030.npy", oof_abs.astype(np.float32))
    np.save(ROOT/"experiments/test_preds/test_lightgbm_exp030.npy", test_abs.astype(np.float32))
    json.dump({
        "experiment_id": "exp030", "model": "lgbm_delta_bwell_ncc_tda",
        "phase": "feature_engineering",
        "cv_rmse": r_full, "cv_rmse_lateral": r_lat,
        "fold_rmses_lateral": fold_rmses,
        "n_features": len(feat_cols),
        "notes": "exp026 + TDA offsets + GR lag/lead + diff/env/energy + multi-window rolling + slope features",
        "oof_path": "experiments/oof/oof_lightgbm_exp030.npy",
        "test_path": "experiments/test_preds/test_lightgbm_exp030.npy",
    }, open(ROOT/"experiments/results/exp030.json","w"), indent=2)
    print(f"exp030: lateral={r_lat:.4f}")


if __name__ == "__main__":
    main()
