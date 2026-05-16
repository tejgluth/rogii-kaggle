"""exp026: Delta target + formation b_well anchoring + GR NCC alignment.

Building on exp025 (delta target). Adds:
1. b_well formation offset: for each formation F, b_F = median(TVT_known + Z_known - F)
   on the heel rows. Then predicted TVT for lateral row = -Z + F + b_F.
   These give 6 (one per formation) physical-prior TVT estimates per row.
2. Multi-scale NCC anchor: cross-correlate the lateral GR window with typewell GR
   at varying offsets; pick best lag. (Mimics the 9.25 RMSE kernel's NCC.)
3. Last-known anchor features: TVT_input value, GR at heel, distance from heel.

All features can be computed from the train/test CSVs + typewell CSVs.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def multi_scale_ncc_offsets(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    """Return TVT prediction per hgr row using multi-scale NCC against known GR/TVT."""
    out_tvt = []
    out_score = []
    for hw in hws:
        win = 2 * hw + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out_tvt.append(np.full(nh, ktvt[-1] if len(ktvt) else 0, dtype=np.float32))
            out_score.append(np.zeros(nh, dtype=np.float32))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        M = len(sts)
        if M == 0:
            out_tvt.append(np.full(nh, ktvt[-1] if len(ktvt) else 0, dtype=np.float32))
            out_score.append(np.zeros(nh, dtype=np.float32))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode='edge')
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out_tvt.append(ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32))
        out_score.append(score)
    return out_tvt, out_score


def build_well(hw_path, tw_path, is_train):
    wid = hw_path.stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path)
    if tw_path.exists():
        tw = pd.read_csv(tw_path).sort_values("TVT")
    else:
        tw = None

    kn_mask = hw["TVT_input"].notna()
    if not kn_mask.any():
        return None
    kn = hw[kn_mask]
    ev = hw[~kn_mask]
    if is_train and "TVT" not in hw.columns:
        return None

    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    last_md = float(lk["MD"])
    last_x = float(lk["X"]); last_y = float(lk["Y"]); last_z = float(lk["Z"])
    last_gr = float(lk["GR"]) if pd.notna(lk["GR"]) else float(kn["GR"].mean())

    # Build per-row features for ALL rows in the well
    out = pd.DataFrame(index=hw.index)
    out["well_id"] = wid
    out["MD"] = hw["MD"].values
    out["X"] = hw["X"].values
    out["Y"] = hw["Y"].values
    out["Z"] = hw["Z"].values
    out["GR"] = hw["GR"].values
    out["TVT_input"] = hw["TVT_input"].values
    out["is_lateral"] = (~kn_mask).values
    out["last_known_tvt"] = last_tvt
    out["last_known_md"] = last_md
    out["last_known_gr"] = last_gr
    out["md_since_anchor"] = hw["MD"].values - last_md
    out["dx_since_anchor"] = hw["X"].values - last_x
    out["dy_since_anchor"] = hw["Y"].values - last_y
    out["dz_since_anchor"] = hw["Z"].values - last_z
    out["xy_dist"] = np.sqrt(out["dx_since_anchor"]**2 + out["dy_since_anchor"]**2)

    # GR features
    gr = pd.Series(hw["GR"]).ffill().bfill().fillna(0)
    out["gr_filled"] = gr.values
    out["gr_rm_30"] = gr.rolling(30, center=True, min_periods=1).mean().values
    out["gr_rm_100"] = gr.rolling(100, center=True, min_periods=1).mean().values
    out["gr_rs_30"] = gr.rolling(30, center=True, min_periods=1).std().fillna(0).values

    # b_well anchoring per formation
    ktvt = kn["TVT_input"].values.astype(np.float64)
    kz = kn["Z"].values.astype(np.float64)
    z_all = hw["Z"].values.astype(np.float64)
    for fm in FORMATIONS:
        if fm in hw.columns and hw[fm].notna().any():
            f_all = hw[fm].values.astype(np.float64)
            f_kn = f_all[kn_mask.values]
            # b_well = median(TVT + Z - F) on known rows
            bv = ktvt + kz - f_kn
            b_full = float(np.median(bv))
            n = len(bv)
            if n >= 50:
                b_late = float(np.median(bv[-50:]))
            else:
                b_late = b_full
            # WLS
            w = np.exp(0.02 * np.arange(n)); w /= w.sum()
            b_wls = float(np.dot(w, bv))
            # Predicted TVT per row using each variant:
            pred_full = -z_all + f_all + b_full
            pred_late = -z_all + f_all + b_late
            pred_wls = -z_all + f_all + b_wls
            out[f"fm_{fm}_tvt"] = pred_full.astype(np.float32)
            out[f"fm_{fm}_tvt_late"] = pred_late.astype(np.float32)
            out[f"fm_{fm}_tvt_wls"] = pred_wls.astype(np.float32)
            out[f"fm_{fm}_bwell"] = b_full
            out[f"fm_{fm}_bwell_late"] = b_late
            # also TVT_input value of the formation depth differences
            out[f"fm_{fm}_minus_z"] = (f_all - z_all).astype(np.float32)
            out[f"fm_{fm}_minus_lk"] = float(f_all[0] - last_tvt)
        else:
            for c in [f"fm_{fm}_tvt", f"fm_{fm}_tvt_late", f"fm_{fm}_tvt_wls",
                      f"fm_{fm}_minus_z"]:
                out[c] = 0.0
            for c in [f"fm_{fm}_bwell", f"fm_{fm}_bwell_late", f"fm_{fm}_minus_lk"]:
                out[c] = 0.0

    # Mean/std of formation TVT predictions across formations
    fm_preds = np.column_stack([out[f"fm_{fm}_tvt"].values for fm in FORMATIONS])
    out["fm_mean_tvt"] = fm_preds.mean(1)
    out["fm_std_tvt"] = fm_preds.std(1)
    out["fm_min_tvt"] = fm_preds.min(1)
    out["fm_max_tvt"] = fm_preds.max(1)
    out["fm_mean_minus_lk"] = out["fm_mean_tvt"].values - last_tvt

    # NCC alignment vs typewell
    if tw is not None and "GR" in tw.columns:
        tw_tvt = tw["TVT"].values.astype(np.float32)
        tw_gr = tw["GR"].values.astype(np.float32)
        kgr = kn["GR"].ffill().bfill().fillna(0).values.astype(np.float32)
        ktvt_f32 = ktvt.astype(np.float32)
        # NCC against ktvt is only useful for finding TVT shift inside known regime
        # but we want lateral predictions — use typewell as the reference instead
        # Actually the kernel's NCC compares lateral GR (hgr) to known-region GR (kgr) to find
        # where lateral pattern matches a known pattern → corresponding TVT.
        hgr = gr.values.astype(np.float32)
        ncc_tvts, ncc_scores = multi_scale_ncc_offsets(kgr, ktvt_f32, hgr, hws=(8, 15, 25))
        for i, hw_n in enumerate([8, 15, 25]):
            out[f"ncc{hw_n}_tvt"] = ncc_tvts[i]
            out[f"ncc{hw_n}_score"] = ncc_scores[i]
        # Score-weighted ensemble
        tvts = np.stack(ncc_tvts, 1)
        scores = np.stack(ncc_scores, 1)
        sw = np.exp(3. * scores); sw /= sw.sum(1, keepdims=True) + 1e-9
        out["ncc_ens_tvt"] = (tvts * sw).sum(1).astype(np.float32)
        out["ncc_max_score"] = scores.max(1)
        # Also cross-correlate lateral GR against typewell GR directly
        # Use typewell GR at TVT vs heel's TVT ratio
        # Skip — already heavy.
    else:
        for hw_n in [8, 15, 25]:
            out[f"ncc{hw_n}_tvt"] = last_tvt
            out[f"ncc{hw_n}_score"] = 0
        out["ncc_ens_tvt"] = last_tvt
        out["ncc_max_score"] = 0

    if is_train:
        out["TVT"] = hw["TVT"].values
        out["delta_target"] = hw["TVT"].values - last_tvt
    return out


def main():
    t0 = time.time()
    print("Loading train wells...", flush=True)
    train_files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    dfs = []
    for i, p in enumerate(train_files):
        tw_path = p.parent / (p.stem.replace("__horizontal_well", "__typewell") + ".csv")
        df = build_well(p, tw_path, is_train=True)
        if df is not None:
            dfs.append(df)
        if (i+1) % 100 == 0:
            print(f"  loaded {i+1}/{len(train_files)} ({time.time()-t0:.0f}s)", flush=True)
    train = pd.concat(dfs, ignore_index=True)
    print(f"train rows={len(train)} cols={train.shape[1]}")

    print("Loading test wells...", flush=True)
    test_files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    tdfs = []
    for p in test_files:
        tw_path = p.parent / (p.stem.replace("__horizontal_well", "__typewell") + ".csv")
        df = build_well(p, tw_path, is_train=False)
        if df is not None:
            tdfs.append(df)
    test = pd.concat(tdfs, ignore_index=True)
    print(f"test rows={len(test)}")

    skip = {"well_id", "TVT", "delta_target", "is_lateral"}
    feature_cols = [c for c in train.columns if c not in skip]
    print(f"#features: {len(feature_cols)}")

    X = train[feature_cols].values.astype(np.float32)
    y_delta = train["delta_target"].values.astype(np.float32)
    y_abs = train["TVT"].values.astype(np.float32)
    last_known = train["last_known_tvt"].values.astype(np.float32)
    is_lateral = train["is_lateral"].values
    groups = train["well_id"].values

    Xt = test[feature_cols].values.astype(np.float32)
    test_last_known = test["last_known_tvt"].values.astype(np.float32)

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.03, num_leaves=127, min_data_in_leaf=30,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=2.0, verbose=-1, num_threads=-1, max_bin=255,
    )

    gkf = GroupKFold(5)
    oof_delta = np.zeros(len(y_delta), dtype=np.float32)
    test_preds_folds = []
    fold_rmses_lat = []
    for fi, (tr, va) in enumerate(gkf.split(X, y_delta, groups)):
        dtr = lgb.Dataset(X[tr], y_delta[tr])
        dva = lgb.Dataset(X[va], y_delta[va], reference=dtr)
        model = lgb.train(params, dtr, num_boost_round=4000,
                          valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_delta[va] = model.predict(X[va], num_iteration=model.best_iteration)
        test_preds_folds.append(model.predict(Xt, num_iteration=model.best_iteration))
        abs_pred = last_known[va] + oof_delta[va]
        m_lat = is_lateral[va]
        r = rmse(abs_pred[m_lat], y_abs[va][m_lat])
        fold_rmses_lat.append(r)
        print(f" fold{fi}: best_iter={model.best_iteration} lateral_RMSE={r:.4f}", flush=True)

    oof_abs = last_known + oof_delta
    overall_lat = rmse(oof_abs[is_lateral], y_abs[is_lateral])
    overall_full = rmse(oof_abs, y_abs)
    print(f"\nOOF lateral RMSE: {overall_lat:.4f}")
    print(f"OOF full RMSE: {overall_full:.4f}")

    test_delta = np.mean(np.column_stack(test_preds_folds), axis=1)
    test_abs = test_last_known + test_delta
    np.save(ROOT/"experiments/oof/oof_lightgbm_exp026.npy", oof_abs.astype(np.float32))
    np.save(ROOT/"experiments/test_preds/test_lightgbm_exp026.npy", test_abs.astype(np.float32))
    json.dump({
        "experiment_id": "exp026", "model": "lightgbm_delta+bwell+ncc",
        "phase": "feature_engineering",
        "cv_rmse": overall_full, "cv_rmse_lateral": overall_lat,
        "fold_rmses_lateral": fold_rmses_lat,
        "n_features": len(feature_cols),
        "features_used": feature_cols,
        "training_time_seconds": time.time() - t0,
        "notes": "Delta target + formation b_well anchoring + multi-scale NCC alignment.",
        "oof_path": "experiments/oof/oof_lightgbm_exp026.npy",
        "test_path": "experiments/test_preds/test_lightgbm_exp026.npy",
    }, open(ROOT/"experiments/results/exp026.json", "w"), indent=2)
    print(f"exp026 saved: lateral={overall_lat:.4f} full={overall_full:.4f}")


if __name__ == "__main__":
    main()
