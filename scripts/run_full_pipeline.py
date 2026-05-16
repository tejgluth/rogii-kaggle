"""
End-to-end Rogii pipeline:
  1. Load train+test wells and typewells
  2. Build features (geometric + GR rolling + typewell-correlation + TVT_input extrapolation)
  3. Train LightGBM and XGBoost with 5-fold GroupKFold (by well_id)
  4. Save OOF and test predictions
  5. Hill-climbing + Ridge stacking
  6. Write submission.csv
"""
import os, sys, json, time, glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/rogii-wellbore-geology-prediction"
OOF_DIR = ROOT / "experiments/oof"
TEST_DIR = ROOT / "experiments/test_preds"
RES_DIR = ROOT / "experiments/results"
for d in (OOF_DIR, TEST_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------- Load ----------
def load_split(split):
    rows = []
    typewells = {}
    folder = DATA / split
    horiz_files = sorted(p for p in folder.glob("*__horizontal_well.csv"))
    for hf in horiz_files:
        wid = hf.name.replace("__horizontal_well.csv", "")
        df = pd.read_csv(hf)
        df.columns = [c.lower() for c in df.columns]
        df["well_id"] = wid
        df["row_idx"] = np.arange(len(df))
        rows.append(df)
        tw = pd.read_csv(folder / f"{wid}__typewell.csv")
        tw.columns = [c.lower() for c in tw.columns]
        tw = tw.sort_values("tvt").reset_index(drop=True)
        typewells[wid] = tw
    full = pd.concat(rows, ignore_index=True)
    return full, typewells


# ---------- Feature engineering ----------
def per_well_extrap(df):
    """For each well: fill TVT_input forward, compute distance to heel/last-known,
    slope/intercept of known TVT_input over MD as linear extrapolation."""
    df = df.copy()
    md = df["md"].values
    well = df["well_id"].values
    tvt_in = df["tvt_input"].values.astype(np.float64)

    extrap = np.full(len(df), np.nan, dtype=np.float64)
    dist_from_known = np.zeros(len(df), dtype=np.float64)
    last_known_tvt = np.full(len(df), np.nan, dtype=np.float64)
    last_known_md = np.full(len(df), np.nan, dtype=np.float64)
    slope_arr = np.zeros(len(df), dtype=np.float64)
    intercept_arr = np.zeros(len(df), dtype=np.float64)
    n_known_arr = np.zeros(len(df), dtype=np.float64)
    well_id_to_idx = {}

    for wid, sub_idx in pd.Series(np.arange(len(df))).groupby(well).groups.items():
        idx = np.asarray(sub_idx)
        sub_md = md[idx]
        sub_t = tvt_in[idx]
        known = np.isfinite(sub_t)
        n_known = known.sum()
        n_known_arr[idx] = n_known
        if n_known >= 2:
            # linear regression
            x = sub_md[known]
            y = sub_t[known]
            xm = x.mean(); ym = y.mean()
            denom = ((x - xm) ** 2).sum()
            slope = ((x - xm) * (y - ym)).sum() / max(denom, 1e-9)
            intercept = ym - slope * xm
        elif n_known == 1:
            slope = 0.0
            intercept = sub_t[known][0]
        else:
            slope = 0.0
            intercept = np.nanmedian(tvt_in) if np.isfinite(np.nanmedian(tvt_in)) else 0.0
        extrap_w = slope * sub_md + intercept
        slope_arr[idx] = slope
        intercept_arr[idx] = intercept
        extrap[idx] = extrap_w
        # last known forward-fill
        last_t = np.nan; last_m = np.nan
        lkt = np.empty_like(sub_t); lkm = np.empty_like(sub_t)
        for i in range(len(idx)):
            if known[i]:
                last_t = sub_t[i]; last_m = sub_md[i]
            lkt[i] = last_t; lkm[i] = last_m
        last_known_tvt[idx] = lkt
        last_known_md[idx] = lkm
        dist_from_known[idx] = sub_md - lkm
    df["tvt_extrap"] = extrap
    df["tvt_input_filled"] = np.where(np.isfinite(tvt_in), tvt_in, extrap)
    df["last_known_tvt"] = last_known_tvt
    df["last_known_md"] = last_known_md
    df["dist_from_known_md"] = dist_from_known
    df["tvt_input_slope"] = slope_arr
    df["tvt_input_intercept"] = intercept_arr
    df["n_known_tvt_input"] = n_known_arr
    return df


def gr_features(df):
    df = df.copy()
    gr = df["gr"].copy()
    # Forward fill GR within well
    df["gr_filled"] = df.groupby("well_id")["gr"].transform(lambda s: s.ffill().bfill())
    gr_f = df["gr_filled"]
    grouped = df.groupby("well_id", sort=False)["gr_filled"]
    for w in (5, 15, 50, 150):
        df[f"gr_mean_{w}"] = grouped.transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"gr_std_{w}"] = grouped.transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
    for p in (1, 5, 20):
        df[f"gr_diff_{p}"] = grouped.transform(lambda s: s.diff(p)).fillna(0)
    df["gr_pct_rank"] = grouped.transform(lambda s: s.rank(pct=True))
    df["gr_zscore"] = grouped.transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    well_stats = df.groupby("well_id")["gr_filled"].agg(["mean", "std", "min", "max"]).rename(
        columns=lambda c: f"well_gr_{c}")
    df = df.join(well_stats, on="well_id")
    df["gr_isna"] = gr.isna().astype(np.float32)
    return df


def typewell_features(df, typewells, max_lag=80, window=51):
    """Local correlation between lateral GR and typewell GR, evaluated at lateral TVT_input + lag."""
    df = df.copy()
    feat_names = [
        "tw_best_lag", "tw_best_corr", "tw_corr_lag0",
        "tw_corr_peakiness", "tw_gr_at_best_lag",
        "tw_gr_minus_local", "tw_best_lag_smooth",
        "tw_gr_at_extrap", "tw_gr_at_lastknown",
    ]
    for n in feat_names:
        df[n] = np.float32(0.0)

    gr_f = df["gr_filled"].values.astype(np.float64)
    extrap = df["tvt_input_filled"].values.astype(np.float64)
    last_known = df["last_known_tvt"].values.astype(np.float64)
    last_known = np.where(np.isfinite(last_known), last_known, extrap)
    well = df["well_id"].values

    lags = np.arange(-max_lag, max_lag + 1, dtype=np.float64)
    out = {n: np.zeros(len(df), dtype=np.float32) for n in feat_names}

    for wid, idx in pd.Series(np.arange(len(df))).groupby(well, sort=False).groups.items():
        idx = np.asarray(idx)
        tw = typewells[wid]
        tw_tvt = tw["tvt"].values.astype(np.float64)
        tw_gr = tw["gr"].values.astype(np.float64)
        valid = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
        tw_tvt = tw_tvt[valid]; tw_gr = tw_gr[valid]
        if len(tw_tvt) < 2:
            continue
        order = np.argsort(tw_tvt)
        tw_tvt = tw_tvt[order]; tw_gr = tw_gr[order]

        loc_gr = gr_f[idx]
        prior = extrap[idx]
        n = len(idx)
        half = window // 2
        starts = np.maximum(0, np.arange(n) - half)
        ends = np.minimum(n, np.arange(n) + half + 1)
        counts = (ends - starts).astype(np.float64)
        cs = np.r_[0.0, np.cumsum(loc_gr)]
        cs2 = np.r_[0.0, np.cumsum(loc_gr * loc_gr)]
        sum_x = cs[ends] - cs[starts]
        sum_x2 = cs2[ends] - cs2[starts]
        var_x = np.maximum(sum_x2 - (sum_x ** 2) / counts, 0.0)

        # vectorize over lags
        corr = np.empty((len(lags), n), dtype=np.float32)
        for li, lag in enumerate(lags):
            tw_at = np.interp(prior + lag, tw_tvt, tw_gr)
            csy = np.r_[0.0, np.cumsum(tw_at)]
            csy2 = np.r_[0.0, np.cumsum(tw_at * tw_at)]
            csxy = np.r_[0.0, np.cumsum(loc_gr * tw_at)]
            sum_y = csy[ends] - csy[starts]
            sum_y2 = csy2[ends] - csy2[starts]
            sum_xy = csxy[ends] - csxy[starts]
            cov = sum_xy - (sum_x * sum_y) / counts
            var_y = np.maximum(sum_y2 - (sum_y ** 2) / counts, 0.0)
            denom = np.sqrt(var_x * var_y) + 1e-6
            corr[li] = (cov / denom).astype(np.float32)

        best_idx = np.argmax(corr, axis=0)
        best_corr = corr[best_idx, np.arange(n)]
        best_lag = lags[best_idx].astype(np.float32)
        lag0 = int(np.where(lags == 0)[0][0])
        corr0 = corr[lag0]
        corr_std = corr.std(axis=0) + 1e-6
        peak = (best_corr - corr.mean(axis=0)) / corr_std
        tw_at_best = np.interp(prior + best_lag.astype(np.float64), tw_tvt, tw_gr)
        tw_at_extrap = np.interp(prior, tw_tvt, tw_gr)
        tw_at_lk = np.interp(last_known[idx], tw_tvt, tw_gr)
        # smooth lag
        smooth = pd.Series(best_lag).rolling(75, center=True, min_periods=1).median().values

        out["tw_best_lag"][idx] = best_lag
        out["tw_best_corr"][idx] = best_corr
        out["tw_corr_lag0"][idx] = corr0
        out["tw_corr_peakiness"][idx] = peak.astype(np.float32)
        out["tw_gr_at_best_lag"][idx] = tw_at_best.astype(np.float32)
        out["tw_gr_minus_local"][idx] = (tw_at_best - loc_gr).astype(np.float32)
        out["tw_best_lag_smooth"][idx] = smooth.astype(np.float32)
        out["tw_gr_at_extrap"][idx] = tw_at_extrap.astype(np.float32)
        out["tw_gr_at_lastknown"][idx] = tw_at_lk.astype(np.float32)

    for n, v in out.items():
        df[n] = v
    return df


def add_geometry(df):
    df = df.copy()
    for c in ("x", "y", "z"):
        if c in df.columns:
            df[f"{c}_rel"] = df.groupby("well_id")[c].transform(lambda s: s - s.iloc[0])
    df["md_rel"] = df.groupby("well_id")["md"].transform(lambda s: s - s.iloc[0])
    df["z_grad"] = df.groupby("well_id")["z"].transform(lambda s: s.diff().fillna(0))
    df["md_step"] = df.groupby("well_id")["md"].transform(lambda s: s.diff().fillna(0))
    return df


def build_all_features(df, typewells):
    print("  per-well TVT_input extrap...")
    df = per_well_extrap(df)
    print("  GR rolling...")
    df = gr_features(df)
    print("  geometry...")
    df = add_geometry(df)
    print("  typewell correlations (this is the slow one)...")
    df = typewell_features(df, typewells)
    # Replace inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ---------- Training ----------
FEATURE_BLACKLIST = {"tvt", "well_id", "well_file", "row_idx", "gr", "tvt_input"}

def get_feature_cols(df):
    cols = [c for c in df.columns if c not in FEATURE_BLACKLIST and df[c].dtype != object]
    return cols


def train_lgbm(X_tr, y_tr, X_va, y_va, X_te, **kw):
    import lightgbm as lgb
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l2=1.0,
        verbosity=-1,
        n_jobs=-1,
    )
    params.update(kw)
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    p_te = model.predict(X_te, num_iteration=model.best_iteration)
    return p_va, p_te, model


def train_xgb(X_tr, y_tr, X_va, y_va, X_te, **kw):
    import xgboost as xgb
    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
    )
    params.update(kw)
    dtr = xgb.DMatrix(X_tr, y_tr)
    dva = xgb.DMatrix(X_va, y_va)
    dte = xgb.DMatrix(X_te)
    model = xgb.train(
        params, dtr, num_boost_round=3000,
        evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=0,
    )
    p_va = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
    p_te = model.predict(dte, iteration_range=(0, model.best_iteration + 1))
    return p_va, p_te, model


def kfold_train(X_tr_full, y_tr, groups, X_te, fn, name, n_splits=5):
    print(f"\n=== Training {name} ({n_splits}-fold GroupKFold) ===")
    oof = np.zeros(len(y_tr), dtype=np.float32)
    test_pred = np.zeros(X_te.shape[0], dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    fold_rmses = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_full, y_tr, groups)):
        t0 = time.time()
        p_va, p_te, _ = fn(X_tr_full[tr_idx], y_tr[tr_idx], X_tr_full[va_idx], y_tr[va_idx], X_te)
        oof[va_idx] = p_va
        test_pred += p_te / n_splits
        rmse = np.sqrt(mean_squared_error(y_tr[va_idx], p_va))
        fold_rmses.append(rmse)
        print(f"  fold {fold}: rmse={rmse:.4f}  time={time.time()-t0:.1f}s")
    overall = np.sqrt(mean_squared_error(y_tr, oof))
    print(f"{name} OOF RMSE: {overall:.4f}  fold-std: {np.std(fold_rmses):.4f}")
    return oof, test_pred, overall, np.std(fold_rmses)


# ---------- Stacking ----------
def hill_climb(oof_dict, y, groups, n_iter=200):
    from sklearn.metrics import mean_squared_error
    # start with best
    best_rmse = float("inf"); selected = []
    for k, v in oof_dict.items():
        r = np.sqrt(mean_squared_error(y, v))
        if r < best_rmse:
            best_rmse = r; selected = [k]
    print(f"start: {selected[0]}={best_rmse:.4f}")
    weights = {k: 1 for k in selected}
    cur = oof_dict[selected[0]].copy()
    for _ in range(n_iter):
        improved = False
        for k in oof_dict:
            cand = (cur * len(selected) + oof_dict[k]) / (len(selected) + 1)
            r = np.sqrt(mean_squared_error(y, cand))
            if r < best_rmse - 1e-5:
                best_rmse = r
                selected.append(k)
                cur = cand
                improved = True
                print(f"  +{k}: {r:.4f}")
                break
        if not improved:
            break
    return selected, best_rmse


def ridge_stack(oof_dict, y, groups, test_dict):
    keys = list(oof_dict)
    Xm = np.column_stack([oof_dict[k] for k in keys])
    Xt = np.column_stack([test_dict[k] for k in keys])
    gkf = GroupKFold(5)
    oof = np.zeros(len(y))
    test = np.zeros(Xt.shape[0])
    for tr, va in gkf.split(Xm, y, groups):
        r = Ridge(alpha=1.0, positive=True)
        r.fit(Xm[tr], y[tr])
        oof[va] = r.predict(Xm[va])
        test += r.predict(Xt) / 5
    rmse = np.sqrt(mean_squared_error(y, oof))
    return oof, test, rmse, keys


# ---------- Main ----------
def main():
    t0 = time.time()
    print("Loading train...")
    train_df, train_tw = load_split("train")
    print(f"  train: {train_df.shape}, wells={train_df['well_id'].nunique()}")
    print("Loading test...")
    test_df, test_tw = load_split("test")
    print(f"  test:  {test_df.shape}, wells={test_df['well_id'].nunique()}")

    print("Building train features...")
    train_feat = build_all_features(train_df, train_tw)
    print("Building test features...")
    test_feat = build_all_features(test_df, test_tw)

    # Align feature columns
    feat_cols = [c for c in train_feat.columns
                 if c in test_feat.columns and c not in FEATURE_BLACKLIST
                 and train_feat[c].dtype != object]
    print(f"feature cols: {len(feat_cols)}")

    X_tr = train_feat[feat_cols].fillna(-999).values.astype(np.float32)
    y_tr = train_feat["tvt"].values.astype(np.float32)
    groups = train_feat["well_id"].values
    X_te = test_feat[feat_cols].fillna(-999).values.astype(np.float32)

    print(f"X_tr={X_tr.shape}  X_te={X_te.shape}")

    # Save feature names
    Path(RES_DIR / "feature_cols_phase4.json").write_text(json.dumps(feat_cols, indent=2))

    # Train LightGBM (deep)
    oof_lgbm, te_lgbm, rmse_lgbm, std_lgbm = kfold_train(X_tr, y_tr, groups, X_te, train_lgbm, "lgbm_phase4")
    np.save(OOF_DIR / "oof_lgbm_phase4.npy", oof_lgbm.astype(np.float32))
    np.save(TEST_DIR / "test_lgbm_phase4.npy", te_lgbm.astype(np.float32))

    # Train XGBoost
    oof_xgb, te_xgb, rmse_xgb, std_xgb = kfold_train(X_tr, y_tr, groups, X_te, train_xgb, "xgb_phase4")
    np.save(OOF_DIR / "oof_xgb_phase4.npy", oof_xgb.astype(np.float32))
    np.save(TEST_DIR / "test_xgb_phase4.npy", te_xgb.astype(np.float32))

    # Light LGBM variant (more rounds, smaller LR for diversity)
    def train_lgbm_v2(*a, **kw):
        return train_lgbm(*a, learning_rate=0.02, num_leaves=63, lambda_l2=3.0)
    oof_lgbm2, te_lgbm2, rmse_lgbm2, _ = kfold_train(X_tr, y_tr, groups, X_te, train_lgbm_v2, "lgbm_phase4_v2")
    np.save(OOF_DIR / "oof_lgbm_phase4_v2.npy", oof_lgbm2.astype(np.float32))
    np.save(TEST_DIR / "test_lgbm_phase4_v2.npy", te_lgbm2.astype(np.float32))

    # ---- Stacking ----
    print("\n=== Phase 4 Stacking ===")
    oof_dict = {
        "lgbm_phase4": oof_lgbm,
        "xgb_phase4": oof_xgb,
        "lgbm_phase4_v2": oof_lgbm2,
    }
    test_dict = {
        "lgbm_phase4": te_lgbm,
        "xgb_phase4": te_xgb,
        "lgbm_phase4_v2": te_lgbm2,
    }
    # Include older OOFs if they exist
    for old in ("oof_lgbm_exp_b001", "oof_xgb_exp_b002", "oof_lightgbm_exp003"):
        p = OOF_DIR / f"{old}.npy"
        tp = TEST_DIR / f"{old.replace('oof','test')}.npy"
        if p.exists() and tp.exists():
            arr = np.load(p)
            if arr.shape[0] == len(y_tr):
                oof_dict[old] = arr
                test_dict[old] = np.load(tp)

    # Hill climb
    selected, hc_rmse = hill_climb(oof_dict, y_tr, groups)
    print(f"Hill-climbing RMSE: {hc_rmse:.4f}, selected: {selected}")

    # Ridge stack
    ridge_oof, ridge_test, ridge_rmse, keys = ridge_stack(oof_dict, y_tr, groups, test_dict)
    print(f"Ridge stacker RMSE: {ridge_rmse:.4f}")

    # Mean of selected
    mean_test = np.mean([test_dict[k] for k in selected], axis=0)

    # Pick the better one
    if ridge_rmse < hc_rmse:
        final_test = ridge_test
        final_oof = ridge_oof
        method = "ridge"
        final_rmse = ridge_rmse
    else:
        final_test = mean_test
        final_oof = np.mean([oof_dict[k] for k in selected], axis=0)
        method = "hill_climb_mean"
        final_rmse = hc_rmse

    np.save(TEST_DIR / "final_ensemble.npy", final_test.astype(np.float32))
    np.save(OOF_DIR / "final_ensemble.npy", final_oof.astype(np.float32))

    stacking_result = {
        "method": method,
        "cv_rmse_final": float(final_rmse),
        "cv_rmse_lgbm_phase4": float(rmse_lgbm),
        "cv_rmse_xgb_phase4": float(rmse_xgb),
        "cv_rmse_lgbm_phase4_v2": float(rmse_lgbm2),
        "cv_rmse_hill_climb": float(hc_rmse),
        "cv_rmse_ridge": float(ridge_rmse),
        "hill_climb_selected": selected,
        "ridge_keys": keys,
        "n_features": len(feat_cols),
    }
    (RES_DIR / "stacking_result.json").write_text(json.dumps(stacking_result, indent=2))
    print(json.dumps(stacking_result, indent=2))

    # ---- Submission ----
    print("\n=== Building submission ===")
    ss = pd.read_csv(DATA / "sample_submission.csv")
    # Match by well_id_row_idx
    test_ids = (test_feat["well_id"].astype(str) + "_" + test_feat["row_idx"].astype(int).astype(str)).values
    id_to_pred = dict(zip(test_ids, final_test))
    # Post-process: if TVT_input is known, use it (it equals TVT exactly)
    tvt_in = test_feat["tvt_input"].values
    for i, tid in enumerate(test_ids):
        if np.isfinite(tvt_in[i]):
            id_to_pred[tid] = tvt_in[i]
    ss["tvt"] = ss["id"].map(id_to_pred)
    n_missing = ss["tvt"].isna().sum()
    print(f"missing in submission: {n_missing}")
    if n_missing:
        ss["tvt"] = ss["tvt"].fillna(ss["tvt"].median())
    sub_path = ROOT / "submission.csv"
    ss.to_csv(sub_path, index=False)
    print(f"Wrote {sub_path}: shape {ss.shape}  range [{ss['tvt'].min():.2f}, {ss['tvt'].max():.2f}]")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    return stacking_result


if __name__ == "__main__":
    main()
