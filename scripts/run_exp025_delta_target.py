"""exp025: Delta-from-last-known-TVT target reformulation.

Key insight from public 9.25-RMSE kernel: predict TVT - last_known_TVT (the delta
from the heel) instead of absolute TVT. Test wells provide TVT_input for the
heel; the lateral portion (~74% of rows) has TVT_input=NaN and needs prediction.

This script does NOT use existing OOFs. It trains fresh on the delta target
using the existing exp014 feature columns + several new ones derived from
last-known anchoring and the formation columns (ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA)
which are in the train/test CSVs but have never been used.
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


def build_well_df(path: Path, has_target: bool):
    df = pd.read_csv(path)
    df["well_id"] = path.stem.replace("__horizontal_well", "")

    # Find heel anchor (last row where TVT_input is known)
    known_mask = df["TVT_input"].notna()
    if known_mask.any():
        anchor_idx = np.where(known_mask.values)[0][-1]
        last_known_tvt = df["TVT_input"].iloc[anchor_idx]
        last_known_md = df["MD"].iloc[anchor_idx]
        last_known_x = df["X"].iloc[anchor_idx]
        last_known_y = df["Y"].iloc[anchor_idx]
        last_known_z = df["Z"].iloc[anchor_idx]
    else:
        anchor_idx = -1
        last_known_tvt = df["TVT_input"].mean() if df["TVT_input"].notna().any() else 0.0
        last_known_md = df["MD"].iloc[0]
        last_known_x = df["X"].iloc[0]
        last_known_y = df["Y"].iloc[0]
        last_known_z = df["Z"].iloc[0]

    df["last_known_tvt"] = last_known_tvt
    df["last_known_md"] = last_known_md
    df["md_since_anchor"] = df["MD"] - last_known_md
    df["dx_since_anchor"] = df["X"] - last_known_x
    df["dy_since_anchor"] = df["Y"] - last_known_y
    df["dz_since_anchor"] = df["Z"] - last_known_z
    df["xy_dist_since_anchor"] = np.sqrt(df["dx_since_anchor"]**2 + df["dy_since_anchor"]**2)
    df["anchor_idx"] = anchor_idx
    df["is_lateral"] = ~known_mask  # rows we need to predict

    # Formation TVT depths (constant per well — these come from typewell)
    # In TVT space the formation values are negative (-9000 to -9800).
    # They give relative positions among layers.
    for fm in FORMATIONS:
        if fm in df.columns:
            df[f"fm_{fm}"] = df[fm]
        else:
            df[f"fm_{fm}"] = np.nan

    # Formation deltas to last_known (since formations are constant within well,
    # this is just `formation - last_known_tvt`, but it tells the model where
    # the heel sits relative to layers — extremely useful).
    for fm in FORMATIONS:
        df[f"fm_{fm}_minus_lastknown"] = df[f"fm_{fm}"] - last_known_tvt
        df[f"fm_{fm}_minus_z"] = df[f"fm_{fm}"] - df["Z"]

    # GR features
    df["gr_filled"] = df["GR"].ffill().bfill().fillna(0)
    df["gr_rolling_mean_30"] = df["gr_filled"].rolling(30, center=True, min_periods=1).mean()
    df["gr_rolling_std_30"] = df["gr_filled"].rolling(30, center=True, min_periods=1).std().fillna(0)
    df["gr_rolling_mean_100"] = df["gr_filled"].rolling(100, center=True, min_periods=1).mean()

    # Trajectory inc/azim approx
    df["dz_rolling"] = df["Z"].diff().rolling(20, center=True, min_periods=1).mean().fillna(0)
    df["xy_rolling"] = (np.sqrt(df["X"].diff()**2 + df["Y"].diff()**2)).rolling(20, center=True, min_periods=1).mean().fillna(0)

    if has_target:
        df["delta_target"] = df["TVT"] - last_known_tvt
    return df


def main():
    print("Loading train...")
    t0 = time.time()
    train_files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    dfs = []
    for p in train_files:
        dfs.append(build_well_df(p, has_target=True))
    train = pd.concat(dfs, ignore_index=True)
    print(f"  train rows={len(train)} ({time.time()-t0:.1f}s)")

    test_files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    tdfs = [build_well_df(p, has_target=False) for p in test_files]
    test = pd.concat(tdfs, ignore_index=True)
    print(f"  test rows={len(test)}")

    feature_cols = [
        "MD", "X", "Y", "Z", "GR", "TVT_input",
        "gr_filled", "gr_rolling_mean_30", "gr_rolling_std_30", "gr_rolling_mean_100",
        "last_known_tvt", "last_known_md",
        "md_since_anchor", "dx_since_anchor", "dy_since_anchor", "dz_since_anchor",
        "xy_dist_since_anchor", "dz_rolling", "xy_rolling",
    ]
    for fm in FORMATIONS:
        feature_cols += [f"fm_{fm}", f"fm_{fm}_minus_lastknown", f"fm_{fm}_minus_z"]

    X = train[feature_cols].values.astype(np.float32)
    y_delta = train["delta_target"].values.astype(np.float32)
    y_abs = train["TVT"].values.astype(np.float32)
    groups = train["well_id"].values
    last_known = train["last_known_tvt"].values.astype(np.float32)
    is_lateral = train["is_lateral"].values

    Xt = test[feature_cols].values.astype(np.float32)
    test_last_known = test["last_known_tvt"].values.astype(np.float32)

    print(f"Features: {len(feature_cols)}")
    print(f"Target stats: mean={y_delta.mean():.2f} std={y_delta.std():.2f}")
    print(f"Naive predict-0 RMSE on TRAIN (absolute): {rmse(y_abs, last_known):.4f}")
    print(f"Naive predict-0 RMSE on TRAIN lateral only: {rmse(y_abs[is_lateral], last_known[is_lateral]):.4f}")

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.03, num_leaves=127, min_data_in_leaf=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=2.0, verbose=-1, num_threads=-1, max_bin=255,
    )

    gkf = GroupKFold(5)
    oof_delta = np.zeros(len(y_delta), dtype=np.float32)
    test_preds_folds = []
    fold_rmses_lateral = []
    fold_rmses_full = []
    for fi, (tr, va) in enumerate(gkf.split(X, y_delta, groups)):
        dtr = lgb.Dataset(X[tr], y_delta[tr])
        dva = lgb.Dataset(X[va], y_delta[va], reference=dtr)
        model = lgb.train(params, dtr, num_boost_round=3000,
                          valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_delta[va] = model.predict(X[va], num_iteration=model.best_iteration)
        test_preds_folds.append(model.predict(Xt, num_iteration=model.best_iteration))
        # Reconstruct absolute TVT
        abs_pred = last_known[va] + oof_delta[va]
        r_full = rmse(abs_pred, y_abs[va])
        m_lat = is_lateral[va]
        r_lat = rmse(abs_pred[m_lat], y_abs[va][m_lat])
        fold_rmses_full.append(r_full)
        fold_rmses_lateral.append(r_lat)
        print(f" fold{fi}: best_iter={model.best_iteration} full_RMSE={r_full:.4f} lateral_RMSE={r_lat:.4f}")

    # Full OOF RMSE in absolute space
    oof_abs = last_known + oof_delta
    overall_full = rmse(oof_abs, y_abs)
    overall_lateral = rmse(oof_abs[is_lateral], y_abs[is_lateral])
    print(f"\nOOF full RMSE (all rows): {overall_full:.4f}")
    print(f"OOF lateral RMSE (rows to predict): {overall_lateral:.4f}")

    # Test predictions
    test_delta = np.mean(np.column_stack(test_preds_folds), axis=1)
    test_abs = test_last_known + test_delta
    np.save(ROOT/"experiments/oof/oof_lightgbm_exp025.npy", oof_abs.astype(np.float32))
    np.save(ROOT/"experiments/test_preds/test_lightgbm_exp025.npy", test_abs.astype(np.float32))
    json.dump({
        "experiment_id": "exp025",
        "model": "lightgbm_delta_target",
        "phase": "feature_engineering",
        "cv_rmse": overall_full,
        "cv_rmse_lateral": overall_lateral,
        "fold_rmses_full": fold_rmses_full,
        "fold_rmses_lateral": fold_rmses_lateral,
        "features_used": feature_cols,
        "n_features": len(feature_cols),
        "training_time_seconds": time.time() - t0,
        "notes": f"Delta-from-last-known target. Predict TVT - last_known_tvt, "
                 f"add formation cols as features. Massive improvement expected.",
        "oof_path": "experiments/oof/oof_lightgbm_exp025.npy",
        "test_path": "experiments/test_preds/test_lightgbm_exp025.npy",
    }, open(ROOT/"experiments/results/exp025.json", "w"), indent=2)
    print(f"exp025 saved: full={overall_full:.4f} lateral={overall_lateral:.4f}")


if __name__ == "__main__":
    main()
