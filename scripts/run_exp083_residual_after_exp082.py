"""exp083: residual correction after exp082 validation stack."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def per_well_groups_from_cache(groups):
    return np.asarray(groups)


def per_well_savgol(pred, groups, window, poly):
    out = pred.copy().astype(np.float32)
    for well in np.unique(groups):
        sel = groups == well
        seg = out[sel]
        win = min(window, len(seg))
        if win % 2 == 0:
            win -= 1
        if win >= poly + 2:
            out[sel] = savgol_filter(seg, win, poly)
    return out


def main():
    cache = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
    X = cache["X"]
    Xt = cache["Xt"]
    y = cache["y_abs"]
    is_lat = cache["is_lateral"]
    groups = np.asarray(cache["groups"])

    base_oof = np.load(ROOT / "experiments/oof/oof_stack_exp082.npy")
    base_test = np.load(ROOT / "experiments/test_preds/test_stack_exp082.npy")
    residual = (y - base_oof).astype(np.float32)
    base_score = rmse(base_oof[is_lat], y[is_lat])
    print(f"base exp082 rmse={base_score:.6f} mse={base_score ** 2:.6f}", flush=True)

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.02,
        num_leaves=31,
        min_data_in_leaf=300,
        feature_fraction=0.7,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=2.0,
        lambda_l2=4.0,
        verbose=-1,
        seed=8301,
        num_threads=-1,
    )
    gkf = GroupKFold(5)
    oof_resid = np.zeros(len(y), dtype=np.float32)
    test_resids = []
    folds = []
    for fold, (tr, va) in enumerate(gkf.split(X, residual, groups)):
        dtr = lgb.Dataset(X[tr], label=residual[tr])
        dva = lgb.Dataset(X[va], label=residual[va], reference=dtr)
        model = lgb.train(
            params,
            dtr,
            num_boost_round=3000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof_resid[va] = model.predict(X[va], num_iteration=model.best_iteration)
        test_resids.append(model.predict(Xt, num_iteration=model.best_iteration))
        raw_score = rmse((base_oof[va] + oof_resid[va])[is_lat[va]], y[va][is_lat[va]])
        folds.append({"fold": fold, "best_iter": int(model.best_iteration), "rmse_w1": raw_score})
        print(f"fold{fold}: best_iter={model.best_iteration} rmse_w1={raw_score:.6f}", flush=True)

    test_resid = np.mean(np.column_stack(test_resids), axis=1).astype(np.float32)
    results = []
    for w in np.arange(-0.5, 1.51, 0.02):
        pred = base_oof + w * oof_resid
        score = rmse(pred[is_lat], y[is_lat])
        results.append({"method": f"raw_w{w:.2f}", "rmse": score, "w": float(w), "oof": pred, "test": base_test + w * test_resid})

    # Small smoothing sweep on the best residual blend.
    best = min(results, key=lambda r: r["rmse"])
    for win in [51, 101, 201, 301]:
        for poly in [2, 3]:
            pred = per_well_savgol(best["oof"], groups, win, poly)
            score = rmse(pred[is_lat], y[is_lat])
            results.append({
                "method": f"{best['method']}_savgol_w{win}_p{poly}",
                "rmse": score,
                "w": best["w"],
                "oof": pred,
                "test": best["test"],
            })

    results.sort(key=lambda r: r["rmse"])
    winner = results[0]
    np.save(ROOT / "experiments/oof/oof_stack_exp083.npy", winner["oof"].astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_stack_exp083.npy", winner["test"].astype(np.float32))
    np.save(ROOT / "experiments/oof/oof_lgbm_exp083_resid.npy", oof_resid)
    np.save(ROOT / "experiments/test_preds/test_lgbm_exp083_resid.npy", test_resid)
    payload = {
        "experiment_id": "exp083",
        "phase": "residual",
        "base": "exp082",
        "base_rmse": base_score,
        "base_mse": base_score ** 2,
        "lat_rmse": winner["rmse"],
        "lat_mse": winner["rmse"] ** 2,
        "winner": winner["method"],
        "folds": folds,
        "top": [{"method": r["method"], "rmse": r["rmse"], "mse": r["rmse"] ** 2, "w": r["w"]} for r in results[:20]],
    }
    (ROOT / "experiments/results/exp083.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:5000], flush=True)


if __name__ == "__main__":
    main()
