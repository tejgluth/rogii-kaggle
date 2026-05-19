"""exp077: XGBoost exp063 params trained only on scored lateral rows.

The competition scores lateral TVT rows only. Previous delta models trained on all
rows, including known heel rows. This run keeps GroupKFold by well but filters the
training and validation DMatrix to lateral rows only.
"""
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    t0 = time.time()
    cache = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
    X = cache["X"]
    y_d = cache["y_delta"]
    y_a = cache["y_abs"]
    lk = cache["last_known"]
    is_lat = cache["is_lateral"]
    groups = cache["groups"]
    Xt = cache["Xt"]
    tlk = cache["test_last_known"]

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=5,
        min_child_weight=25,
        subsample=0.85,
        colsample_bytree=0.65,
        reg_lambda=3.0,
        learning_rate=0.015,
        seed=101,
        device="cuda",
        tree_method="hist",
        max_bin=255,
        verbosity=0,
    )
    gkf = GroupKFold(5)
    oof_delta = np.zeros(len(y_d), np.float32)
    test_folds = []
    fold_rows = []
    dte = xgb.DMatrix(Xt)

    for fold, (tr, va) in enumerate(gkf.split(X, y_d, groups)):
        tr_lat = tr[is_lat[tr]]
        va_lat = va[is_lat[va]]
        dtr = xgb.DMatrix(X[tr_lat], label=y_d[tr_lat])
        dva = xgb.DMatrix(X[va_lat], label=y_d[va_lat])
        model = xgb.train(
            params,
            dtr,
            num_boost_round=8000,
            evals=[(dva, "lat_val")],
            early_stopping_rounds=300,
            verbose_eval=0,
        )
        oof_delta[va_lat] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        # Known rows are unscored; keep their absolute prediction anchored for diagnostics.
        known_va = va[~is_lat[va]]
        oof_delta[known_va] = y_d[known_va]
        test_folds.append(model.predict(dte, iteration_range=(0, model.best_iteration + 1)))
        score = rmse(lk[va_lat] + oof_delta[va_lat], y_a[va_lat])
        fold_rows.append({"fold": fold, "best_iter": int(model.best_iteration), "lat_rmse": score})
        print(f"fold{fold}: best_iter={model.best_iteration} lat_rmse={score:.6f}", flush=True)

    oof_abs = (lk + oof_delta).astype(np.float32)
    test_abs = (tlk + np.mean(np.column_stack(test_folds), axis=1)).astype(np.float32)
    score = rmse(oof_abs[is_lat], y_a[is_lat])
    np.save(ROOT / "experiments/oof/oof_xgboost_exp077.npy", oof_abs)
    np.save(ROOT / "experiments/test_preds/test_xgboost_exp077.npy", test_abs)
    payload = {
        "experiment_id": "exp077",
        "phase": "modeling",
        "model": "xgboost_lateral_only_exp063_params",
        "lat_rmse": score,
        "lat_mse": score ** 2,
        "folds": fold_rows,
        "params": params,
        "elapsed": time.time() - t0,
    }
    (ROOT / "experiments/results/exp077.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
