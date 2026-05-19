"""exp078: XGBoost exp063 params with known heel rows downweighted."""
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
    known_weight = 0.10
    weights = np.where(is_lat, 1.0, known_weight).astype(np.float32)

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
        dtr = xgb.DMatrix(X[tr], label=y_d[tr], weight=weights[tr])
        dva = xgb.DMatrix(X[va], label=y_d[va], weight=weights[va])
        model = xgb.train(
            params,
            dtr,
            num_boost_round=8000,
            evals=[(dva, "val_weighted")],
            early_stopping_rounds=300,
            verbose_eval=0,
        )
        oof_delta[va] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        test_folds.append(model.predict(dte, iteration_range=(0, model.best_iteration + 1)))
        score = rmse(lk[va][is_lat[va]] + oof_delta[va][is_lat[va]], y_a[va][is_lat[va]])
        fold_rows.append({"fold": fold, "best_iter": int(model.best_iteration), "lat_rmse": score})
        print(f"fold{fold}: best_iter={model.best_iteration} lat_rmse={score:.6f}", flush=True)

    oof_abs = (lk + oof_delta).astype(np.float32)
    test_abs = (tlk + np.mean(np.column_stack(test_folds), axis=1)).astype(np.float32)
    score = rmse(oof_abs[is_lat], y_a[is_lat])
    np.save(ROOT / "experiments/oof/oof_xgboost_exp078.npy", oof_abs)
    np.save(ROOT / "experiments/test_preds/test_xgboost_exp078.npy", test_abs)
    payload = {
        "experiment_id": "exp078",
        "phase": "modeling",
        "model": "xgboost_exp063_known_weight_0.10",
        "known_weight": known_weight,
        "lat_rmse": score,
        "lat_mse": score ** 2,
        "folds": fold_rows,
        "params": params,
        "elapsed": time.time() - t0,
    }
    (ROOT / "experiments/results/exp078.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
