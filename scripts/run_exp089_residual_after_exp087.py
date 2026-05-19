"""exp089: fourth-stage LGBM residual correction after exp087."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    cache = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
    X = cache["X"]
    Xt = cache["Xt"]
    y = cache["y_abs"]
    is_lat = cache["is_lateral"]
    groups = np.asarray(cache["groups"])
    base_oof = np.load(ROOT / "experiments/oof/oof_stack_exp087.npy")
    base_test = np.load(ROOT / "experiments/test_preds/test_stack_exp087.npy")
    residual = (y - base_oof).astype(np.float32)
    base_score = rmse(base_oof[is_lat], y[is_lat])
    print(f"base exp087 rmse={base_score:.6f} mse={base_score ** 2:.6f}", flush=True)

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.015,
        num_leaves=31,
        min_data_in_leaf=500,
        feature_fraction=0.65,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=4.0,
        lambda_l2=8.0,
        verbose=-1,
        seed=8901,
        num_threads=-1,
    )
    oof_resid = np.zeros(len(y), dtype=np.float32)
    test_resids = []
    folds = []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(X, residual, groups)):
        dtr = lgb.Dataset(X[tr], label=residual[tr])
        dva = lgb.Dataset(X[va], label=residual[va], reference=dtr)
        model = lgb.train(
            params,
            dtr,
            num_boost_round=2500,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof_resid[va] = model.predict(X[va], num_iteration=model.best_iteration)
        test_resids.append(model.predict(Xt, num_iteration=model.best_iteration))
        score = rmse((base_oof[va] + oof_resid[va])[is_lat[va]], y[va][is_lat[va]])
        folds.append({"fold": fold, "best_iter": int(model.best_iteration), "rmse_w1": score})
        print(f"fold{fold}: best_iter={model.best_iteration} rmse_w1={score:.6f}", flush=True)

    test_resid = np.mean(np.column_stack(test_resids), axis=1).astype(np.float32)
    results = []
    for w in np.arange(-0.5, 2.51, 0.01):
        pred = base_oof + w * oof_resid
        score = rmse(pred[is_lat], y[is_lat])
        results.append((score, float(w), pred, base_test + w * test_resid))
    results.sort(key=lambda x: x[0])
    score, w, pred, test = results[0]
    np.save(ROOT / "experiments/oof/oof_stack_exp089.npy", pred.astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_stack_exp089.npy", test.astype(np.float32))
    np.save(ROOT / "experiments/oof/oof_lgbm_exp089_resid.npy", oof_resid)
    np.save(ROOT / "experiments/test_preds/test_lgbm_exp089_resid.npy", test_resid)
    payload = {
        "experiment_id": "exp089",
        "phase": "residual",
        "base": "exp087",
        "base_rmse": base_score,
        "base_mse": base_score ** 2,
        "lat_rmse": score,
        "lat_mse": score ** 2,
        "residual_weight": w,
        "folds": folds,
        "top": [{"rmse": s, "mse": s * s, "w": ww} for s, ww, _, __ in results[:30]],
    }
    (ROOT / "experiments/results/exp089.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:5000], flush=True)


if __name__ == "__main__":
    main()
