"""exp071: fast honest outer-fold audit for stack/residual validity.

All choices are fitted on outer-train wells and scored on untouched outer-validation
wells. This intentionally avoids the expensive smoothing cascade from exp050.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def log(msg):
    print(msg, flush=True)


def load_targets():
    ys, groups, lateral = [], [], []
    for p in sorted(TRAIN_DIR.glob("*__horizontal_well.csv")):
        df = pd.read_csv(p, usecols=["TVT", "TVT_input"])
        ys.append(df["TVT"].to_numpy(np.float32))
        groups.append(np.full(len(df), p.stem.replace("__horizontal_well", ""), dtype=object))
        lateral.append(df["TVT_input"].isna().to_numpy())
    return np.concatenate(ys), np.concatenate(groups), np.concatenate(lateral)


def load_base_pool(y, lateral, limit=12):
    pool = []
    for p in sorted((ROOT / "experiments/oof").glob("oof_*.npy")):
        name = p.stem.replace("oof_", "")
        if "stack_" in name:
            continue
        arr = np.load(p, allow_pickle=False)
        if arr.shape != y.shape:
            continue
        score = rmse(arr[lateral], y[lateral])
        if score < 12.0:
            pool.append((name, arr.astype(np.float64), score))
    pool.sort(key=lambda x: x[2])
    return pool[:limit]


def select_train_stack(OOF, y, train_mask, lateral, names):
    train_lat = train_mask & lateral
    member_scores = np.array([rmse(OOF[train_lat, j], y[train_lat]) for j in range(OOF.shape[1])])
    best_j = int(np.argmin(member_scores))
    best_pred = OOF[:, best_j]
    best_score = float(member_scores[best_j])
    best_desc = f"single:{names[best_j]}"

    # Top-3 simplex grid, fitted only on outer train.
    top = np.argsort(member_scores)[:3]
    for w0 in np.arange(0, 1.01, 0.05):
        for w1 in np.arange(0, 1.01 - w0, 0.05):
            w2 = 1.0 - w0 - w1
            w = np.array([w0, w1, w2], dtype=np.float64)
            pred = OOF[:, top] @ w
            score = rmse(pred[train_lat], y[train_lat])
            if score < best_score:
                best_score = score
                best_pred = pred
                best_desc = "grid:" + ",".join(f"{names[i]}={wi:.2f}" for i, wi in zip(top, w))

    # Greedy duplicate-averaging as in exp050, fitted only on train.
    selected = [best_j]
    score = float(member_scores[best_j])
    improved = True
    while improved and len(selected) < 20:
        improved = False
        for j in range(OOF.shape[1]):
            cand = selected + [j]
            pred = OOF[:, cand].mean(axis=1)
            cand_score = rmse(pred[train_lat], y[train_lat])
            if cand_score < score - 1e-5:
                selected = cand
                score = cand_score
                improved = True
    if score < best_score:
        best_score = score
        best_pred = OOF[:, selected].mean(axis=1)
        best_desc = "hill:" + ",".join(names[i] for i in selected)

    return best_pred.astype(np.float32), best_score, best_desc


def fit_residual_outer(X, residual, groups, tr, va):
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
        seed=7101,
        num_threads=-1,
    )
    train_groups = groups[tr]
    inner = GroupKFold(3)
    preds = []
    for k, (itr, iva) in enumerate(inner.split(X[tr], residual[tr], train_groups)):
        dtr = lgb.Dataset(X[tr][itr], label=residual[tr][itr])
        dva = lgb.Dataset(X[tr][iva], label=residual[tr][iva], reference=dtr)
        model = lgb.train(
            params,
            dtr,
            num_boost_round=1000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        log(f"    residual inner{k}: best_iter={model.best_iteration}")
        preds.append(model.predict(X[va], num_iteration=model.best_iteration))
    return np.mean(np.column_stack(preds), axis=1).astype(np.float32)


def main():
    t0 = time.time()
    y, groups, lateral = load_targets()
    cache = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
    X = cache["X"]
    if len(X) != len(y) or not np.array_equal(np.asarray(cache["groups"]), groups):
        raise RuntimeError("Feature cache row order does not match train CSV order")

    pool = load_base_pool(y, lateral)
    names = [p[0] for p in pool]
    OOF = np.column_stack([p[1] for p in pool])
    log(f"Loaded {len(pool)} base learners")
    for name, _, score in pool:
        log(f"  {name:28s} lat_rmse={score:.4f} mse={score ** 2:.4f}")

    audit_stack = np.zeros(len(y), dtype=np.float32)
    audit_resid = np.zeros(len(y), dtype=np.float32)
    rows = []
    outer = GroupKFold(5)

    for fold, (tr, va) in enumerate(outer.split(OOF, y, groups)):
        train_mask = np.zeros(len(y), dtype=bool)
        train_mask[tr] = True
        val_lat = lateral[va]
        stack_pred, train_score, desc = select_train_stack(OOF, y, train_mask, lateral, names)
        audit_stack[va] = stack_pred[va]
        stack_score = rmse(audit_stack[va][val_lat], y[va][val_lat])

        residual = (y - stack_pred).astype(np.float32)
        resid_pred = fit_residual_outer(X, residual, groups, tr, va)
        # Choose residual shrinkage on outer-train via OOF-like residual fit is not available;
        # report fixed conservative shrinkages on untouched validation.
        best_fold = (stack_score, 0.0)
        for w in [0.25, 0.5, 0.75, 1.0]:
            cand = stack_pred[va] + w * resid_pred
            score = rmse(cand[val_lat], y[va][val_lat])
            if score < best_fold[0]:
                best_fold = (score, w)
        audit_resid[va] = stack_pred[va] + best_fold[1] * resid_pred

        row = {
            "fold": fold,
            "train_stack_rmse": train_score,
            "val_stack_rmse": stack_score,
            "val_best_resid_rmse": best_fold[0],
            "val_best_resid_w_oracle": best_fold[1],
            "selector": desc,
        }
        rows.append(row)
        log(
            f"fold{fold}: stack_rmse={stack_score:.4f} "
            f"resid_rmse={best_fold[0]:.4f} resid_w={best_fold[1]:.2f} {desc[:140]}"
        )

    stack_rmse = rmse(audit_stack[lateral], y[lateral])
    resid_rmse = rmse(audit_resid[lateral], y[lateral])
    result = {
        "experiment_id": "exp071",
        "phase": "audit",
        "stack_outer_lat_rmse": stack_rmse,
        "stack_outer_lat_mse": stack_rmse ** 2,
        "residual_outer_lat_rmse": resid_rmse,
        "residual_outer_lat_mse": resid_rmse ** 2,
        "note": "Residual fold shrinkage is oracle-on-validation; use only as an upper-bound diagnostic.",
        "folds": rows,
        "elapsed": time.time() - t0,
    }
    np.save(ROOT / "experiments/oof/oof_stack_exp071_audit.npy", audit_stack)
    np.save(ROOT / "experiments/oof/oof_stack_exp071_resid_audit.npy", audit_resid)
    (ROOT / "experiments/results/exp071.json").write_text(json.dumps(result, indent=2))
    log(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
