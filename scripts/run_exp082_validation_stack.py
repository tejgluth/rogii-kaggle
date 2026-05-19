"""exp082: validation-optimized stack over all saved strong OOF/test predictions.

This intentionally optimizes lateral validation MSE aggressively. It is useful for
finding the lowest local validation score and a leaderboard-probing submission.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_target():
    ys, lateral = [], []
    for p in sorted(TRAIN_DIR.glob("*__horizontal_well.csv")):
        df = pd.read_csv(p, usecols=["TVT", "TVT_input"])
        ys.append(df["TVT"].to_numpy(np.float32))
        lateral.append(df["TVT_input"].isna().to_numpy())
    return np.concatenate(ys), np.concatenate(lateral)


def paired_test_path(oof_path):
    name = oof_path.name.replace("oof_", "test_")
    return ROOT / "experiments/test_preds" / name


def main():
    y, lateral = load_target()
    rows = []
    candidates = []
    for oof_path in sorted((ROOT / "experiments/oof").glob("oof_*.npy")):
        arr = np.load(oof_path)
        if arr.shape != y.shape:
            continue
        test_path = paired_test_path(oof_path)
        if not test_path.exists():
            continue
        score = rmse(arr[lateral], y[lateral])
        name = oof_path.stem.replace("oof_", "")
        if score < 3.2:
            candidates.append((name, arr.astype(np.float64), np.load(test_path).astype(np.float64), score))
            rows.append({"name": name, "rmse": score, "mse": score ** 2})
    candidates.sort(key=lambda x: x[3])
    print("candidates", len(candidates), flush=True)
    for r in sorted(rows, key=lambda x: x["rmse"])[:40]:
        print(f"{r['name']:32s} rmse={r['rmse']:.6f} mse={r['mse']:.6f}", flush=True)

    names = [c[0] for c in candidates]
    O = np.column_stack([c[1] for c in candidates])
    T = np.column_stack([c[2] for c in candidates])
    yl = y[lateral].astype(np.float64)
    Ol = O[lateral]

    results = []

    # NNLS with intercept, then normalized positive weights for stable test scale.
    coef, _ = nnls(np.column_stack([Ol, np.ones(len(yl))]), yl)
    pred = np.column_stack([O, np.ones(len(y))]) @ coef
    test = np.column_stack([T, np.ones(len(T))]) @ coef
    score = rmse(pred[lateral], y[lateral])
    results.append({"method": "nnls_intercept", "rmse": score, "oof": pred, "test": test,
                    "info": {"coef": dict(zip(names + ["intercept"], coef.tolist()))}})
    print(f"nnls_intercept rmse={score:.6f} mse={score ** 2:.6f}", flush=True)

    coef2, _ = nnls(Ol, yl)
    w = coef2 / (coef2.sum() + 1e-12)
    pred = O @ w
    test = T @ w
    score = rmse(pred[lateral], y[lateral])
    results.append({"method": "nnls_simplex", "rmse": score, "oof": pred, "test": test,
                    "info": {"weights": dict(zip(names, w.tolist()))}})
    print(f"nnls_simplex rmse={score:.6f} mse={score ** 2:.6f}", flush=True)

    # Greedy duplicate averaging; tends to exploit complementary residuals.
    selected = [0]
    best = candidates[0][3]
    improved = True
    while improved and len(selected) < 80:
        improved = False
        for j in range(len(candidates)):
            sel = selected + [j]
            pred = O[:, sel].mean(axis=1)
            score = rmse(pred[lateral], y[lateral])
            if score < best - 1e-6:
                best = score
                selected = sel
                improved = True
    pred = O[:, selected].mean(axis=1)
    test = T[:, selected].mean(axis=1)
    score = rmse(pred[lateral], y[lateral])
    results.append({"method": "hillclimb_duplicate_avg", "rmse": score, "oof": pred, "test": test,
                    "info": {"members": [names[i] for i in selected]}})
    print(f"hillclimb_duplicate_avg rmse={score:.6f} mse={score ** 2:.6f}", flush=True)

    # Fine grid around top two/three validation candidates.
    top = list(range(min(4, len(candidates))))
    for step in [0.01, 0.02, 0.05]:
        best_grid = None
        if len(top) >= 2:
            for w0 in np.arange(0, 1 + 1e-9, step):
                rem1 = 1.0 - w0
                for w1 in np.arange(0, rem1 + 1e-9, step):
                    rem2 = rem1 - w1
                    if len(top) == 2:
                        weights = np.array([w0, 1.0 - w0])
                    elif len(top) == 3:
                        weights = np.array([w0, w1, rem2])
                    else:
                        for w2 in np.arange(0, rem2 + 1e-9, step):
                            weights = np.array([w0, w1, w2, rem2 - w2])
                            pred = O[:, top] @ weights
                            score = rmse(pred[lateral], y[lateral])
                            if best_grid is None or score < best_grid[0]:
                                best_grid = (score, weights.copy())
                        continue
                    pred = O[:, top[:len(weights)]] @ weights
                    score = rmse(pred[lateral], y[lateral])
                    if best_grid is None or score < best_grid[0]:
                        best_grid = (score, weights.copy())
        if best_grid is not None:
            weights = best_grid[1]
            pred = O[:, top[:len(weights)]] @ weights
            test = T[:, top[:len(weights)]] @ weights
            results.append({"method": f"grid_top{len(weights)}_step{step}", "rmse": best_grid[0],
                            "oof": pred, "test": test,
                            "info": {"weights": dict(zip([names[i] for i in top[:len(weights)]], weights.tolist()))}})
            print(f"grid step={step} rmse={best_grid[0]:.6f} mse={best_grid[0] ** 2:.6f}", flush=True)

    results.sort(key=lambda x: x["rmse"])
    winner = results[0]
    np.save(ROOT / "experiments/oof/oof_stack_exp082.npy", winner["oof"].astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_stack_exp082.npy", winner["test"].astype(np.float32))
    payload = {
        "experiment_id": "exp082",
        "phase": "stacking",
        "lat_rmse": winner["rmse"],
        "lat_mse": winner["rmse"] ** 2,
        "winner": winner["method"],
        "winner_info": winner["info"],
        "top_results": [
            {"method": r["method"], "rmse": r["rmse"], "mse": r["rmse"] ** 2, "info": r["info"]}
            for r in results[:10]
        ],
        "candidate_scores": sorted(rows, key=lambda x: x["rmse"])[:50],
    }
    (ROOT / "experiments/results/exp082.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:6000], flush=True)


if __name__ == "__main__":
    main()
