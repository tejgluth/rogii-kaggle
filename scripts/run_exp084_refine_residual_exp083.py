"""exp084: refine exp083 residual shrinkage and apply matching test smoothing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_train_meta():
    y, groups, lateral = [], [], []
    for p in sorted(TRAIN_DIR.glob("*__horizontal_well.csv")):
        df = pd.read_csv(p, usecols=["TVT", "TVT_input"])
        well = p.stem.replace("__horizontal_well", "")
        y.append(df["TVT"].to_numpy(np.float32))
        lateral.append(df["TVT_input"].isna().to_numpy())
        groups.append(np.full(len(df), well, dtype=object))
    return np.concatenate(y), np.concatenate(groups), np.concatenate(lateral)


def load_test_groups():
    groups = []
    for p in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
        n = len(pd.read_csv(p, usecols=["MD"]))
        groups.append(np.full(n, p.stem.replace("__horizontal_well", ""), dtype=object))
    return np.concatenate(groups)


def per_well_savgol(pred, groups, window, poly):
    if window <= 0:
        return pred.copy().astype(np.float32)
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
    y, train_groups, lateral = load_train_meta()
    test_groups = load_test_groups()
    base_oof = np.load(ROOT / "experiments/oof/oof_stack_exp082.npy")
    base_test = np.load(ROOT / "experiments/test_preds/test_stack_exp082.npy")
    resid_oof = np.load(ROOT / "experiments/oof/oof_lgbm_exp083_resid.npy")
    resid_test = np.load(ROOT / "experiments/test_preds/test_lgbm_exp083_resid.npy")

    results = []
    for w in np.arange(0.8, 3.01, 0.02):
        raw_oof = base_oof + w * resid_oof
        raw_test = base_test + w * resid_test
        for win in [0, 31, 51, 75, 101, 151, 201, 251, 301]:
            for poly in ([2, 3] if win > 0 else [0]):
                oof = per_well_savgol(raw_oof, train_groups, win, poly)
                score = rmse(oof[lateral], y[lateral])
                results.append({
                    "method": "raw" if win == 0 else f"savgol_w{win}_p{poly}",
                    "w": float(w),
                    "rmse": score,
                    "oof": oof,
                    "test_raw": raw_test,
                    "win": win,
                    "poly": poly,
                })
        if int(round((w - 0.8) / 0.02)) % 25 == 0:
            best_so_far = min(results, key=lambda r: r["rmse"])
            print(
                f"w={w:.2f} best={best_so_far['rmse']:.6f} "
                f"{best_so_far['method']} bw={best_so_far['w']:.2f}",
                flush=True,
            )

    results.sort(key=lambda r: r["rmse"])
    winner = results[0]
    test = per_well_savgol(winner["test_raw"], test_groups, winner["win"], winner["poly"])
    np.save(ROOT / "experiments/oof/oof_stack_exp084.npy", winner["oof"].astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_stack_exp084.npy", test.astype(np.float32))
    payload = {
        "experiment_id": "exp084",
        "phase": "postprocess",
        "base": "exp082 + exp083 residual",
        "lat_rmse": winner["rmse"],
        "lat_mse": winner["rmse"] ** 2,
        "winner": winner["method"],
        "residual_weight": winner["w"],
        "top": [
            {"method": r["method"], "w": r["w"], "rmse": r["rmse"], "mse": r["rmse"] ** 2}
            for r in results[:30]
        ],
    }
    (ROOT / "experiments/results/exp084.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:5000], flush=True)


if __name__ == "__main__":
    main()
