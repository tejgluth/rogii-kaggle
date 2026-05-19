"""exp072: focused postprocessing sweep for the clean exp063 XGBoost model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_train_meta():
    ys, groups, lateral = [], [], []
    for p in sorted(TRAIN_DIR.glob("*__horizontal_well.csv")):
        df = pd.read_csv(p, usecols=["TVT", "TVT_input"])
        ys.append(df["TVT"].to_numpy(np.float32))
        groups.append(np.full(len(df), p.stem.replace("__horizontal_well", ""), dtype=object))
        lateral.append(df["TVT_input"].isna().to_numpy())
    return np.concatenate(ys), np.concatenate(groups), np.concatenate(lateral)


def load_test_groups():
    groups = []
    for p in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
        n = len(pd.read_csv(p, usecols=["MD"]))
        groups.append(np.full(n, p.stem.replace("__horizontal_well", ""), dtype=object))
    return np.concatenate(groups)


def per_well_savgol(preds, groups, window, poly):
    out = preds.copy().astype(np.float32)
    for well in np.unique(groups):
        sel = groups == well
        seg = out[sel]
        win = min(window, len(seg))
        if win % 2 == 0:
            win -= 1
        if win >= poly + 2:
            out[sel] = savgol_filter(seg, win, poly)
    return out


def per_well_median(preds, groups, k):
    out = preds.copy().astype(np.float32)
    for well in np.unique(groups):
        sel = groups == well
        seg = out[sel]
        kk = min(k, len(seg))
        if kk % 2 == 0:
            kk -= 1
        if kk >= 3:
            out[sel] = medfilt(seg, kk)
    return out


def main():
    y, groups, lateral = load_train_meta()
    test_groups = load_test_groups()
    oof = np.load(ROOT / "experiments/oof/oof_xgboost_exp063.npy")
    test = np.load(ROOT / "experiments/test_preds/test_xgboost_exp063.npy")

    results = [{"method": "raw", "rmse": rmse(oof[lateral], y[lateral]), "oof": oof, "test": test}]
    print(f"raw lat_rmse={results[0]['rmse']:.6f} mse={results[0]['rmse'] ** 2:.6f}", flush=True)

    for win in [31, 51, 75, 101, 151, 201, 251, 301, 401, 501, 701]:
        for poly in [2, 3]:
            so = per_well_savgol(oof, groups, win, poly)
            st = per_well_savgol(test, test_groups, win, poly)
            score = rmse(so[lateral], y[lateral])
            results.append({"method": f"savgol_w{win}_p{poly}", "rmse": score, "oof": so, "test": st})
            print(f"savgol_w{win}_p{poly} lat_rmse={score:.6f} mse={score ** 2:.6f}", flush=True)

    base = min(results, key=lambda r: r["rmse"])
    for mk in [31, 51, 75, 101, 151]:
        mo = per_well_median(base["oof"], groups, mk)
        mt = per_well_median(base["test"], test_groups, mk)
        for win in [101, 201, 301, 401, 501]:
            for poly in [2, 3]:
                so = per_well_savgol(mo, groups, win, poly)
                st = per_well_savgol(mt, test_groups, win, poly)
                score = rmse(so[lateral], y[lateral])
                results.append({
                    "method": f"{base['method']}_med{mk}_savgol_w{win}_p{poly}",
                    "rmse": score,
                    "oof": so,
                    "test": st,
                })
                print(f"{results[-1]['method']} lat_rmse={score:.6f} mse={score ** 2:.6f}", flush=True)

    results.sort(key=lambda r: r["rmse"])
    winner = results[0]
    np.save(ROOT / "experiments/oof/oof_xgboost_exp072.npy", winner["oof"].astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_xgboost_exp072.npy", winner["test"].astype(np.float32))
    payload = {
        "experiment_id": "exp072",
        "phase": "postprocess",
        "base": "exp063",
        "winner": winner["method"],
        "lat_rmse": winner["rmse"],
        "lat_mse": winner["rmse"] ** 2,
        "top": [{"method": r["method"], "rmse": r["rmse"], "mse": r["rmse"] ** 2} for r in results[:20]],
    }
    (ROOT / "experiments/results/exp072.json").write_text(json.dumps(payload, indent=2))
    print("\nWINNER", json.dumps(payload["top"][0], indent=2), flush=True)


if __name__ == "__main__":
    main()
