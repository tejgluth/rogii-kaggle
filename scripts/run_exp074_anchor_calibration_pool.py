"""exp074: apply valid anchor calibration to a pool of strong OOF/test predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_meta(directory, is_train):
    dfs = []
    for p in sorted(directory.glob("*__horizontal_well.csv")):
        cols = ["MD", "TVT_input"]
        if is_train:
            cols.append("TVT")
        df = pd.read_csv(p, usecols=cols)
        df["well"] = p.stem.replace("__horizontal_well", "")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def calibrate(pred, meta, mode, n_known, shrink):
    out = pred.copy().astype(np.float32)
    cursor = 0
    for _, df in meta.groupby("well", sort=False):
        n = len(df)
        seg = out[cursor:cursor + n]
        known = df["TVT_input"].notna().to_numpy()
        if known.any():
            residual = df.loc[known, "TVT_input"].to_numpy(np.float32) - seg[known]
            tail = residual[-n_known:]
            if mode == "median":
                bias = float(np.median(tail))
            elif mode == "mean":
                bias = float(np.mean(tail))
            elif mode == "last":
                bias = float(residual[-1])
            else:
                raise ValueError(mode)
            out[cursor:cursor + n] = seg + shrink * bias
        cursor += n
    return out


def main():
    train_meta = load_meta(TRAIN_DIR, True)
    test_meta = load_meta(TEST_DIR, False)
    y = train_meta["TVT"].to_numpy(np.float32)
    lateral = train_meta["TVT_input"].isna().to_numpy()

    pool = [
        ("xgboost_exp063", "experiments/oof/oof_xgboost_exp063.npy", "experiments/test_preds/test_xgboost_exp063.npy"),
        ("xgboost_exp072", "experiments/oof/oof_xgboost_exp072.npy", "experiments/test_preds/test_xgboost_exp072.npy"),
        ("stack_exp050", "experiments/oof/oof_stack_exp050.npy", "experiments/test_preds/test_stack_exp050.npy"),
        ("stack_exp056", "experiments/oof/oof_stack_exp056.npy", "experiments/test_preds/test_stack_exp056.npy"),
        ("stack_exp070", "experiments/oof/oof_stack_exp070.npy", "experiments/test_preds/test_stack_exp070.npy"),
    ]
    results = []
    for name, oof_path, test_path in pool:
        po = ROOT / oof_path
        pt = ROOT / test_path
        if not po.exists() or not pt.exists():
            continue
        oof = np.load(po)
        test = np.load(pt)
        base_score = rmse(oof[lateral], y[lateral])
        results.append({"base": name, "method": "raw", "rmse": base_score, "oof": oof, "test": test})
        print(f"{name} raw rmse={base_score:.6f} mse={base_score ** 2:.6f}", flush=True)
        for mode in ["median", "mean"]:
            for n_known in [50, 100, 200]:
                for shrink in np.arange(0.30, 0.81, 0.05):
                    co = calibrate(oof, train_meta, mode, n_known, float(shrink))
                    score = rmse(co[lateral], y[lateral])
                    if score < base_score:
                        ct = calibrate(test, test_meta, mode, n_known, float(shrink))
                        results.append({
                            "base": name,
                            "method": f"{mode}_n{n_known}_shrink{shrink:.2f}",
                            "rmse": score,
                            "oof": co,
                            "test": ct,
                        })

    results.sort(key=lambda r: r["rmse"])
    winner = results[0]
    np.save(ROOT / "experiments/oof/oof_stack_exp074.npy", winner["oof"].astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_stack_exp074.npy", winner["test"].astype(np.float32))
    payload = {
        "experiment_id": "exp074",
        "phase": "postprocess",
        "winner_base": winner["base"],
        "winner": winner["method"],
        "lat_rmse": winner["rmse"],
        "lat_mse": winner["rmse"] ** 2,
        "top": [
            {"base": r["base"], "method": r["method"], "rmse": r["rmse"], "mse": r["rmse"] ** 2}
            for r in results[:30]
        ],
    }
    (ROOT / "experiments/results/exp074.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
