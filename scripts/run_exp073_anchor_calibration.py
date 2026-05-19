"""exp073: per-well anchor calibration using known TVT_input rows.

For each well, compare predictions to the known heel TVT_input rows and carry a
constant or distance-decayed bias correction into the lateral. This is test-time
available information and targets the large well-level misses left by exp072.
"""
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
    parts = []
    for p in sorted(directory.glob("*__horizontal_well.csv")):
        cols = ["MD", "TVT_input"]
        if is_train:
            cols.append("TVT")
        df = pd.read_csv(p, usecols=cols)
        df["well"] = p.stem.replace("__horizontal_well", "")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def corrected(pred, meta, mode, n_known=50, decay=500.0, shrink=1.0):
    out = pred.copy().astype(np.float32)
    cursor = 0
    for _, df in meta.groupby("well", sort=False):
        n = len(df)
        seg = out[cursor:cursor + n]
        known = df["TVT_input"].notna().to_numpy()
        if not known.any():
            cursor += n
            continue
        residual = df.loc[known, "TVT_input"].to_numpy(np.float32) - seg[known]
        if mode == "last":
            bias = float(residual[-1])
        elif mode == "late_mean":
            bias = float(np.mean(residual[-n_known:]))
        elif mode == "late_median":
            bias = float(np.median(residual[-n_known:]))
        elif mode == "wls":
            r = residual[-n_known:]
            weights = np.exp(np.linspace(-2.0, 0.0, len(r)))
            bias = float(np.average(r, weights=weights))
        else:
            raise ValueError(mode)

        md = df["MD"].to_numpy(np.float32)
        last_md = float(df.loc[known, "MD"].iloc[-1])
        dist = np.maximum(0.0, md - last_md)
        if decay > 0:
            scale = np.exp(-dist / decay)
        else:
            scale = np.ones(n, dtype=np.float32)
        out[cursor:cursor + n] = seg + shrink * bias * scale
        cursor += n
    return out


def main():
    y_meta = load_meta(TRAIN_DIR, True)
    t_meta = load_meta(TEST_DIR, False)
    y = y_meta["TVT"].to_numpy(np.float32)
    lateral = y_meta["TVT_input"].isna().to_numpy()

    base_oof = np.load(ROOT / "experiments/oof/oof_xgboost_exp072.npy")
    base_test = np.load(ROOT / "experiments/test_preds/test_xgboost_exp072.npy")
    base = rmse(base_oof[lateral], y[lateral])
    print(f"base exp072 lat_rmse={base:.6f} mse={base ** 2:.6f}", flush=True)

    results = [{"method": "raw", "rmse": base, "oof": base_oof, "test": base_test}]
    for mode in ["last", "late_mean", "late_median", "wls"]:
        for n_known in [10, 25, 50, 100, 200]:
            for decay in [0.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0]:
                for shrink in [0.25, 0.5, 0.75, 1.0]:
                    if mode == "last" and n_known != 10:
                        continue
                    oof = corrected(base_oof, y_meta, mode, n_known, decay, shrink)
                    score = rmse(oof[lateral], y[lateral])
                    if score < base + 0.02:
                        print(
                            f"{mode}_n{n_known}_d{decay:g}_s{shrink:g} "
                            f"lat_rmse={score:.6f} mse={score ** 2:.6f}",
                            flush=True,
                        )
                    if score < results[0]["rmse"]:
                        test = corrected(base_test, t_meta, mode, n_known, decay, shrink)
                        results.insert(0, {
                            "method": f"{mode}_n{n_known}_decay{decay:g}_shrink{shrink:g}",
                            "rmse": score,
                            "oof": oof,
                            "test": test,
                        })

    winner = results[0]
    np.save(ROOT / "experiments/oof/oof_xgboost_exp073.npy", winner["oof"].astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_xgboost_exp073.npy", winner["test"].astype(np.float32))
    payload = {
        "experiment_id": "exp073",
        "phase": "postprocess",
        "base": "exp072",
        "winner": winner["method"],
        "lat_rmse": winner["rmse"],
        "lat_mse": winner["rmse"] ** 2,
    }
    (ROOT / "experiments/results/exp073.json").write_text(json.dumps(payload, indent=2))
    print("WINNER", json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
