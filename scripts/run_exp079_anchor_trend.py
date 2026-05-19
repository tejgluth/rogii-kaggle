"""exp079: anchor calibration with optional known-row residual trend.

Constant per-well anchor correction helped substantially. This experiment tests
whether residual slope over the known heel rows can improve lateral extrapolation.
All signals used here are available for train and test: predictions, TVT_input on
known rows, MD, and well grouping.
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
    frames = []
    for p in sorted(directory.glob("*__horizontal_well.csv")):
        cols = ["MD", "TVT_input"]
        if is_train:
            cols.append("TVT")
        df = pd.read_csv(p, usecols=cols)
        df["well"] = p.stem.replace("__horizontal_well", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def tail_bias(resid, mode):
    if mode == "mean":
        return float(np.mean(resid))
    if mode == "median":
        return float(np.median(resid))
    if mode == "last":
        return float(resid[-1])
    raise ValueError(mode)


def tail_slope(md, resid, mode):
    if len(resid) < 2:
        return 0.0
    if mode == "none":
        return 0.0
    if mode == "last_first":
        denom = float(md[-1] - md[0])
        return 0.0 if abs(denom) < 1e-6 else float((resid[-1] - resid[0]) / denom)
    if mode == "linear":
        x = md.astype(np.float64)
        y = resid.astype(np.float64)
        x = x - x.mean()
        denom = float(np.dot(x, x))
        return 0.0 if denom < 1e-9 else float(np.dot(x, y - y.mean()) / denom)
    raise ValueError(mode)


def apply_calibration(pred, meta, cfg):
    out = pred.copy().astype(np.float32)
    cur = 0
    for _, df in meta.groupby("well", sort=False):
        n_rows = len(df)
        seg = out[cur:cur + n_rows]
        known = df["TVT_input"].notna().to_numpy()
        if known.any():
            md = df["MD"].to_numpy(np.float32)
            known_md = md[known]
            resid = df.loc[known, "TVT_input"].to_numpy(np.float32) - seg[known]
            n_tail = min(cfg["n_tail"], len(resid))
            tail_r = resid[-n_tail:]
            tail_md = known_md[-n_tail:]
            bias = tail_bias(tail_r, cfg["bias_mode"])
            slope = tail_slope(tail_md, tail_r, cfg["slope_mode"])
            dist = np.maximum(0.0, md - float(known_md[-1]))
            trend = slope * dist
            if cfg["clip"] > 0:
                trend = np.clip(trend, -cfg["clip"], cfg["clip"])
            corr = cfg["bias_shrink"] * bias + cfg["slope_shrink"] * trend
            out[cur:cur + n_rows] = seg + corr.astype(np.float32)
        cur += n_rows
    return out


def main():
    train_meta = load_meta(TRAIN_DIR, True)
    test_meta = load_meta(TEST_DIR, False)
    y = train_meta["TVT"].to_numpy(np.float32)
    lateral = train_meta["TVT_input"].isna().to_numpy()

    bases = [
        ("clean_exp072", "experiments/oof/oof_xgboost_exp072.npy", "experiments/test_preds/test_xgboost_exp072.npy"),
        ("aggressive_exp070", "experiments/oof/oof_stack_exp070.npy", "experiments/test_preds/test_stack_exp070.npy"),
    ]

    all_results = []
    best_payload = None
    for base_name, oof_path, test_path in bases:
        oof = np.load(ROOT / oof_path)
        test = np.load(ROOT / test_path)
        base_score = rmse(oof[lateral], y[lateral])
        print(f"{base_name} raw rmse={base_score:.6f} mse={base_score ** 2:.6f}", flush=True)

        configs = []
        if base_name == "clean_exp072":
            bias_shrinks = [0.52, 0.56, 0.58, 0.60, 0.64]
            n_tails = [100, 120, 140, 160, 180]
        else:
            bias_shrinks = [0.30, 0.34, 0.36, 0.38, 0.42]
            n_tails = [100, 140, 160, 180, 220]

        for bias_mode in ["median", "mean"]:
            for n_tail in n_tails:
                for bias_shrink in bias_shrinks:
                    configs.append({
                        "bias_mode": bias_mode,
                        "n_tail": n_tail,
                        "bias_shrink": bias_shrink,
                        "slope_mode": "none",
                        "slope_shrink": 0.0,
                        "clip": 0.0,
                    })
                    for slope_mode in ["linear", "last_first"]:
                        for slope_shrink in [-0.5, -0.25, 0.25, 0.5, 0.75, 1.0]:
                            for clip in [5.0, 10.0, 20.0, 40.0]:
                                configs.append({
                                    "bias_mode": bias_mode,
                                    "n_tail": n_tail,
                                    "bias_shrink": bias_shrink,
                                    "slope_mode": slope_mode,
                                    "slope_shrink": slope_shrink,
                                    "clip": clip,
                                })

        for i, cfg in enumerate(configs):
            pred = apply_calibration(oof, train_meta, cfg)
            score = rmse(pred[lateral], y[lateral])
            row = {"base": base_name, "rmse": score, "mse": score ** 2, **cfg}
            all_results.append(row)
            if best_payload is None or score < best_payload["rmse"]:
                test_pred = apply_calibration(test, test_meta, cfg)
                best_payload = {"base": base_name, "rmse": score, "oof": pred, "test": test_pred, "cfg": cfg}
                print(f"new best {base_name} rmse={score:.6f} mse={score ** 2:.6f} cfg={cfg}", flush=True)

    all_results.sort(key=lambda x: x["rmse"])
    winner = best_payload
    np.save(ROOT / "experiments/oof/oof_stack_exp079.npy", winner["oof"].astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_stack_exp079.npy", winner["test"].astype(np.float32))
    result = {
        "experiment_id": "exp079",
        "phase": "postprocess",
        "winner_base": winner["base"],
        "lat_rmse": winner["rmse"],
        "lat_mse": winner["rmse"] ** 2,
        "winner_config": winner["cfg"],
        "top": all_results[:30],
    }
    (ROOT / "experiments/results/exp079.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
