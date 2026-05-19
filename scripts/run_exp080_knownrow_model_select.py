"""exp080: per-well model selection/blending using known-row residuals.

We have a clean calibrated candidate (exp076) and an aggressive calibrated candidate
(exp075). The aggressive candidate wins OOF but has stack-selection risk. For each
well, known TVT_input rows are available in train and test, so use their residuals
as a local validation signal to choose or blend candidates per well.
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
        cols = ["TVT_input"]
        if is_train:
            cols.append("TVT")
        df = pd.read_csv(p, usecols=cols)
        df["well"] = p.stem.replace("__horizontal_well", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def per_well_blend(preds, names, meta, mode, n_tail, temp, margin):
    out = np.zeros_like(next(iter(preds.values())), dtype=np.float32)
    choices = {name: 0 for name in names}
    cur = 0
    for _, df in meta.groupby("well", sort=False):
        n = len(df)
        known = df["TVT_input"].notna().to_numpy()
        scores = []
        for name in names:
            seg = preds[name][cur:cur + n]
            if known.any():
                resid = df.loc[known, "TVT_input"].to_numpy(np.float32) - seg[known]
                tail = resid[-min(n_tail, len(resid)):]
                score = float(np.sqrt(np.mean(tail * tail)))
            else:
                score = 0.0
            scores.append(score)
        scores = np.asarray(scores, dtype=np.float64)

        if mode == "choose":
            order = np.argsort(scores)
            if len(order) > 1 and scores[order[1]] - scores[order[0]] < margin:
                # If known-row evidence is close, average the top two instead of overreacting.
                weights = np.zeros(len(names), dtype=np.float64)
                weights[order[:2]] = 0.5
            else:
                weights = np.zeros(len(names), dtype=np.float64)
                weights[order[0]] = 1.0
        elif mode == "softmax":
            centered = scores - scores.min()
            weights = np.exp(-centered / max(temp, 1e-6))
            weights /= weights.sum()
        elif mode == "rank":
            ranks = np.empty_like(scores)
            ranks[np.argsort(scores)] = np.arange(len(scores), dtype=np.float64)
            weights = 1.0 / (1.0 + ranks)
            weights /= weights.sum()
        else:
            raise ValueError(mode)

        for wi, name in zip(weights, names):
            out[cur:cur + n] += wi * preds[name][cur:cur + n]
        choices[names[int(np.argmax(weights))]] += 1
        cur += n
    return out, choices


def main():
    train_meta = load_meta(TRAIN_DIR, True)
    test_meta = load_meta(TEST_DIR, False)
    y = train_meta["TVT"].to_numpy(np.float32)
    lateral = train_meta["TVT_input"].isna().to_numpy()

    candidates = {
        "clean076": (
            np.load(ROOT / "experiments/oof/oof_xgboost_exp076.npy"),
            np.load(ROOT / "experiments/test_preds/test_xgboost_exp076.npy"),
        ),
        "aggr075": (
            np.load(ROOT / "experiments/oof/oof_stack_exp075.npy"),
            np.load(ROOT / "experiments/test_preds/test_stack_exp075.npy"),
        ),
        "aggr074": (
            np.load(ROOT / "experiments/oof/oof_stack_exp074.npy"),
            np.load(ROOT / "experiments/test_preds/test_stack_exp074.npy"),
        ),
        "clean073": (
            np.load(ROOT / "experiments/oof/oof_xgboost_exp073.npy"),
            np.load(ROOT / "experiments/test_preds/test_xgboost_exp073.npy"),
        ),
    }
    names = list(candidates)
    oof_preds = {k: v[0] for k, v in candidates.items()}
    test_preds = {k: v[1] for k, v in candidates.items()}

    rows = []
    for name in names:
        score = rmse(oof_preds[name][lateral], y[lateral])
        rows.append({"method": name, "rmse": score, "mse": score ** 2})
        print(f"{name} rmse={score:.6f} mse={score ** 2:.6f}", flush=True)

    best = None
    for mode in ["choose", "softmax", "rank"]:
        for n_tail in [20, 50, 100, 150, 250, 500]:
            temps = [0.25, 0.5, 1.0, 2.0, 5.0] if mode == "softmax" else [1.0]
            margins = [0.0, 0.25, 0.5, 1.0, 2.0] if mode == "choose" else [0.0]
            for temp in temps:
                for margin in margins:
                    pred, choices = per_well_blend(oof_preds, names, train_meta, mode, n_tail, temp, margin)
                    score = rmse(pred[lateral], y[lateral])
                    row = {
                        "method": f"{mode}_tail{n_tail}_temp{temp:g}_margin{margin:g}",
                        "rmse": score,
                        "mse": score ** 2,
                        "choices": choices,
                    }
                    rows.append(row)
                    if best is None or score < best["rmse"]:
                        test_pred, test_choices = per_well_blend(test_preds, names, test_meta, mode, n_tail, temp, margin)
                        best = {**row, "oof": pred, "test": test_pred, "test_choices": test_choices}
                        print(f"new best {row}", flush=True)

    rows.sort(key=lambda x: x["rmse"])
    np.save(ROOT / "experiments/oof/oof_stack_exp080.npy", best["oof"].astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_stack_exp080.npy", best["test"].astype(np.float32))
    result = {
        "experiment_id": "exp080",
        "phase": "stacking",
        "lat_rmse": best["rmse"],
        "lat_mse": best["mse"],
        "winner": best["method"],
        "choices": best["choices"],
        "test_choices": best["test_choices"],
        "top": rows[:30],
    }
    (ROOT / "experiments/results/exp080.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)[:5000], flush=True)


if __name__ == "__main__":
    main()
