#!/usr/bin/env python3
"""exp117: leak-safe OOF blend/postprocess sweep for exp116 cached models.

This does not train a new base model. It uses only GroupKFold OOF predictions
from the locally trained exp116 fast cache, selects blend/postprocess settings
on tuning wells, and reports scores on separate held-out wells.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import root_mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "local_runs" / "exp116_fast_cache"
OUT = ROOT / "experiments" / "results" / "exp117_oof_postprocess_sweep.json"

TEST_WELLS = {"000d7d20", "00bbac68", "00e12e8b"}
SEEDS = [117, 118, 119, 120, 121, 122, 123]
HOLDOUT_FRACTION = 0.20


def smooth_by_well(values: np.ndarray, wells: np.ndarray, win: int, poly: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if win <= 0:
        return values.copy()
    out = values.copy()
    for well in pd.unique(wells):
        idx = np.flatnonzero(wells == well)
        n = len(idx)
        w = min(int(win), n if n % 2 else n - 1)
        if w >= poly + 2:
            out[idx] = savgol_filter(values[idx], w, poly).astype(np.float32)
    return out


def apply_pp(md_delta: np.ndarray, pf_delta: np.ndarray, md_since: np.ndarray, alpha: float, tau: float, w_pf: float) -> np.ndarray:
    d = md_delta * (1.0 - w_pf) + pf_delta * w_pf
    if tau > 0:
        d = d * (1.0 - np.exp(-np.maximum(md_since, 0.0) / tau))
    return d * alpha


def rmse(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return float(root_mean_squared_error(y[mask], pred[mask]))


def make_pred(
    oof: pd.DataFrame,
    base: np.ndarray,
    pf_delta: np.ndarray,
    md_since: np.ndarray,
    wells: np.ndarray,
    params: dict,
) -> np.ndarray:
    cat_w = float(params["cat_w"])
    md_delta = (1.0 - cat_w) * oof["lightgbm-1"].to_numpy(np.float32) + cat_w * oof["catboost-1"].to_numpy(np.float32)
    pred = base + apply_pp(
        md_delta,
        pf_delta,
        md_since,
        float(params["alpha"]),
        float(params["tau"]),
        float(params["w_pf"]),
    )
    return smooth_by_well(pred, wells, int(params["sg_win"]), int(params["sg_poly"]))


def main() -> None:
    cols = ["well", "last_known_tvt", "pf_ancc", "md_since", "target"]
    train = pd.read_csv(RUN / "train.csv", usecols=cols, dtype={"well": str})
    oof = pd.read_csv(RUN / "oof_preds.csv")

    wells = train["well"].astype(str).to_numpy()
    base = train["last_known_tvt"].to_numpy(np.float32)
    target = (base + train["target"].to_numpy(np.float32)).astype(np.float32)
    pf_delta = (train["pf_ancc"].to_numpy(np.float32) - base).astype(np.float32)
    md_since = train["md_since"].to_numpy(np.float32)
    score_like = np.isfinite(target) & np.isfinite(base) & (md_since > 0) & ~np.isin(wells, list(TEST_WELLS))
    eligible_wells = np.array(sorted(pd.unique(wells[score_like]).astype(str)))

    current = {
        "cat_w": 0.8889999999999999,
        "alpha": 0.98,
        "tau": 100.0,
        "w_pf": 0.12,
        "sg_win": 17,
        "sg_poly": 3,
    }

    grid = []
    # Targeted grid around the submitted recipe. This avoids overfitting and
    # keeps the sweep cheap enough to repeat across several well holdouts.
    for cat_w in [0.85, 0.889, 0.925, 0.95, 1.0]:
        for alpha in [0.95, 0.98, 1.00]:
            for tau in [75.0, 100.0, 150.0]:
                for w_pf in [0.10, 0.12, 0.16]:
                    for sg_win, sg_poly in [(17, 3), (51, 3), (101, 3)]:
                        grid.append(
                            {
                                "cat_w": float(cat_w),
                                "alpha": float(alpha),
                                "tau": float(tau),
                                "w_pf": float(w_pf),
                                "sg_win": int(sg_win),
                                "sg_poly": int(sg_poly),
                            }
                        )

    current_pred = make_pred(oof, base, pf_delta, md_since, wells, current)
    split_reports = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(eligible_wells)
        n_holdout = max(1, int(round(len(shuffled) * HOLDOUT_FRACTION)))
        holdout_wells = set(map(str, shuffled[:n_holdout]))
        holdout = score_like & np.isin(wells, list(holdout_wells))
        tune = score_like & ~np.isin(wells, list(holdout_wells))

        split_reports.append(
            {
                "seed": seed,
                "holdout_rows": int(holdout.sum()),
                "holdout_wells": int(len(holdout_wells)),
                "selected": None,
                "selected_tune_rmse": float("inf"),
                "selected_holdout_rmse": None,
                "current_tune_rmse": rmse(target, current_pred, tune),
                "current_holdout_rmse": rmse(target, current_pred, holdout),
                "_tune_mask": tune,
                "_holdout_mask": holdout,
            }
        )

    for i, params in enumerate(grid):
        if i % 100 == 0:
            print(f"candidate {i}/{len(grid)}", flush=True)
        pred = make_pred(oof, base, pf_delta, md_since, wells, params)
        for rec in split_reports:
            tune_score = rmse(target, pred, rec["_tune_mask"])
            if tune_score < rec["selected_tune_rmse"]:
                rec["selected_tune_rmse"] = float(tune_score)
                rec["selected_holdout_rmse"] = rmse(target, pred, rec["_holdout_mask"])
                rec["selected"] = params

    for rec in split_reports:
        rec.pop("_tune_mask", None)
        rec.pop("_holdout_mask", None)
        print(
            f"seed={rec['seed']} current_holdout={rec['current_holdout_rmse']:.5f} "
            f"selected_holdout={rec['selected_holdout_rmse']:.5f} selected={rec['selected']}",
            flush=True,
        )

    selected_h = np.array([r["selected_holdout_rmse"] for r in split_reports], dtype=np.float64)
    current_h = np.array([r["current_holdout_rmse"] for r in split_reports], dtype=np.float64)
    payload = {
        "experiment_id": "exp117",
        "route": "oof_only_blend_pp_sweep",
        "grid_size": len(grid),
        "seeds": SEEDS,
        "current_params": current,
        "current_all_oof_rmse": rmse(target, current_pred, score_like),
        "split_reports": split_reports,
        "selected_holdout_mean": float(selected_h.mean()),
        "selected_holdout_std": float(selected_h.std(ddof=0)),
        "current_holdout_mean": float(current_h.mean()),
        "current_holdout_std": float(current_h.std(ddof=0)),
        "mean_holdout_gain": float(current_h.mean() - selected_h.mean()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2)[:6000])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
