#!/usr/bin/env python3
"""exp120: leak-aware stack/postprocess sweep for the v10 artifact package.

Uses only the public v10 artifact OOF diagnostics. Base model predictions are
already out-of-fold; this script adds a second GroupKFold-by-well layer for
stack/postprocess selection so we do not tune on the same wells used to fit the
stack weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "local_runs" / "v10_artifacts"
OUT = ROOT / "experiments" / "results" / "exp120_v10_stack_sweep.json"


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y, pred))


def fit_predict_linear(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    method: str,
    alpha: float = 0.0,
    positive: bool = False,
    fit_intercept: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    oof = np.zeros(len(y), dtype=np.float32)
    fold_coefs = []
    fold_intercepts = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups=groups):
        if method == "saved":
            raise RuntimeError("saved does not fit")
        if method == "nnls":
            coef, _ = nnls(X[tr].astype(np.float64), y[tr].astype(np.float64))
            intercept = 0.0
        elif method == "ridge":
            model = Ridge(alpha=float(alpha), fit_intercept=fit_intercept, positive=positive)
            model.fit(X[tr], y[tr])
            coef = model.coef_.astype(np.float64)
            intercept = float(model.intercept_)
        else:
            raise ValueError(method)
        oof[va] = (X[va].astype(np.float64) @ coef + intercept).astype(np.float32)
        fold_coefs.append(coef.astype(float).tolist())
        fold_intercepts.append(intercept)

    if method == "nnls":
        final_coef, _ = nnls(X.astype(np.float64), y.astype(np.float64))
        final_intercept = 0.0
    else:
        model = Ridge(alpha=float(alpha), fit_intercept=fit_intercept, positive=positive)
        model.fit(X, y)
        final_coef = model.coef_.astype(np.float64)
        final_intercept = float(model.intercept_)
    return oof, final_coef.astype(np.float32), float(final_intercept)


def apply_pp(
    stack_delta: np.ndarray,
    pf_delta: np.ndarray,
    md_since: np.ndarray,
    *,
    alpha: float,
    tau: float | None,
    w_pf: float,
) -> np.ndarray:
    d = ((1.0 - w_pf) * stack_delta.astype(np.float32) + w_pf * pf_delta.astype(np.float32)).astype(np.float32)
    if tau is not None:
        d *= 1.0 - np.exp(-np.maximum(md_since, 0.0) / float(tau))
    return d * float(alpha)


def postprocess_grid(
    stack_delta: np.ndarray,
    y_abs: np.ndarray,
    last: np.ndarray,
    pf_delta: np.ndarray,
    md_since: np.ndarray,
) -> list[dict]:
    rows = []
    for alpha in np.arange(0.80, 1.081, 0.02):
        for tau in [None, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 250.0, 500.0]:
            for w_pf in np.arange(0.0, 0.301, 0.025):
                pred = last + apply_pp(stack_delta, pf_delta, md_since, alpha=float(alpha), tau=tau, w_pf=float(w_pf))
                rows.append(
                    {
                        "alpha": float(alpha),
                        "tau": None if tau is None else float(tau),
                        "w_pf": float(w_pf),
                        "rmse": rmse(y_abs, pred),
                    }
                )
    rows.sort(key=lambda r: r["rmse"])
    return rows


def main() -> None:
    cfg = json.loads((ART / "inference_config.json").read_text())
    keys = json.loads((ART / "prediction_keys.json").read_text())
    data = np.load(ART / "oof_val_predictions.npz")
    S = data["predictions"].astype(np.float32)
    y_delta = data["target"].astype(np.float32)
    meta = pd.read_csv(
        ART / "oof_val_meta.csv",
        usecols=["well", "target", "last_known_tvt", "md_since", "pf_ancc"],
        dtype={"well": "string"},
    )
    wells = meta["well"].astype(str).to_numpy()
    last = meta["last_known_tvt"].to_numpy(np.float32)
    md_since = meta["md_since"].to_numpy(np.float32)
    pf_delta = (meta["pf_ancc"].to_numpy(np.float32) - last).astype(np.float32)
    y_abs = (last + y_delta).astype(np.float32)

    saved_coef = np.asarray(cfg["stacker"]["coef"], dtype=np.float32)
    candidates = []
    saved_stack = (S @ saved_coef + float(cfg["stacker"].get("intercept", 0.0))).astype(np.float32)
    candidates.append(
        {
            "name": "saved_positive_ridge",
            "method": "saved",
            "alpha": None,
            "positive": True,
            "fit_intercept": False,
            "stack_rmse_delta": rmse(y_delta, saved_stack),
            "coef": saved_coef.astype(float).tolist(),
            "intercept": float(cfg["stacker"].get("intercept", 0.0)),
            "stack_delta": saved_stack,
        }
    )

    recipe_specs = [{"method": "nnls", "alpha": 0.0, "positive": True, "fit_intercept": False}]
    for alpha in [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        for positive in [False, True]:
            for fit_intercept in [False, True]:
                recipe_specs.append(
                    {
                        "method": "ridge",
                        "alpha": float(alpha),
                        "positive": bool(positive),
                        "fit_intercept": bool(fit_intercept),
                    }
                )

    for spec in recipe_specs:
        name = f"{spec['method']}_a{spec['alpha']:g}_pos{int(spec['positive'])}_int{int(spec['fit_intercept'])}"
        print(f"fit {name}", flush=True)
        stack_oof, coef, intercept = fit_predict_linear(S, y_delta, wells, **spec)
        candidates.append(
            {
                "name": name,
                **spec,
                "stack_rmse_delta": rmse(y_delta, stack_oof),
                "coef": coef.astype(float).tolist(),
                "intercept": float(intercept),
                "stack_delta": stack_oof,
            }
        )

    reports = []
    for cand in candidates:
        top_pp = postprocess_grid(cand["stack_delta"], y_abs, last, pf_delta, md_since)[:20]
        report = {k: v for k, v in cand.items() if k != "stack_delta"}
        report["best_postprocess"] = top_pp[0]
        report["top_postprocess"] = top_pp
        reports.append(report)
        print(cand["name"], "stack_delta", cand["stack_rmse_delta"], "best", top_pp[0], flush=True)

    reports.sort(key=lambda r: r["best_postprocess"]["rmse"])
    payload = {
        "experiment_id": "exp120",
        "route": "v10_oof_groupcv_stack_postprocess_sweep",
        "prediction_keys": keys,
        "row_count": int(len(y_delta)),
        "well_count": int(pd.Series(wells).nunique()),
        "saved_config": cfg,
        "best": reports[0],
        "reports": reports,
        "note": "Stack weights are selected by a second GroupKFold by well over saved v10 OOF rows; no test labels are used.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ["experiment_id", "row_count", "well_count", "best"]}, indent=2)[:4000])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
