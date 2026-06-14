#!/usr/bin/env python3
"""exp121: honest v11 OOF stack/postprocess sweep.

The v11 artifact dataset contains model OOF vectors but no train-row metadata.
The exp116 local cache was built from the same sorted official train wells and
has the same row count/order, so this script uses only its non-feature metadata
columns for scoring and group splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "local_runs" / "v11_artifact_full"
META = ROOT / "local_runs" / "exp116_fast_cache" / "train.csv"
OUT = ROOT / "experiments" / "results" / "exp121_v11_oof_stack_sweep.json"


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y, pred))


def smooth_by_well(values: np.ndarray, wells: np.ndarray, win: int, poly: int = 3) -> np.ndarray:
    if win <= 1:
        return values.astype(np.float32, copy=True)
    out = values.astype(np.float32, copy=True)
    for well in pd.unique(wells):
        idx = np.flatnonzero(wells == well)
        n = len(idx)
        w = min(int(win), n if n % 2 else n - 1)
        if w >= poly + 2:
            out[idx] = savgol_filter(values[idx], w, poly).astype(np.float32)
    return out


def apply_pp(delta: np.ndarray, pf_delta: np.ndarray, md_since: np.ndarray, alpha: float, tau, w_pf: float) -> np.ndarray:
    d = ((1.0 - w_pf) * delta.astype(np.float32) + w_pf * pf_delta.astype(np.float32)).astype(np.float32)
    if tau is not None:
        d *= 1.0 - np.exp(-np.maximum(md_since, 0.0) / float(tau))
    return d * float(alpha)


def load_frame() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    manifest = json.loads((ART / "manifest.json").read_text())
    keys = list(manifest["model_keys"])
    preds = []
    for key in keys:
        path = ART / "models" / key / "oof_preds.pkl"
        arr = joblib.load(path).astype(np.float32)
        preds.append(arr)
    S = np.column_stack(preds).astype(np.float32)
    usecols = ["well", "id", "target", "last_known_tvt", "pf_ancc", "md_since"]
    meta = pd.read_csv(META, usecols=usecols, dtype={"well": "string", "id": "string"})
    if len(meta) != len(S):
        raise RuntimeError(f"metadata rows {len(meta)} != OOF rows {len(S)}")
    return meta, keys, S


def climber_from_history(S: np.ndarray, keys: list[str]) -> tuple[str, np.ndarray]:
    hist = pd.read_csv(ART / "climber_history.csv")
    coefs = np.zeros(len(keys), dtype=np.float32)
    for model, coef in zip(hist["model"], hist["coef"]):
        coefs[keys.index(str(model))] += float(coef)
    return "saved_climber", coefs


def fit_oof_linear(S: np.ndarray, y: np.ndarray, groups: np.ndarray, method: str, alpha: float = 1.0, positive: bool = False) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, va in GroupKFold(n_splits=5).split(S, y, groups=groups):
        if method == "nnls":
            coef, _ = nnls(S[tr].astype(np.float64), y[tr].astype(np.float64))
            pred = S[va].astype(np.float64) @ coef
        elif method == "ridge":
            model = Ridge(alpha=float(alpha), fit_intercept=False, positive=positive)
            model.fit(S[tr], y[tr])
            pred = model.predict(S[va])
        else:
            raise ValueError(method)
        oof[va] = pred.astype(np.float32)

    if method == "nnls":
        coef, _ = nnls(S.astype(np.float64), y.astype(np.float64))
        coef = coef.astype(np.float32)
    else:
        model = Ridge(alpha=float(alpha), fit_intercept=False, positive=positive)
        model.fit(S, y)
        coef = model.coef_.astype(np.float32)
    return oof, coef


def greedy_hill(S: np.ndarray, y: np.ndarray, keys: list[str], allow_negative: bool, precision: float) -> tuple[str, np.ndarray, np.ndarray]:
    weights = np.arange(-0.5, 0.5001, precision) if allow_negative else np.arange(precision, 0.5001, precision)
    scores = np.array([rmse(y, S[:, i]) for i in range(S.shape[1])], dtype=np.float64)
    start = int(scores.argmin())
    coef = np.zeros(S.shape[1], dtype=np.float64)
    coef[start] = 1.0
    current = S[:, start].astype(np.float32)
    current_score = float(scores[start])
    remaining = [i for i in range(S.shape[1]) if i != start]
    while remaining:
        best = (current_score, None, None, None)
        for idx in remaining:
            new = S[:, idx]
            for w in weights:
                cand = ((1.0 - w) * current + w * new).astype(np.float32)
                score = rmse(y, cand)
                if score < best[0] - 1e-7:
                    best = (score, idx, float(w), cand)
        if best[1] is None:
            break
        score, idx, w, current = best
        coef *= 1.0 - w
        coef[idx] += w
        current_score = score
        remaining.remove(idx)
    tag = f"greedy_neg{int(allow_negative)}_p{precision:g}"
    return tag, current.astype(np.float32), coef.astype(np.float32)


def score_postprocess(stack_delta: np.ndarray, meta: pd.DataFrame, alpha: float, tau, w_pf: float, sg: int) -> float:
    wells = meta["well"].astype(str).to_numpy()
    last = meta["last_known_tvt"].to_numpy(np.float32)
    target_abs = (last + meta["target"].to_numpy(np.float32)).astype(np.float32)
    pf_delta = (meta["pf_ancc"].to_numpy(np.float32) - last).astype(np.float32)
    md_since = meta["md_since"].to_numpy(np.float32)
    delta = apply_pp(stack_delta, pf_delta, md_since, alpha, tau, w_pf)
    pred = last + delta
    if sg:
        pred = smooth_by_well(pred, wells, sg)
    return rmse(target_abs, pred)


def postprocess_report(name: str, stack_delta: np.ndarray, meta: pd.DataFrame) -> dict:
    last = meta["last_known_tvt"].to_numpy(np.float32)
    pf_delta = (meta["pf_ancc"].to_numpy(np.float32) - last).astype(np.float32)
    md_since = meta["md_since"].to_numpy(np.float32)
    target_abs = (last + meta["target"].to_numpy(np.float32)).astype(np.float32)

    rows = []
    # Keep this deliberately small. The rows are ~3.8M, and we are using this
    # to choose robust regions, not to overfit the fifth decimal place.
    for alpha in [0.98, 1.0, 1.02]:
        for tau in [50.0, 75.0, 100.0]:
            for w_pf in [0.03, 0.06, 0.09]:
                delta = apply_pp(stack_delta, pf_delta, md_since, float(alpha), tau, float(w_pf))
                pred = last + delta
                rows.append(
                    {
                        "alpha": float(alpha),
                        "tau": None if tau is None else float(tau),
                        "w_pf": float(w_pf),
                        "sg_win": 0,
                        "rmse": rmse(target_abs, pred),
                    }
                )
    rows.sort(key=lambda r: r["rmse"])
    # The production v11 notebook smooths final absolute predictions with a
    # 17-point SavGol filter. Evaluate that only around the best raw region.
    refined = []
    for item in rows[:10]:
        delta = apply_pp(stack_delta, pf_delta, md_since, item["alpha"], item["tau"], item["w_pf"])
        raw = last + delta
        pred = smooth_by_well(raw, wells, 17)
        refined.append({**item, "sg_win": 17, "rmse": rmse(target_abs, pred)})
    rows = sorted(rows[:20] + refined, key=lambda r: r["rmse"])
    return {"name": name, "stack_delta_rmse": rmse(meta["target"].to_numpy(np.float32), stack_delta), "best": rows[0], "top": rows[:20]}


def main() -> None:
    meta, keys, S = load_frame()
    y = meta["target"].to_numpy(np.float32)
    groups = meta["well"].astype(str).to_numpy()
    candidates = []

    saved_name, saved_coef = climber_from_history(S, keys)
    candidates.append((saved_name, (S @ saved_coef).astype(np.float32), saved_coef))

    for method, alpha, positive in [
        ("nnls", 0.0, True),
        ("ridge", 0.1, True),
        ("ridge", 1.0, True),
        ("ridge", 10.0, True),
        ("ridge", 100.0, True),
        ("ridge", 1.0, False),
        ("ridge", 10.0, False),
        ("ridge", 100.0, False),
    ]:
        tag = f"{method}_a{alpha:g}_pos{int(positive)}"
        print("fit", tag, flush=True)
        oof, coef = fit_oof_linear(S, y, groups, method=method, alpha=alpha, positive=positive)
        candidates.append((tag, oof, coef))

    # The shipped v11 artifact already used the real hill-climbing package.
    # Avoid an expensive duplicate local greedy pass here; use this sweep to
    # test linear stack alternatives against that saved climber.

    prelim = []
    for tag, oof, coef in candidates:
        saved_pp = score_postprocess(oof, meta, alpha=1.0, tau=75.0, w_pf=0.06, sg=17)
        prelim.append(
            {
                "name": tag,
                "saved_pp_rmse": float(saved_pp),
                "stack_delta_rmse": rmse(y, oof),
                "coef": coef.astype(float).tolist(),
            }
        )
        print(f"prelim {tag}: stack={prelim[-1]['stack_delta_rmse']:.6f} saved_pp={saved_pp:.6f}", flush=True)

    prelim.sort(key=lambda r: r["saved_pp_rmse"])
    keep_names = {r["name"] for r in prelim[:4]} | {"saved_climber"}
    reports = []
    for tag, oof, coef in candidates:
        if tag not in keep_names:
            continue
        print("grid", tag, flush=True)
        rep = postprocess_report(tag, oof, meta)
        rep["coef"] = coef.astype(float).tolist()
        rep["prelim_saved_pp_rmse"] = next(r["saved_pp_rmse"] for r in prelim if r["name"] == tag)
        reports.append(rep)

    reports.sort(key=lambda r: r["best"]["rmse"])
    payload = {
        "experiment_id": "exp121",
        "route": "v11_oof_stack_postprocess_sweep",
        "keys": keys,
        "row_count": int(len(meta)),
        "well_count": int(meta["well"].nunique()),
        "best": reports[0],
        "prelim": prelim,
        "reports": reports,
        "note": "Uses v11 saved OOF predictions and exp116 metadata row order; no test labels.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"best": payload["best"], "row_count": payload["row_count"]}, indent=2)[:5000])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
