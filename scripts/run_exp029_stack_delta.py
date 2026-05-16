"""exp029: NNLS stack of delta-target experiments + savgol postprocessing.

Stacks the OOFs from delta-target experiments (exp025, exp026, exp028 if present).
The OOFs are stored as absolute TVT in original CSV row order. We compute RMSE
only on lateral rows (TVT_input is NaN) since that's what the submission scores.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_targets():
    files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    ys, gs, lateral_masks = [], [], []
    for p in files:
        df = pd.read_csv(p, usecols=["TVT", "TVT_input"])
        ys.append(df["TVT"].values.astype(np.float32))
        gs.append(np.full(len(df), p.stem.replace("__horizontal_well", "")))
        lateral_masks.append(df["TVT_input"].isna().values)
    return np.concatenate(ys), np.concatenate(gs), np.concatenate(lateral_masks)


def load_test_groups():
    files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    gs, masks = [], []
    for p in files:
        df = pd.read_csv(p, usecols=["TVT_input"])
        gs.append(np.full(len(df), p.stem.replace("__horizontal_well", "")))
        masks.append(df["TVT_input"].isna().values)
    return np.concatenate(gs), np.concatenate(masks)


def per_well_smooth(preds, groups, mask, window=17, poly=3):
    """Apply savgol smoothing per well, but only to lateral rows (mask=True)."""
    out = preds.copy().astype(np.float32)
    for w in np.unique(groups):
        sel = groups == w
        seg = out[sel]
        if len(seg) < 5:
            continue
        n = len(seg)
        win = min(window, n)
        if win % 2 == 0:
            win -= 1
        if win >= poly + 2:
            out[sel] = savgol_filter(seg, win, poly)
    return out


def main():
    print("Loading targets...")
    y, groups, lateral = load_targets()
    print(f"  y: {y.shape}, lateral: {lateral.sum()} / {len(y)}")
    tg, tmask = load_test_groups()
    print(f"  test: {tg.shape}, lateral: {tmask.sum()} / {len(tg)}")

    members = []
    for tag, e in [("lightgbm","exp025"), ("lightgbm","exp026"), ("combined","exp028")]:
        p = ROOT / f"experiments/oof/oof_{tag}_{e}.npy"
        if p.exists():
            members.append((e, tag, np.load(p)))
            tp = ROOT / f"experiments/test_preds/test_{tag}_{e}.npy"
            print(f" {e}: oof present, lateral RMSE = {rmse(np.load(p)[lateral], y[lateral]):.4f}")
    print(f"members: {[m[0] for m in members]}")
    if not members:
        print("No delta-target OOFs found. Exit.")
        return

    if len(members) == 1:
        e, tag, oof = members[0]
        test = np.load(ROOT / f"experiments/test_preds/test_{tag}_{e}.npy")
        # Just smooth + save
        for win in [11, 17, 31, 51, 101]:
            for poly in [2, 3]:
                sm = per_well_smooth(oof, groups, lateral, window=win, poly=poly)
                r = rmse(sm[lateral], y[lateral])
                print(f" smooth win={win} poly={poly}: lateral_RMSE={r:.4f}")
        return

    # NNLS stack
    X = np.column_stack([m[2] for m in members]).astype(np.float64)
    test_arrays = []
    for e, tag, _ in members:
        test_arrays.append(np.load(ROOT / f"experiments/test_preds/test_{tag}_{e}.npy"))
    Xt = np.column_stack(test_arrays).astype(np.float64)

    gkf = GroupKFold(5)
    stack_oof = np.zeros(len(y))
    stack_test_folds = []
    for fi, (tr, va) in enumerate(gkf.split(X, y, groups)):
        Xtr = np.column_stack([X[tr], np.ones(len(tr))])
        coef, _ = nnls(Xtr, y[tr])
        stack_oof[va] = np.column_stack([X[va], np.ones(len(va))]) @ coef
        stack_test_folds.append(np.column_stack([Xt, np.ones(len(Xt))]) @ coef)
        r = rmse(stack_oof[va][lateral[va]], y[va][lateral[va]])
        print(f" fold{fi}: lateral_RMSE={r:.4f} coef={np.round(coef,3).tolist()}")
    stack_oof = stack_oof.astype(np.float32)
    stack_test = np.mean(np.column_stack(stack_test_folds), axis=1).astype(np.float32)
    raw_lat = rmse(stack_oof[lateral], y[lateral])
    print(f"NNLS stack lateral RMSE: {raw_lat:.4f}")

    # Smoothing sweep
    best = (raw_lat, None, None, stack_oof, stack_test)
    for win in [11, 17, 31, 51, 101, 201]:
        for poly in [2, 3]:
            sm = per_well_smooth(stack_oof, groups, lateral, window=win, poly=poly)
            sm_t = per_well_smooth(stack_test, tg, tmask, window=win, poly=poly)
            r = rmse(sm[lateral], y[lateral])
            mark = "*" if r < best[0] else " "
            print(f" {mark} smooth win={win} poly={poly}: lateral_RMSE={r:.4f}")
            if r < best[0]:
                best = (r, win, poly, sm, sm_t)

    best_r, best_win, best_poly, best_oof, best_test = best
    np.save(ROOT/"experiments/oof/oof_stack_exp029.npy", best_oof)
    np.save(ROOT/"experiments/test_preds/test_stack_exp029.npy", best_test)
    json.dump({
        "experiment_id": "exp029",
        "model": "nnls_stack_delta+savgol",
        "phase": "stacking",
        "cv_rmse_lateral": best_r,
        "savgol_window": best_win, "savgol_poly": best_poly,
        "members": [m[0] for m in members],
        "notes": "NNLS stack of delta-target OOFs + per-well savgol smoothing on lateral rows.",
        "oof_path": "experiments/oof/oof_stack_exp029.npy",
        "test_path": "experiments/test_preds/test_stack_exp029.npy",
    }, open(ROOT/"experiments/results/exp029.json","w"), indent=2)
    print(f"\nexp029 saved: lateral_RMSE={best_r:.4f} (win={best_win} poly={best_poly})")


if __name__ == "__main__":
    main()
