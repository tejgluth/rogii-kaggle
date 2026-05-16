"""exp019 (per-well smoothing) and exp020 (Ridge stack) on existing OOFs."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"


def load_targets_and_groups():
    files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    ys, gs, lens = [], [], []
    for p in files:
        df = pd.read_csv(p, usecols=["TVT"])
        ys.append(df["TVT"].values.astype(np.float32))
        gs.append(np.full(len(df), p.stem.replace("__horizontal_well", "")))
        lens.append(len(df))
    return np.concatenate(ys), np.concatenate(gs), files, lens


def load_test_groups():
    files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    gs, lens, names = [], [], []
    for p in files:
        n = sum(1 for _ in open(p)) - 1
        names.append(p.stem.replace("__horizontal_well", ""))
        gs.append(np.full(n, names[-1]))
        lens.append(n)
    return np.concatenate(gs), names, lens


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def per_well_smooth(preds, groups, window=21, poly=3):
    out = preds.copy().astype(np.float64)
    for w in np.unique(groups):
        mask = groups == w
        seg = out[mask]
        if len(seg) < window + 1:
            continue
        win = min(window, len(seg) - (1 - len(seg) % 2))
        if win % 2 == 0:
            win -= 1
        if win >= poly + 1:
            out[mask] = savgol_filter(seg, window_length=win, polyorder=poly)
    return out.astype(np.float32)


def main():
    print("Loading targets/groups...")
    y, groups, train_files, train_lens = load_targets_and_groups()
    print(f"y: {y.shape}, n_wells: {len(np.unique(groups))}")

    test_groups, test_well_names, test_lens = load_test_groups()
    print(f"test groups: {test_groups.shape}, n_test_wells: {len(np.unique(test_groups))}")

    # -------- Load OOFs --------
    members = {
        "exp014": "catboost",
        "exp013": "lightgbm",
        "exp012": "xgboost",
        "exp009": "lightgbm",
        "exp011": "lightgbm",
        "exp006": "lightgbm",
    }
    oofs, tests = {}, {}
    for e, m in members.items():
        oofs[e] = np.load(ROOT / f"experiments/oof/oof_{m}_{e}.npy")
        tests[e] = np.load(ROOT / f"experiments/test_preds/test_{m}_{e}.npy")
        print(f"{e}: oof rmse={rmse(oofs[e], y):.4f}")

    # ===== exp019: per-well smoothing of exp014 =====
    print("\n=== exp019: per-well savgol smoothing of exp014 ===")
    base = oofs["exp014"]
    base_rmse = rmse(base, y)
    print(f"baseline exp014 RMSE: {base_rmse:.4f}")
    best_cfg, best_oof, best_rmse_v = None, base, base_rmse
    for win, poly in [(11, 3), (21, 3), (41, 3), (61, 3), (81, 3), (101, 3),
                       (151, 3), (201, 3), (41, 2), (81, 2), (151, 2)]:
        sm = per_well_smooth(base, groups, window=win, poly=poly)
        r = rmse(sm, y)
        mark = "*" if r < best_rmse_v else " "
        print(f" {mark} win={win:>3} poly={poly}: RMSE={r:.4f} ({r-base_rmse:+.4f})")
        if r < best_rmse_v:
            best_rmse_v = r
            best_oof = sm
            best_cfg = (win, poly)

    if best_cfg is None:
        print("Smoothing didn't help. Keeping baseline.")
        best_cfg = (None, None)

    # Apply best smoothing to test predictions
    win, poly = best_cfg if best_cfg[0] else (21, 3)
    test_sm = per_well_smooth(tests["exp014"], test_groups, window=win, poly=poly) \
        if best_cfg[0] else tests["exp014"]

    np.save(ROOT / "experiments/oof/oof_catboost_exp019.npy", best_oof.astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_catboost_exp019.npy", test_sm.astype(np.float32))
    exp019_result = {
        "experiment_id": "exp019",
        "model": "catboost+smooth",
        "phase": "postprocessing",
        "base_experiment": "exp014",
        "cv_rmse": best_rmse_v,
        "best_window": best_cfg[0],
        "best_poly": best_cfg[1],
        "baseline_rmse": base_rmse,
        "delta": best_rmse_v - base_rmse,
        "notes": "Per-well Savitzky-Golay smoothing of exp014 OOF and test predictions.",
        "oof_path": "experiments/oof/oof_catboost_exp019.npy",
        "test_path": "experiments/test_preds/test_catboost_exp019.npy",
    }
    with open(ROOT / "experiments/results/exp019.json", "w") as f:
        json.dump(exp019_result, f, indent=2)
    print(f"exp019: best RMSE={best_rmse_v:.4f} (cfg={best_cfg})")

    # ===== exp020: Ridge stacking on OOFs =====
    print("\n=== exp020: Ridge stacking ===")
    member_list = ["exp014", "exp013", "exp012", "exp009", "exp011", "exp006"]
    X_oof = np.column_stack([oofs[e] for e in member_list]).astype(np.float64)
    X_test = np.column_stack([tests[e] for e in member_list]).astype(np.float64)

    gkf = GroupKFold(n_splits=5)
    stack_oof = np.zeros(len(y), dtype=np.float64)
    fold_rmses = []
    fold_coefs = []
    test_preds_folds = []
    best_alpha = None
    # quick alpha sweep on first fold
    fold0 = next(gkf.split(X_oof, y, groups))
    tr0, va0 = fold0
    best_a, best_r = None, 1e18
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(X_oof[tr0], y[tr0])
        p = m.predict(X_oof[va0])
        r = rmse(p, y[va0])
        print(f"  alpha={alpha:>7}: fold0 RMSE={r:.4f}")
        if r < best_r:
            best_r, best_a = r, alpha
    best_alpha = best_a
    print(f"Best alpha (fold0): {best_alpha}")

    gkf = GroupKFold(n_splits=5)
    for fi, (tr, va) in enumerate(gkf.split(X_oof, y, groups)):
        m = Ridge(alpha=best_alpha, fit_intercept=True)
        m.fit(X_oof[tr], y[tr])
        stack_oof[va] = m.predict(X_oof[va])
        fold_rmses.append(rmse(stack_oof[va], y[va]))
        fold_coefs.append(m.coef_.tolist())
        test_preds_folds.append(m.predict(X_test))
        print(f" fold{fi}: RMSE={fold_rmses[-1]:.4f} coefs={m.coef_.round(3).tolist()} intercept={m.intercept_:.2f}")

    stack_rmse = rmse(stack_oof, y)
    test_stack = np.mean(np.column_stack(test_preds_folds), axis=1)
    print(f"\nRidge stack OOF RMSE: {stack_rmse:.4f}  (vs exp014 {rmse(oofs['exp014'],y):.4f})")

    # Try non-negative least squares stack
    from scipy.optimize import nnls
    print("\n--- NNLS (non-negative, no intercept) stack ---")
    gkf2 = GroupKFold(n_splits=5)
    nnls_oof = np.zeros(len(y), dtype=np.float64)
    nnls_test_folds = []
    nnls_coefs = []
    for fi, (tr, va) in enumerate(gkf2.split(X_oof, y, groups)):
        # add intercept column
        Xtr = np.column_stack([X_oof[tr], np.ones(len(tr))])
        coef, _ = nnls(Xtr, y[tr])
        Xva = np.column_stack([X_oof[va], np.ones(len(va))])
        nnls_oof[va] = Xva @ coef
        Xte = np.column_stack([X_test, np.ones(len(X_test))])
        nnls_test_folds.append(Xte @ coef)
        nnls_coefs.append(coef.tolist())
        print(f" fold{fi}: RMSE={rmse(nnls_oof[va], y[va]):.4f} coefs={np.round(coef,3).tolist()}")
    nnls_rmse = rmse(nnls_oof, y)
    nnls_test = np.mean(np.column_stack(nnls_test_folds), axis=1)
    print(f"NNLS stack OOF RMSE: {nnls_rmse:.4f}")

    # Pick winner
    if nnls_rmse < stack_rmse:
        winner_oof, winner_test, winner_rmse, winner_name = nnls_oof, nnls_test, nnls_rmse, "nnls"
    else:
        winner_oof, winner_test, winner_rmse, winner_name = stack_oof, test_stack, stack_rmse, "ridge"
    print(f"Winner: {winner_name} ({winner_rmse:.4f})")

    # Also try smoothing the stack
    sm_stack = per_well_smooth(winner_oof.astype(np.float32), groups,
                                window=best_cfg[0] or 81, poly=best_cfg[1] or 3)
    sm_rmse = rmse(sm_stack, y)
    print(f"Smoothed stack RMSE: {sm_rmse:.4f}")
    if sm_rmse < winner_rmse:
        winner_test = per_well_smooth(winner_test.astype(np.float32), test_groups,
                                       window=best_cfg[0] or 81, poly=best_cfg[1] or 3)
        winner_oof = sm_stack
        winner_rmse = sm_rmse
        winner_name += "+smooth"

    np.save(ROOT / "experiments/oof/oof_stack_exp020.npy", winner_oof.astype(np.float32))
    np.save(ROOT / "experiments/test_preds/test_stack_exp020.npy", winner_test.astype(np.float32))
    exp020_result = {
        "experiment_id": "exp020",
        "model": f"stack_{winner_name}",
        "phase": "stacking",
        "members": member_list,
        "cv_rmse": winner_rmse,
        "ridge_alpha": best_alpha,
        "ridge_rmse": stack_rmse,
        "nnls_rmse": nnls_rmse,
        "fold_rmses_ridge": fold_rmses,
        "fold_coefs_ridge": fold_coefs,
        "fold_coefs_nnls": nnls_coefs,
        "notes": f"Stacking meta-learner ({winner_name}) on OOFs of {member_list}. Winner: {winner_name}.",
        "oof_path": "experiments/oof/oof_stack_exp020.npy",
        "test_path": "experiments/test_preds/test_stack_exp020.npy",
    }
    with open(ROOT / "experiments/results/exp020.json", "w") as f:
        json.dump(exp020_result, f, indent=2)
    print(f"\nexp020 saved: {winner_name} RMSE={winner_rmse:.4f}")

    # Summary
    print("\n========== SUMMARY ==========")
    print(f"exp014 baseline:     {rmse(oofs['exp014'], y):.4f}")
    print(f"exp019 smooth (best):{best_rmse_v:.4f}")
    print(f"exp020 stack:        {winner_rmse:.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
