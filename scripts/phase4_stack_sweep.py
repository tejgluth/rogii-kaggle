"""phase 4: comprehensive stacking sweep on all delta-era OOFs.

Loads every OOF whose lateral RMSE on its own beats the absolute-target floor (< 30),
runs per-member RMSE, hill-climbing avg, NNLS (constant intercept), Ridge w/ non-neg
constraint via positive lasso, per-fold weights via GroupKFold, and a savgol/median
postprocess sweep. Saves the best stacker + a leaderboard of every candidate tried.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import savgol_filter, medfilt
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_targets():
    files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    ys, gs, lats = [], [], []
    for p in files:
        df = pd.read_csv(p, usecols=["TVT", "TVT_input"])
        ys.append(df["TVT"].values.astype(np.float32))
        gs.append(np.full(len(df), p.stem.replace("__horizontal_well", "")))
        lats.append(df["TVT_input"].isna().values)
    return np.concatenate(ys), np.concatenate(gs), np.concatenate(lats)


def load_test_meta():
    files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    gs, m = [], []
    for p in files:
        df = pd.read_csv(p, usecols=["TVT_input"])
        gs.append(np.full(len(df), p.stem.replace("__horizontal_well", "")))
        m.append(df["TVT_input"].isna().values)
    return np.concatenate(gs), np.concatenate(m)


def per_well_savgol(preds, groups, window, poly):
    out = preds.copy().astype(np.float32)
    for w in np.unique(groups):
        sel = groups == w
        seg = out[sel]
        n = len(seg)
        if n < 5:
            continue
        win = min(window, n)
        if win % 2 == 0:
            win -= 1
        if win >= poly + 2:
            out[sel] = savgol_filter(seg, win, poly)
    return out


def per_well_median(preds, groups, k):
    out = preds.copy().astype(np.float32)
    for w in np.unique(groups):
        sel = groups == w
        seg = out[sel]; n = len(seg)
        if n < 5: continue
        kk = min(k, n)
        if kk % 2 == 0: kk -= 1
        if kk >= 3:
            out[sel] = medfilt(seg, kk)
    return out


# ---- main ----
print(f"[{time.strftime('%H:%M:%S')}] loading targets…")
y, groups, lateral = load_targets()
tg, tmask = load_test_meta()
print(f"  train: {len(y)} rows, lateral={int(lateral.sum())} rows, {len(np.unique(groups))} wells")
print(f"  test : {len(tg)} rows, lateral={int(tmask.sum())} rows, {len(np.unique(tg))} wells")

# Score every OOF on lateral RMSE; only consider strong members (< 10).
oof_dir = ROOT / "experiments/oof"
test_dir = ROOT / "experiments/test_preds"
candidates = []
print(f"\n[{time.strftime('%H:%M:%S')}] scoring every OOF…")
for p in sorted(oof_dir.glob("*.npy")):
    a = np.load(p, allow_pickle=False)
    if a.shape != y.shape:
        continue
    tp = test_dir / p.name.replace("oof_", "test_")
    if not tp.exists():
        continue
    lat_rmse = rmse(a[lateral], y[lateral])
    full_rmse = rmse(a, y)
    candidates.append({"name": p.stem.replace("oof_", ""), "oof_path": str(p),
                       "test_path": str(tp), "lat_rmse": lat_rmse, "full_rmse": full_rmse})
    print(f"  {p.stem.replace('oof_',''):30s} lat={lat_rmse:8.4f}  full={full_rmse:8.4f}")

# Strong-member filter: lateral < 12 (exp025 baseline was ~12).
strong = [c for c in candidates if c["lat_rmse"] < 12.0 and "stack_" not in c["name"]]
strong.sort(key=lambda x: x["lat_rmse"])
print(f"\n{len(strong)} strong base learners")
for c in strong:
    print(f"  {c['name']:30s} lat={c['lat_rmse']:.4f}")

names = [c["name"] for c in strong]
OOF = np.column_stack([np.load(c["oof_path"]) for c in strong]).astype(np.float64)
TST = np.column_stack([np.load(c["test_path"]) for c in strong]).astype(np.float64)

results = []
def record(method, lat_rmse, oof, test, info=""):
    print(f"  [{method:40s}] lat={lat_rmse:.4f}  {info}")
    results.append({"method": method, "lat_rmse": lat_rmse, "info": info,
                    "oof": oof, "test": test})

# ---- (a) hill-climb with averaging ----
print(f"\n[{time.strftime('%H:%M:%S')}] hill-climbing on lateral RMSE…")
selected = [int(np.argmin([c["lat_rmse"] for c in strong]))]
best = strong[selected[0]]["lat_rmse"]
improved = True; iters = 0
while improved and iters < 200:
    improved = False
    for j in range(len(strong)):
        cand = selected + [j]
        avg = OOF[:, cand].mean(axis=1)
        r = rmse(avg[lateral], y[lateral])
        if r < best - 1e-5:
            best = r; selected = cand; improved = True
    iters += 1
hc_oof = OOF[:, selected].mean(axis=1)
hc_test = TST[:, selected].mean(axis=1)
record("hillclimb_avg", best, hc_oof, hc_test, f"members={[names[i] for i in selected]}")

# ---- (b) NNLS with intercept, GroupKFold OOF ----
print(f"\n[{time.strftime('%H:%M:%S')}] NNLS w/ intercept, 5-fold GroupKFold…")
gkf = GroupKFold(5)
nnls_oof = np.zeros(len(y))
nnls_test_acc = np.zeros(len(tg))
all_coefs = []
for fi, (tr, va) in enumerate(gkf.split(OOF, y, groups)):
    Xtr = np.column_stack([OOF[tr], np.ones(len(tr))])
    coef, _ = nnls(Xtr, y[tr])
    all_coefs.append(coef)
    nnls_oof[va] = np.column_stack([OOF[va], np.ones(len(va))]) @ coef
    nnls_test_acc += np.column_stack([TST, np.ones(len(tg))]) @ coef
nnls_test = nnls_test_acc / 5
record("nnls_oof_per_fold", rmse(nnls_oof[lateral], y[lateral]), nnls_oof, nnls_test,
       f"mean_coef={np.round(np.mean(all_coefs, axis=0)[:-1], 3).tolist()}")

# ---- (c) NNLS w/o intercept, single global fit ----
print(f"\n[{time.strftime('%H:%M:%S')}] NNLS no-intercept, global fit…")
coef_g, _ = nnls(OOF, y)
w_g = coef_g / (coef_g.sum() + 1e-9)
g_oof = OOF @ w_g
g_test = TST @ w_g
record("nnls_global_simplex", rmse(g_oof[lateral], y[lateral]), g_oof, g_test,
       f"w={dict(zip(names, np.round(w_g, 3).tolist()))}")

# ---- (d) weighted-avg fine grid on top-3 by lateral RMSE ----
top3 = sorted(range(len(strong)), key=lambda i: strong[i]["lat_rmse"])[:3]
print(f"\n[{time.strftime('%H:%M:%S')}] grid search on top-3 simplex: {[names[i] for i in top3]}")
best_grid = (float("inf"), None, None, None)
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < -1e-6: continue
        oof_v = OOF[:, top3] @ np.array([w1, w2, w3])
        r = rmse(oof_v[lateral], y[lateral])
        if r < best_grid[0]:
            tst_v = TST[:, top3] @ np.array([w1, w2, w3])
            best_grid = (r, (w1, w2, w3), oof_v, tst_v)
record("grid_top3_simplex", best_grid[0], best_grid[2], best_grid[3],
       f"weights={dict(zip([names[i] for i in top3], np.round(best_grid[1],3).tolist()))}")

# ---- (e) postprocess sweep on each base candidate ----
print(f"\n[{time.strftime('%H:%M:%S')}] postprocess sweep on every result so far…")
pp_results = []
for r in list(results):
    for win in [51, 101, 201, 301, 501, 801, 1001]:
        for poly in [2, 3]:
            sm_oof = per_well_savgol(r["oof"], groups, win, poly)
            sm_test = per_well_savgol(r["test"], tg, win, poly)
            rr = rmse(sm_oof[lateral], y[lateral])
            pp_results.append({"method": f"{r['method']}_savgol_w{win}_p{poly}",
                               "lat_rmse": rr, "oof": sm_oof, "test": sm_test,
                               "info": r["method"]})
results.extend(pp_results)

# ---- Also try median + savgol cascade on the best base candidate ----
base = min(results, key=lambda r: r["lat_rmse"])
print(f"\n[{time.strftime('%H:%M:%S')}] median+savgol cascade on current leader: {base['method']} lat={base['lat_rmse']:.4f}")
for mk in [51, 101, 201, 301]:
    for win in [301, 501, 801, 1001]:
        for poly in [2, 3]:
            med_oof = per_well_median(base["oof"], groups, mk)
            sm_oof = per_well_savgol(med_oof, groups, win, poly)
            med_test = per_well_median(base["test"], tg, mk)
            sm_test = per_well_savgol(med_test, tg, win, poly)
            rr = rmse(sm_oof[lateral], y[lateral])
            results.append({"method": f"{base['method']}_med{mk}_savgol_w{win}_p{poly}",
                            "lat_rmse": rr, "oof": sm_oof, "test": sm_test, "info": ""})

# Final leaderboard
print("\n========== final leaderboard (top 30) ==========")
results.sort(key=lambda r: r["lat_rmse"])
for r in results[:30]:
    print(f"  {r['method']:60s} lat={r['lat_rmse']:.4f}")

# Save the best
winner = results[0]
print(f"\nWINNER: {winner['method']}  lat_rmse={winner['lat_rmse']:.4f}")
np.save(ROOT / "experiments/oof/oof_stack_exp050.npy", winner["oof"].astype(np.float32))
np.save(ROOT / "experiments/test_preds/test_stack_exp050.npy", winner["test"].astype(np.float32))

# Trim payload for json
log = [{"method": r["method"], "lat_rmse": r["lat_rmse"], "info": r.get("info","")}
       for r in results]
(ROOT / "experiments/results/exp050.json").write_text(json.dumps({
    "experiment_id": "exp050",
    "phase": "stacking",
    "winner": winner["method"],
    "winner_lat_rmse": winner["lat_rmse"],
    "n_candidates_tried": len(results),
    "base_members": names,
    "log": log[:60],
}, indent=2))
print(f"\nsaved oof/test for exp050 + results json")
