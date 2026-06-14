# ROGII — Tomorrow's submission plan (2026-06-14)

Updated after the 2026-06-13 autonomous research session. Read
`experiments/findings_2026-06-13_leak_and_ceiling.md` first.
(Previous plan archived as `tomorrow_submission_plan_2026-06-14_OLD.md`.)

## Bottom line
- The train/test overlap is a **verified trap** — do NOT submit the exact-lookup file.
- `<6` = top-2 on the live leaderboard (#1 5.785). Public forks + honest models all sit
  in the **7.0–7.6** band. Realistic goal tomorrow: defend ~7.5 and probe whether the
  diverse blend dips below it.

## The 3 prepared files (`submissions/tomorrow/`, all 14151 rows, validated, no NaN, 0.01-grid)

| # | file | what it is | expected LB |
|---|------|------------|-------------|
| 1 | `candidate1_lightning_w55.csv` | best **confirmed** single system (genuine SP45/Fleongg blend, no overlay) | **~7.58** (known) |
| 2 | `candidate2_blend4.csv` | 0.38·w55 + 0.22·pilkwang + 0.20·fle3n + 0.20·v8gold | ~7.4–7.6 (best shot under 7.5) |
| 3 | `candidate3_v8gold.csv` | fresh particle-filter + NCC + beam + formation-KNN + LGBM/XGB/CB ensemble (honest OOF 12.4); most decorrelated (3.4–4.4 RMSE from the others) | unknown — diverse probe |

## Submit order (3 slots)
1. **`candidate1_lightning_w55.csv`** first — lock in the known ~7.58 floor.
2. **`candidate2_blend4.csv`** — the diversification play.
3. **`candidate3_v8gold.csv`** — high-variance probe; if it beats 7.58 it validates the
   v8 ensemble as the new base for further work.

Submit as raw CSVs:
```bash
.venv/bin/kaggle competitions submit rogii-wellbore-geology-prediction \
  -f submissions/tomorrow/candidate1_lightning_w55.csv -m "lightning w55 confirmed 7.58"
```
(or push the corresponding notebook kernels if code-submission is required).

## What was tried toward <7 (and what it bought)
- Honest CV harness mirroring the test task exactly (`local_runs/align/harness.py`).
- Novel **global DP/Viterbi GR→typewell aligner** (`local_runs/align/dp_aligner.py`):
  optimal smooth TVT path vs the greedy beam/PF. With a const-prior pull it beats const by
  ~1% (15.69→15.57 pooled; 11.54→11.37 on the test-dup wells). Real but small — in these
  wells TVT is nearly constant, so GR alignment is only a minor correction.
- Spatial structural-surface KNN (TVT+Z from neighbors) — **fails** (88–112); `TVT+Z` is
  not single-valued in (X,Y). Only formation contacts are areal (v8's formation-KNN uses them).
- DP-alignment delta as a tree feature: **negative result** — geometry-only OOF 14.595 →
  geometry+DP 14.793 (worse). The DP path adds noise to the tree; it is NOT used in the
  submissions. (`local_runs/honest/train_lgbm_dp.py`.)

## To actually reach <6 (future work, needs more than a CPU night)
Leaders are ~1.5–2 RMSE ahead of all public work — a genuinely better core alignment.
Open directions: (a) joint DP over (TVT, dip) with curvature penalty + per-well GR
calibration; (b) sequence model for apparent-dip evolution; (c) richer leak-free spatial
formation reconstruction. None likely to clear <6 without the top solution's method.
