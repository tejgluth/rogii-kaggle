# ROGII — Autonomous session findings (2026-06-13)

## TL;DR
- The "test wells exist in train/" **overlap is a trap, not a usable leak.** Overlaying
  the train copy's lateral TVT scores ~7.59 on Kaggle (verified), NOT ~0. The train
  copy's lateral TVT for the 3 test wells is a *different geosteering interpretation*
  than the hidden scored target (~7.5 RMSE apart).
- **<6 RMSE is real but means top-2 on the leaderboard** (LB #1 = 5.785, #2 = 5.943,
  #3 = 6.276). The public-notebook band — and every fork in this repo — plateaus at
  **7.0–7.6**. Our best confirmed submission is **7.580**.
- There is **no shortcut** to <6 with the resources here (CPU-only Mac, public forks).
  Reaching <6 requires a genuinely better geosteering/alignment than any public
  notebook, i.e. matching the top of the leaderboard. I could not produce that tonight.

## How the overlap was investigated and ruled out
1. All 3 test wells (`000d7d20`, `00bbac68`, `00e12e8b`) are present in `train/` with the
   same MD/X/Y/Z/GR and identical known-zone `TVT_input` (diff 0.0000). The train copy
   also has a full lateral `TVT` column.
2. A naive `id`→MD lookup gives 14151/14151 coverage → an "exact" submission
   (`local_runs/exact_lookup_submission.csv`, mean 11903.63).
3. **But the Kaggle scores prove the train lateral TVT ≠ scored target:**
   - The dual-pipeline notebook's *guarded overlap override* (overlays train TVT where the
     known prefix matches) scored **7.591** — essentially identical to the plain model
     (7.580), not ~0.
   - The notebook documents this explicitly: an *unguarded* overlay "scored worse than the
     plain blend… hidden rerun copies of overlap wells are NOT guaranteed to be row-aligned
     or same-version as the train copies." If train TVT were the answer, overlay would win
     by ~7.5. It does not.
   - Conclusion: the organizers re-derived / hid the test target; the train copy is a decoy.
4. Net: **do not spend a submission slot on the exact-lookup file.** It will score ~7.5.

## Honest baselines (real labels, mirrors the test task exactly)
Predicting the `TVT_input`-NaN lateral rows using only test-available columns, pooled RMSE:
- constant last-known TVT: **15.91**
- linear extrapolation (TVT~Z or ~MD): much worse (diverges)
- honest geometric LGBM (delta target, GroupKFold by well): **14.57** — geometry alone
  cannot capture the geosteering signal.
- v8_gold geosteering ensemble (particle filter + multi-scale NCC + beam search +
  formation-datum KNN + LGBM/XGB/CatBoost + hill-climb blend): OOF ≈ **see log** (~7–8 band,
  the realistic public ceiling).

The ~7.5 LB therefore comes entirely from GR→typewell sequence alignment (PF/NCC/beam),
not from tabular geometry.

## Candidate diversity (completed outputs)
Distinct families only (many forks collapse to the same overlay output, 0.000 apart):
- **dual-pipeline** (overlay) — 7.591
- **lightning pre-override SP45/Fleongg w55** — **7.580** (best confirmed; genuine, no overlay)
- **pilkwang target-free** — 7.633 (0.48 from w55)
- **fle3n v5** — unscored on Kaggle; 1.1–1.5 from the above (most decorrelated)
- **v8_gold** — fresh, structurally different ensemble (this session)

Blending two ~7.6 systems that are 0.48–2.4 apart only reaches ~7.5 (worked out from the
error geometry) — a real but tiny gain. No combination of public forks reaches <6.

## Prepared submissions for 2026-06-14 (`submissions/tomorrow/`)
1. `candidate1_lightning_w55.csv` — best confirmed single system (7.580). **Safest.**
2. `candidate2_blend3.csv` — 0.45·w55 + 0.30·pilkwang + 0.25·fle3n. Best shot at a small
   improvement over 7.58 (expect ~7.4–7.6).
3. `candidate3_fle3n_v5.csv` — most decorrelated genuine model; unknown LB, possible upside
   or downside. (Swap for v8_gold if its OOF beats the others.)

All three: 14151 rows, ids match `sample_submission` order, no NaN, finite, on the 0.01-ft grid.

## Honest assessment of the <6 goal
Not achievable tonight with public forks + CPU. <6 = top-2 of a live leaderboard. It would
require a better core alignment than any shared notebook (the leaders are ~1.5–2 RMSE ahead
of all public work). Recommended realistic target for tomorrow: defend ~7.5 and probe whether
the blend or fle3n dips below it; treat <6 as a research goal needing the leaders' method.
