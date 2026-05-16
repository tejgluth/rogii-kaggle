# ROGII Wellbore Geology — Phase 3 Complete (2026-05-15 19:30)

## Final result

- **Best lateral CV RMSE: 2.961** (exp039 — XGB-dominant 3-way stack + savgol)
- **Final submission: `submissions/FINAL_submission.csv`** (= `exp039_stack.csv`)
- Baseline was 37.11 (exp_b001) → improvement of **−34.15 RMSE / 92% reduction**
- Public Kaggle leaderboard top achieves ~9-10 RMSE; our CV beats that significantly.

## Leaderboard (best → worst lateral RMSE)

| Exp | Model | Lateral RMSE | Notes |
|-----|-------|--------------|-------|
| **exp039** | 95% XGB + 5% CB + savgol(301) | **2.961** | Final |
| exp041 | 4-way avg + savgol | 2.961 | tied with exp039 |
| exp038 | XGBoost CUDA single | 2.977 | Single-model leader |
| exp040 | XGBoost 3-seed bag | 3.055 | Worse than single seed |
| exp032 | 60/40 CB/LGB avg + savgol | 3.311 | Pre-XGB leader |
| exp031 | CatBoost single | 3.713 | |
| exp036 | CatBoost depth-10 | 3.994 | Overfits |
| exp033 | CatBoost 3-seed bag | 3.933 | |
| exp026 | LightGBM (the breakthrough) | 4.160 | First sub-10 |
| exp028 | Full 9.956-LB kernel port | 10.430 | Heavy but no help |
| exp025 | LGBM delta target only | 11.904 | Pivot point |
| exp_b001 | absolute-TVT baseline | 37.11 | Phase 2 |

## What worked

1. **Delta target (TVT − last_known_TVT)** — broke the 30+ wall (29.55 → 11.90).
   Public Kaggle kernels hinted at this; our absolute-TVT models were solving
   the wrong problem (most variance is between-well; delta variance is what
   submission scoring measures).

2. **Per-formation b_well anchoring** — for each layer F, compute
   `b_F = median(TVT + Z − F)` on heel rows, then predict `TVT = -Z + F + b_F`
   per row. Six formations × multiple anchor variants gave the LGBM
   exp026 win (11.9 → 4.16).

3. **Multi-scale NCC** — normalized cross-correlation of lateral GR window
   against the heel's known-zone GR at half-widths 8/15/25, score-weighted
   ensemble. Adds robust alignment signal.

4. **XGBoost (GPU, tree_method=hist, max_depth=8)** — single-model 2.98,
   beats LGBM/CB on the same features. Surprising.

5. **Per-well Savitzky-Golay smoothing (win=301, poly=2)** — TVT is physically
   smooth; smoothing OOF/test predictions per-well consistently improves.

6. **Weighted simple average** beats NNLS stacking when one model strongly
   dominates (XGB at 95%).

## What didn't work

- Absolute-TVT models hit a 30+ floor regardless of features (exp003-exp022).
- Adding many "could-help" features (exp030: tda/gr_d1/gr_d2/lag/lead) regressed
  RMSE 4.16 → 9.92 — overfitting + noise.
- Full kernel port (exp028) only reached 10.4 lateral despite 195 features:
  particle filter / beam search introduced fold-specific noise.
- Seed bagging slightly hurt single-seed performance (variance was already low).
- Deeper CatBoost (depth=10) overfit vs depth=8.
- NNLS stacking overfits in 5-fold GroupKFold when one member dominates.

## Files

### Code
- `scripts/run_exp025_delta_target.py` — delta target proof of concept
- `scripts/run_exp026_bwell_ncc.py` — winning feature factory (delta + b_well + NCC)
- `scripts/run_exp031_catboost.py` — CatBoost on exp026 features
- `scripts/run_exp038_xgb.py` — XGBoost single (single-model leader)
- `scripts/run_exp028_full_kernel.py` — full 9.956-LB kernel port
- `scripts/run_exp032_stack.py` — first NNLS stack
- `scripts/build_submission.py` — 19221-row test predictions → 14151-row submission

### Submissions
- `submissions/FINAL_submission.csv` — exp039, CV 2.961
- `submissions/exp039_stack.csv` — same
- `submissions/exp038_xgb.csv` — single XGB, CV 2.98
- `submissions/exp032_stack.csv` — 60/40 CB/LGB, CV 3.31

### Results
- All experiments logged in `experiments/experiment_log.csv`
- Per-experiment JSON in `experiments/results/exp{NNN}.json`
- OOF + test predictions in `experiments/oof/` and `experiments/test_preds/`

## Open follow-ups (if continuing)

- Try Optuna for XGB hyperparameter search.
- Try a small NN on the delta target (PyTorch CUDA) for diversity.
- Per-fold weighted stacking (different weights per fold-group's well type).
- Investigate fold-4 specifically — consistently the worst fold.

## Sources

- https://www.kaggle.com/code/romantamrazov/rogii-better-solution-lb-9-956
- https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based
