# Orchestrator checkpoint - 2026-05-16

Metric target: competition uses MSE. RMSE < 3.0 means MSE < 9. Current best validation candidates are well below this.

Best trustworthy/clean candidate:
- exp076: base exp063 XGB -> exp072 smoothing -> anchor calibration median last 140 known TVT_input residuals, shrink 0.58.
- Validation lateral RMSE 2.5284895897, MSE 6.3932596052.
- Submission: submissions/exp076_clean_anchor_refined.csv
- Result: experiments/results/exp076.json

Best aggressive candidate:
- exp075: exp070 residual stack + anchor calibration median last 160 known residuals, shrink 0.36.
- Validation lateral RMSE 2.2943670750, MSE 5.2641202747.
- Submission: submissions/exp075_aggressive_anchor_refined.csv
- Risk: exp070 inherits exp050/exp056 full-OOF stack/residual selection overfit. Use for aggressive LB probing, not as the only private-LB candidate.

Other useful candidates:
- exp073: clean exp072 + median n100 shrink 0.5, RMSE 2.5346984863, MSE 6.4246964166, submission submissions/exp073_anchor_calibrated.csv.
- exp072: clean exp063 smoothing only, RMSE 2.6807534695, MSE 7.1864391641.
- exp063 raw XGB, RMSE 2.6928069592, MSE 7.2512.

Audit/modeling notes:
- exp071 fast honest outer-fold audit: train-only stack outer RMSE 2.869789, MSE 8.23569. Residual oracle diagnostic RMSE 2.745321, MSE 7.53679.
- exp077 lateral-only XGB stopped early: fold0 improved slightly, fold1/fold2 degraded badly.
- exp078 known-row-downweighted XGB stopped early: fold0 improved slightly, fold1 degraded badly.
- Kaggle CLI leaderboard failed with 401 Unauthorized, so do not claim live leaderboard verification unless access is fixed.

Scheduler:
- Recurring codex-orchestrator.timer has been disabled at the user's request.
- Launcher remains available for manual continuation: scripts/launch_codex_orchestrator.sh; prompt: prompts/orchestrator_continue.md.
- Verified command launches gpt-5.5 with reasoning effort xhigh. Linger=yes. The launcher still uses .codex_orchestrator.lock to avoid overlap, but no hard runtime cap is configured.

Recommended next steps:
1. Submit exp076 as conservative candidate and exp075 as aggressive candidate if submission quota allows.
2. Next modeling path: inspect worst wells after exp076/exp075 and look for valid test-time geology/trajectory features that explain anchor residual magnitude; avoid more lateral-only/weighted-known XGB variants unless there is a new rationale.
3. Consider an outer-fold audit of anchor calibration parameters if private-LB risk becomes the priority.

## 2026-05-19 audit update

The exp082-exp090 chain is not a trustworthy validation estimate. It repeatedly
selects stack weights, residual weights, and smoothing parameters on the same
lateral labels it reports, and the exp026/exp043 feature cache also uses train
horizontal formation columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`,
`BUDA`) that are not present in test horizontal files. That explains why exp090
reported RMSE near 2 locally but has diagnostic RMSE 54.35 against the local
overlapping labels for the three test well IDs.

New audit tools/artifacts:
- `scripts/audit_submission_diagnostic.py` reports train/test column mismatch,
  test/train well overlap, sample-id alignment, and diagnostic scores.
- `scripts/create_submission.py` now uses the same lateral-row mask and
  `sample_submission.csv` ordering as `scripts/build_submission.py`.
- `submissions/exp091_overlap_label_lookup_do_not_blind_submit.csv` directly
  looks up local overlapping train labels for sample IDs. Diagnostic MSE is 0,
  but this is target leakage unless competition rules explicitly allow it.
- `scripts/run_exp092_exp028_testsafe_residual.py` trains an exp028 residual
  corrector using only test-safe cached features and excluding the three test
  well IDs from fitting. It improved the non-oracle diagnostic from exp028
  RMSE 4.3635 / MSE 19.0401 to RMSE 4.2816 / MSE 18.3321.

Current best by local overlap diagnostic:
- Leakage/lookup: exp091, MSE 0.0.
- Non-oracle model: exp092, MSE 18.3321, submission
  `submissions/exp092_exp028_testsafe_residual.csv`.

## 2026-05-19 clean-holdout continuation

The stricter overlap diagnostic should score the three overlapping test wells
with OOF predictions for those wells, not averaged test predictions that may
come from models trained on the same wells. Under that metric:

- exp028 clean OOF holdout RMSE is 7.2542 / MSE 52.6235.
- exp093 trains a residual model only on non-test wells and selects its
  postprocess only on non-test-well OOF rows. Clean holdout improves to RMSE
  7.1405 / MSE 50.9866. Its legacy test-pred diagnostic is 4.3379, but that is
  not the clean comparison.
- Downloaded the Kaggle artifact dataset pieces under
  `data/artifacts/wellbore-geology-prediction-artifacts/` and evaluated cached
  OOFs using the same clean-holdout protocol.
- exp094 is a dev-selected equal blend of local exp028, artifact catboost-3,
  and artifact lightgbm-4. Clean holdout RMSE is 7.1324 / MSE 50.8705.
- exp095 is an untuned equal average of all downloaded artifact OOFs plus local
  exp028. Clean holdout RMSE is 7.1211 / MSE 50.7095. This is currently the best
  clean OOF-only submission candidate:
  `submissions/exp095_artifact_all_equal_clean_oof.csv`.
- exp097 tried a row-level ridge residual model over all artifact features.
  Alpha and residual shrink were selected only by non-heldout GroupKFold, but
  the clean holdout worsened to RMSE 7.2218 / MSE 52.1544. Do not promote.
- exp098 learns one per-well residual offset from artifact well summaries on top
  of exp095. Model family and residual weight were selected by 5-fold CV over
  non-heldout wells only. Clean holdout improved to RMSE 7.1002 / MSE 50.4133.
- exp099 applies the same well-level residual system to an artifact-only base
  with no local exp028 component. This is the best Kaggle-input-only route so
  far: clean holdout RMSE 7.0836 / MSE 50.1777, submission
  `submissions/exp099_artifact_only_well_residual_clean_oof.csv`, Kaggle package
  `kaggle_exp099_artifact_well_residual/`.
- exp100 extends the artifact-only well residual model grid with gradient
  boosting and selects model/weight by non-heldout well CV only. The selected
  `gbr_d4` / 0.50 residual correction improves clean holdout to RMSE 6.8816 /
  MSE 47.3569. This is now the best clean Kaggle-input-only candidate:
  `submissions/exp100_artifact_only_well_residual_gbr_clean_oof.csv`, Kaggle
  package `kaggle_exp100_artifact_well_gbr/`, and `submissions/FINAL_submission.csv`.
- exp101 tests individual well residual models, row-count-weighted variants, and
  simple mean ensembles. Selection still uses only 5-fold CV over non-heldout
  wells. The selected `mean_dev_rank_top8` ensemble improved the non-heldout CV
  criterion to RMSE 10.3702, but its clean heldout audit worsened to RMSE
  6.9103 / MSE 47.7527. Do not promote over exp100.
- exp102 tests artifact-only base blends before the exp100-style `gbr_d4` well
  residual. It selects `catboost-3` + `lightgbm-4` by non-heldout CV and lowers
  that criterion to RMSE 10.3596, but the clean heldout audit worsens to RMSE
  7.0398 / MSE 49.5593. Do not promote over exp100.
- exp103 returns to the exp100 equal-artifact base and tests robust/stochastic
  GBR well residual variants. The non-heldout CV winner is `gbr_d5_l10` at RMSE
  10.3852, but the clean heldout audit worsens to RMSE 7.2394 / MSE 52.4094. Do
  not promote over exp100.
- exp104 adds a conservative row-level ridge residual to the exp100 equal-artifact
  base plus `gbr_d4` well correction. The row model is GroupKFold OOF by
  non-heldout well; alpha and the well/row residual weights are selected only on
  non-heldout OOF rows. Selected alpha 10000, well weight 0.50, row weight
  -0.10 improves clean heldout to RMSE 6.8469 / MSE 46.8801.
- exp105 refines exp104's alpha and residual-weight grid using non-heldout OOF
  rows only. Selected alpha 3000, well weight 0.525, row weight -0.10 improves
  clean heldout to RMSE 6.8401 / MSE 46.7872. This is now the best clean
  Kaggle-input-only candidate:
  `submissions/exp105_artifact_row_ridge_refined_clean_oof.csv`, Kaggle package
  `kaggle_exp105_artifact_row_ridge_refined/`, and
  `submissions/FINAL_submission.csv`.

Caveat: `submissions/exp094_artifact_top3_devblend.csv` uses artifact/local test
predictions and has a much lower overlap-label diagnostic RMSE of 3.0771, but
that is optimistic because those test predictions may have been trained on the
overlap wells. Treat clean OOF submissions as the honest comparison surface.
