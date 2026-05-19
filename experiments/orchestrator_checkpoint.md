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
