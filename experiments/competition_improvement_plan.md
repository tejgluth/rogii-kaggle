# Fair RMSE Improvement Plan For ROGII

## Summary

- Keep `exp115` as the current clean Kaggle-ready candidate: RMSE `6.7761`
  on the local clean holdout, using only competition input plus
  `ravaghi/wellbore-geology-prediction-artifacts`. `exp114` is the previous
  smoothed-component candidate at RMSE `6.7768`; `exp109` remains the raw
  residual-stack baseline at RMSE `6.8267`.
- Improve only with leak-safe experiments: no training, model selection, blend
  weighting, feature selection, or postprocess tuning on the sample/heldout
  wells.
- Treat the current heldout audit as a sanity check, not as the optimization
  target, because repeated inspection can indirectly overfit.
- Prefer a self-contained notebook when practical, but allow the artifact
  dataset when it materially improves the honest validation result.

## Clean Validation Protocol

- Freeze sample-submission wells `000d7d20`, `00bbac68`, and `00e12e8b` as
  audit-only.
- Select models, residual weights, blend weights, features, and postprocess
  parameters using nested GroupKFold by well on all non-heldout wells.
- Promote only candidates that improve the non-heldout nested/CV criterion and
  do not obviously degrade the audit.
- Record every failed experiment so saturated paths are not re-tested.
- Report row-weighted RMSE, well-equal RMSE, per-well RMSE, bias, and
  submission validity for every candidate.

## Experiment Queue

- Ensemble weighting:
  - Test convex, nonnegative blends of the seven artifact members selected
    only by nested non-heldout CV.
  - Try robust blend objectives: Huber residual loss, per-well equal weighting,
    and uncertainty-weighted rows using artifact disagreement.
  - Do not use the heldout result to choose weights.
- Well-level residual correction:
  - Extend exp100's `gbr_d4` with CatBoost/LightGBM/ExtraTrees/HistGBR
    well-summary residual models.
  - Add spatial neighborhood features from well-level XYZ summaries because
    nearby wells and dip direction can help.
  - Use outer-fold selection to avoid repeating exp101/103-style dev-CV wins
    that failed on audit.
- Row-level residual correction:
  - Continue from exp109's compact ridge family, but test HuberRegressor,
    ElasticNet, BayesianRidge, small LightGBM, and monotonic/spline residual
    smoothers.
  - Add interaction features only when CV-stable: artifact disagreement, row
    fraction, GR rolling texture, DTW/beam confidence, and base-vs-last-known
    deltas.
  - Penalize candidates that only improve row-weight tuning but increase
    per-well bias.
- Physics/postprocess:
  - Test smoothness postprocesses selected inside nested CV: Savitzky-Golay,
    median, spline, and total-variation denoising.
  - Constrain postprocesses to per-well prediction sequences only; no
    label-derived corrections from heldout/sample wells.
  - Prefer conservative shrinkage toward exp109 when postprocess gains are
    small.
- Public-method ideas:
  - Revisit DWT/wavelet and particle/beam/DTW routes from the public `9.674`
    artifact guide, but only as new OOF features or clean base members.
  - Use the task-brief insight that horizontal GR before prediction start may
    correlate better with later lateral GR than the vertical typewell alone.
  - Explore spatial dip models from nearby training wells, selected only by
    group CV.

## Test Plan

- For every experiment:
  - Run the script with `.venv/bin/python`.
  - Write `experiments/results/expNNN.json` with full selection metadata.
  - Write candidate CSV under `submissions/`.
  - Confirm sample row count, ID order, finite predictions, and no missing TVT.
- For a promoted candidate:
  - Update `submissions/FINAL_submission.csv`.
  - Build a Kaggle-ready notebook that uses only the competition input and
    needed artifact dataset.
  - Execute the notebook locally and verify its `submission.csv` exactly
    matches the promoted CSV.
  - Strip notebook outputs, update checkpoint, commit, pull with
    rebase/autostash, and attempt push.

## References

- ROGII task brief:
  https://github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md
- Current public DWT/artifact notebook in repo:
  `rogii-dwt-artifact-ensemble-guide-9-674.ipynb`
