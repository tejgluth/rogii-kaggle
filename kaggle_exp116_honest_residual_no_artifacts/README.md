# ROGII exp116 honest residual no-artifacts submission

Self-contained Kaggle notebook that uses only the official competition input. It trains the exp092-style base ensemble at runtime, then replaces the fixed residual correction with a well-level validation protocol:

- excludes any train wells that appear in the runtime test set
- reserves deterministic holdout wells for reporting only
- selects residual weight and smoothing on GroupKFold OOF predictions from tuning wells
- applies a predeclared residual guard, so tiny tuning-only improvements do not trigger a noisy correction
- reports the selected recipe once on the untouched holdout wells
- retrains the residual model on all eligible non-test wells before writing `submission.csv`

Recommended Kaggle settings:

- Input: competition only, `rogii-wellbore-geology-prediction`
- Accelerator: GPU P100 or GPU T4 x2
- Internet: Off
- Persistence: Files only
- Dependency manager: no manual installs initially

This is slower than cached-model notebooks because it trains all base models from scratch. Use the printed `holdout_selected_rmse` versus `holdout_base_rmse` as the honest local signal for whether the residual selection helped.
