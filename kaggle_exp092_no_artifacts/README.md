# ROGII exp092 no-artifacts submission

Competition-data-only version of exp092. This package intentionally does **not** attach or use `ravaghi/wellbore-geology-prediction-artifacts`. It builds train/test features from the official competition files, trains the LightGBM/CatBoost ensemble at runtime, applies the exp092 test-safe residual correction, validates the output, and writes `/kaggle/working/submission.csv`.

Recommended Kaggle settings:

- Input: competition only, `rogii-wellbore-geology-prediction`
- Accelerator: GPU T4 x2 or P100
- Internet: Off
- Persistence: Files only
- Dependency manager: no manual installs initially

Expected runtime is much longer than artifact-backed notebooks because it trains all base models from scratch. Based on previous no-artifact runs, expect multiple hours.
