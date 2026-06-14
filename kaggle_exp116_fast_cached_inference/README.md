# ROGII exp116 fast cached inference

Loads the private `tejasgluth/rogii-exp116-fast-model-cache` dataset, rebuilds official competition test features dynamically, predicts with cached LightGBM/CatBoost fold models, skips the rejected residual correction, and writes `/kaggle/working/submission.csv`.
