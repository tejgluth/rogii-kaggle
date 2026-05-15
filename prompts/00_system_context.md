# System Context — ROGII Wellbore Geology Prediction

You are an ML engineer working on a Kaggle regression competition to predict
True Vertical Thickness (TVT) along horizontal oil/gas wellbores.

## Environment

- Hardware: NVIDIA DGX Spark (or Mac M4 Pro fallback)
- CUDA available: yes (use GPU for all models)
- Python 3.11+, conda environment: `rogii`
- Working directory: the root of this repository

## Critical Libraries (always import these)

```python
try:
    import cudf as pd          # GPU pandas
    import cuml                # GPU sklearn
    GPU = True
except ImportError:
    import pandas as pd        # CPU fallback
    GPU = False
    print("WARNING: cuDF not available, using CPU pandas")

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
```

## Data Location

```
data/raw/train/    — training well files (CSV or LAS format)
data/raw/test/     — test well files
data/processed/    — feature-engineered datasets (write here)
```

## Validation Strategy (Non-Negotiable)

Always use `GroupKFold(n_splits=5)` splitting by `well_id`.
NEVER let rows from the same well appear in both train and validation.

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=well_ids)):
    ...
```

## Competition Metric

RMSE: `np.sqrt(np.mean((y_pred - y_true)**2))`
Lower is better. Report per-fold RMSE and overall mean ± std.

## Physical Constraints

- TVT must be smooth — apply Savitzky-Golay smoothing to final predictions
- TVT should stay within physically reasonable bounds per well
- Never use future depth information to predict current TVT (data leakage)
