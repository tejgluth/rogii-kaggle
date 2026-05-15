# Codex Worker Instructions

You are a Codex worker agent. You have been given a specific ML experiment
to implement and run. Follow these rules exactly.

## Rules

1. **Read the experiment spec** — it tells you exactly what to build
2. **Use GPU everywhere** — `device="cuda"` for all models, `cudf` for data loading
3. **GroupKFold by well_id** — never mix wells across folds
4. **Save all output files** — OOF, test preds, and result JSON
5. **Write clean, runnable Python** — no pseudocode, no TODOs
6. **Run the code** — don't just write it, execute it
7. **Verify outputs exist** — check file sizes before finishing
8. **Write result JSON** — include cv_rmse, notes, and all required fields

## File Structure

```
ROOT/
├── src/               ← import utilities from here
│   ├── data_loader.py
│   ├── features.py
│   └── evaluate.py
├── data/
│   ├── raw/train/     ← source data
│   └── processed/     ← write processed data here
└── experiments/
    ├── oof/           ← write OOF predictions here
    ├── test_preds/    ← write test predictions here
    └── results/       ← write result JSON here
```

## Import Pattern

```python
import sys
sys.path.insert(0, ".")  # run from ROOT

from src.data_loader import load_wells
from src.features import build_features
from src.evaluate import compute_cv_rmse
```

## Completion Signal

Print this exact line as your final output:
```
EXPERIMENT COMPLETE: cv_rmse=X.XXXX
```

If the experiment failed, print:
```
EXPERIMENT FAILED: <reason in one line>
```
