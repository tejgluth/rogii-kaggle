# ROGII Ensemble Submission Notebook

Use `rogii_ensemble_submission.ipynb` for the next Kaggle submission attempt.
This notebook is derived from `rogii-dwt-artifact-ensemble-guide-9-674.ipynb`
and is a safer reset after the reconstructed exp090 notebook scored poorly.

## Required Kaggle Inputs

Attach both inputs in the Kaggle notebook right sidebar:

- Competition input: `rogii-wellbore-geology-prediction`
- Dataset input: `ravaghi/wellbore-geology-prediction-artifacts`

The artifact dataset is important. It contains cached feature/model artifacts
used by the guide. Without it, the notebook may fall back to slower retraining
or miss the intended score path.

## Recommended Settings

- Accelerator: GPU T4 x2, or P100 if T4 is unavailable
- Internet: Off
- Persistence: Files only
- Pin original environment: Off unless Kaggle package versions break
- Dependency manager: no manual installs initially

## Expected Output

The notebook writes:

```text
/kaggle/working/submission.csv
```

The final validation cell checks shape, id order, finite predictions, and prints
the contents of the output directory.

## CLI Push

If Kaggle API credentials are fixed:

```bash
set -a
source .env
set +a

/Users/speakeasy/Library/Python/3.12/bin/kaggle kernels push -p kaggle_ensemble_guide
```

Then submit a completed version:

```bash
/Users/speakeasy/Library/Python/3.12/bin/kaggle competitions submit \
  -c rogii-wellbore-geology-prediction \
  -f submission.csv \
  -k tejasgluth/rogii-ensemble-submission \
  -v <VERSION> \
  -m "artifact ensemble guide"
```
