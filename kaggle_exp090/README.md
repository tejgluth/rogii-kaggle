# exp090 Kaggle Notebook Package

This folder contains a kernel-only Kaggle notebook for the ROGII competition.
It trains from Kaggle's competition input files at runtime and writes
`/kaggle/working/submission.csv`.

Files:

- `exp090_submission.ipynb` - upload/push this Kaggle notebook.
- `exp090_kernel.py` - same code as the notebook, kept for review and editing.
- `kernel-metadata.json` - Kaggle CLI metadata for `kaggle kernels push`.

The notebook is intentionally not a wrapper around the local CSV. ROGII swaps in
a hidden test set during scoring, so hardcoded local predictions are not valid.

## Pipeline

- Build exp026 feature matrix from raw train/test well CSVs.
- Train exp063 and exp051 XGBoost base models.
- Recreate the selected exp050, exp056, exp070, exp075, and exp082 blends.
- Train exp083, exp085, exp087, and exp089 LightGBM residual stages.
- Apply the exp084, exp086, and exp090 blend/smoothing weights.
- Write `submission.csv` in sample-submission order.

## Push And Submit

After refreshing Kaggle API credentials in `.env`:

```bash
set -a
source .env
set +a

/Users/speakeasy/Library/Python/3.12/bin/kaggle kernels push -p kaggle_exp090
/Users/speakeasy/Library/Python/3.12/bin/kaggle kernels status tejasgluth/rogii-exp090-submission
```

When the kernel run is complete, submit its output:

```bash
/Users/speakeasy/Library/Python/3.12/bin/kaggle competitions submit \
  -c rogii-wellbore-geology-prediction \
  -f submission.csv \
  -k tejasgluth/rogii-exp090-submission \
  -v <VERSION> \
  -m "exp090 full kernel pipeline"
```

For manual upload, create a Kaggle notebook with GPU enabled, attach the ROGII
competition data source, paste/upload `exp090_submission.ipynb`, commit it, and
submit the generated `submission.csv` from the completed notebook version.
