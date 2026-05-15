# EDA Task for exp001

Write and run a Python script at `experiments/scripts/run_eda.py` that performs
full exploratory data analysis on the ROGII competition data.

## Data paths (actual locations on disk)
- Train wells: `data/raw/rogii-wellbore-geology-prediction/train/`
- Test wells:  `data/raw/rogii-wellbore-geology-prediction/test/`
- File naming: `{hash}__horizontal_well.csv` and `{hash}__typewell.csv`
- Also PNG images (ignore those)

## What to analyze and print

### 1. File inventory
- Count horizontal_well files in train and test
- Count typewell files in train
- Print a sample filename for each type

### 2. Column structure
- Load one horizontal_well CSV and one typewell CSV
- Print columns, dtypes, shape, head(3) for each

### 3. Train well statistics (loop over ALL train horizontal wells)
- Collect: well_id, n_rows, depth_min, depth_max, gr_mean, gr_std, gr_min, gr_max, tvt_mean, tvt_std, tvt_min, tvt_max
- Print summary statistics (mean/std/min/max across all wells)
- Print per-well TVT range (max-min) — sorted descending

### 4. Test well statistics (loop over ALL test horizontal wells)
- Same columns EXCEPT tvt (test has no target)
- Print shape and GR stats

### 5. Typewell statistics
- For each typewell: depth range, GR mean/std, number of rows
- Do typewell depth ranges overlap with horizontal well depths?

### 6. Missing values
- For train wells: count NaNs in each column across all wells
- For test wells: same

### 7. TVT distribution
- Overall: mean, std, min, max, p5, p25, p50, p75, p95
- Per-well TVT range statistics

### 8. GR correlation with TVT
- Compute correlation(GR, TVT) for each train well
- Print mean and std of these correlations

## Output files to create

1. `experiments/results/eda_report.md` — markdown report with:
   - Data shape summary table (train wells, test wells, typewell files)
   - Key findings (8-10 bullet points)
   - Feature engineering ideas based on findings
   - Data quality issues

2. `experiments/results/eda_stats.json` — machine-readable stats:
   ```json
   {
     "n_train_wells": ...,
     "n_test_wells": ...,
     "n_typewell_files": ...,
     "train_total_rows": ...,
     "test_total_rows": ...,
     "train_columns": [...],
     "test_columns": [...],
     "typewell_columns": [...],
     "tvt_mean": ...,
     "tvt_std": ...,
     "tvt_min": ...,
     "tvt_max": ...,
     "gr_mean": ...,
     "gr_std": ...,
     "gr_tvt_corr_mean": ...,
     "gr_tvt_corr_std": ...,
     "has_missing_values": true/false,
     "missing_value_columns": [...]
   }
   ```

3. `experiments/results/exp001.json` — result metadata:
   ```json
   {
     "experiment_id": "exp001",
     "model": "none",
     "cv_rmse": null,
     "cv_rmse_std": null,
     "features_used": [],
     "n_features": 0,
     "training_time_seconds": 0,
     "notes": "<one-line summary of most important EDA finding>"
   }
   ```

4. Create placeholder files (empty numpy arrays) at:
   - `experiments/oof/oof_none_exp001.npy`
   - `experiments/test_preds/test_none_exp001.npy`

## After running, print this exact line:
```
EXPERIMENT COMPLETE: cv_rmse=None
```

Run the script now. Use pandas (not cudf) since this is EDA. Make sure
`experiments/scripts/` directory exists before writing the script.
