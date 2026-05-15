# Phase 1: Exploratory Data Analysis

Write and run a complete EDA script. Save output to `experiments/results/eda_report.md`.

## What to Explore

### Data Structure
- List all files in `data/raw/train/` and `data/raw/test/`
- Identify file formats (CSV, LAS, or both)
- For each well: how many depth samples? What columns exist?
- How many unique wells in train vs test?

### Target Variable (TVT)
- Distribution of TVT values (mean, std, min, max, percentiles)
- TVT range per well — do different wells have different TVT ranges?
- Plot TVT vs depth for 3 sample wells

### Gamma Ray (GR) Log
- GR distribution (mean, std, min, max)
- GR range per well
- Plot GR vs depth for same 3 wells
- Correlation between GR and TVT

### Typewell Reference
- How many Typewell files exist?
- What depth range does the Typewell cover?
- Does the Typewell GR range match the lateral well GR range?

### Trajectory Data
- Columns available: MD, TVD, inclination, azimuth, X, Y, Z?
- Correlation between TVD and TVT
- Dog-leg severity (rate of inclination change) — is it significant?

### Missing Data
- Any NaN values in GR, TVT, trajectory columns?
- How to handle them?

### Well-Level Statistics
- How much does TVT vary between wells?
- Are test wells geologically similar to train wells?
- Any outlier wells?

## Output

Write a markdown report to `experiments/results/eda_report.md` with:
1. Data shape summary table
2. Key findings (5-10 bullet points)
3. Feature engineering ideas suggested by the data
4. Any data quality issues to watch out for

Also save summary statistics to `experiments/results/eda_stats.json`.
