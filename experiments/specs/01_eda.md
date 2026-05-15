# Phase 1: Exploratory Data Analysis

Run this only after `data/raw/` contains the Kaggle competition files.

## Goal

Write and run a complete EDA script for the ROGII wellbore geology prediction competition. Save the report and machine-readable stats for later modeling decisions.

## Required Outputs

- `experiments/results/eda_report.md`
- `experiments/results/eda_stats.json`
- Any generated plots should be saved under `experiments/results/eda_plots/`

## What to Explore

### Data Structure

- List all files under `data/raw/`, including nested train/test/typewell folders.
- Identify file formats and naming patterns.
- For each well, report the number of depth samples and available columns.
- Count unique wells in train and test.

### Target Variable

- Summarize TVT values: mean, standard deviation, min, max, and percentiles.
- Report TVT range by well.
- Plot TVT versus depth for three representative training wells.

### Gamma Ray Log

- Summarize GR values: mean, standard deviation, min, max, and percentiles.
- Report GR range by well.
- Plot GR versus depth for the same three wells.
- Estimate the relationship between GR and TVT where both are available.

### Typewell Reference

- Count Typewell files.
- Report Typewell depth coverage.
- Compare Typewell GR distribution and depth range against lateral wells.

### Trajectory Data

- List available trajectory columns, such as MD, TVD, inclination, azimuth, X, Y, and Z.
- Measure the relationship between TVD and TVT if both exist.
- Estimate dog-leg severity or inclination change rate if trajectory columns support it.

### Missing Data

- Report missing values for GR, TVT, trajectory columns, and well identifiers.
- Recommend handling strategies for fields with material missingness.

### Well-Level Statistics

- Quantify TVT and GR variation between wells.
- Compare train and test distributions where feature columns overlap.
- Flag outlier wells and explain why they look unusual.

## Report Contents

`eda_report.md` must include:

1. A data shape summary table.
2. Five to ten high-signal findings.
3. Feature engineering ideas suggested by the data.
4. Data quality issues to watch during modeling.
5. A short next-step recommendation for the first baseline model.

`eda_stats.json` must include:

- File inventory.
- Well counts.
- Per-column summary statistics.
- Missing-value counts.
- Per-well summary statistics.
- Paths to generated plots.

## Execution Rules

- Run from the repository root.
- Prefer existing helpers in `src/` if they work with the downloaded data.
- Keep the EDA script in `experiments/results/eda_script.py` for reproducibility.
- If the raw file layout differs from expectations, adapt the loader and document the observed layout in the report.
- Do not train a model in this phase.
