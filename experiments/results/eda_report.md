# exp001 EDA Report

## Data Shape Summary

| Dataset | Files | Rows |
| --- | --- | --- |
| train horizontal wells | 773 | 5,092,255 |
| test horizontal wells | 3 | 19,221 |
| train typewell files | 773 | 1,567,045 |

## Key Findings

- The dataset has 773 train horizontal wells, 3 test horizontal wells, and 773 train typewell files.
- Train horizontals contain 5,092,255 rows; test horizontals contain 19,221 rows.
- Test horizontal wells are few (3) relative to train, so validation should be grouped by well rather than row-random.
- TVT ranges from 9,245.1900 to 12,893.8900, with mean 11,503.6440 and std 639.9711.
- Per-well TVT range varies from 158.9300 to 1,256.7000, indicating different target span difficulty by well.
- Train GR mean/std are 87.7524/23.7683; test GR mean/std are 76.4598/30.4902.
- Mean per-well GR/TVT correlation is -0.2314 with std 0.2531, so raw GR alone is not a stable linear target proxy.
- 100.0% of train typewell TVT ranges overlap the overall train horizontal MD range.
- Typewells use `TVT` as their depth-like column and include optional `Geology`; horizontals use `MD` plus trajectory columns.
- Some columns contain missing values and should be guarded before feature generation.

## Feature Engineering Ideas

- Use grouped validation by `well_id` to estimate generalization to unseen wells.
- Create local GR window features: rolling mean, std, min, max, slope, and percentile rank along measured depth.
- Use `TVT_input` cautiously as a strong baseline feature; verify whether its semantics are available at inference without leakage.
- Add trajectory features from `MD`, `X`, `Y`, and `Z`: deltas, local inclination proxies, curvature proxies, and depth-normalized movement.
- Align horizontal wells to typewell GR signatures with depth offsets or dynamic time warping style features.
- Normalize GR per well and possibly per nearby typewell to reduce well-specific scale shifts.

## Data Quality Issues

- Typewell and horizontal files have different schemas; feature code must branch by file type.
- `Geology` exists in typewells only and is sparse, so it is not a direct horizontal-well feature.
- The test set has only three horizontal wells, making leaderboard sensitivity likely.
- Depth meaning differs by file family: horizontals expose `MD`, while typewells expose `TVT`.
- Missing values are present in: ANCC, EGFDL, GR, TVT_input.

## TVT Distribution

- Mean: 11,503.6440
- Std: 639.9711
- Min / p5 / p25 / p50 / p75 / p95 / max: 9,245.1900 / 10,679.0100 / 10,987.9300 / 11,354.5100 / 12,038.2600 / 12,669.4200 / 12,893.8900
- Per-well TVT range mean/std/min/max: 734.4689 / 175.4844 / 158.9300 / 1,256.7000
