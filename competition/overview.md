# Competition Overview

- **Name**: ROGII — Wellbore Geology Prediction
- **URL**: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- **Organizer**: ROGII (geosteering software company)
- **Metric**: RMSE (Root Mean Squared Error) — lower is better
- **Task**: Regression — predict TVT for every depth sample

## Leaderboard Reference Points

| RMSE | Meaning |
|------|---------|
| ~15  | Public baseline (simple models) |
| ~12  | LightGBM with basic features |
| ~10  | Top quartile range |
| <8   | Strong result |
| <5   | Top 10% likely |

Update this table as you make submissions.

## Timeline

Check competition page for current deadline:
https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

## Submission Format

CSV file with columns: `[row_id, tvt]` or similar.
Check `data/raw/sample_submission.csv` for exact format after downloading data.

## Key Rules

- Public LB = fraction of test data. Don't overfit to public LB.
- Max 5 submissions per day.
- Teams of up to 3 (solo is fine).
