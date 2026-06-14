# ROGII Tomorrow Submission Plan - 2026-06-14

Goal: spend tomorrow's competition submission slots only on completed, scoreable, rerunnable notebooks with the best chance to break out of the public-notebook `7.5x` band.

## Check First

```bash
.venv/bin/kaggle competitions submissions rogii-wellbore-geology-prediction -v
.venv/bin/kaggle kernels list --mine --sort-by dateRun --page-size 30
.venv/bin/kaggle kernels status tejasgluth/rogii-shuaixu-exp098-top3-gpu0-retry-fork
.venv/bin/kaggle kernels status tejasgluth/rogii-opengs7-v25b-cb-ensemble-fork
.venv/bin/kaggle kernels status tejasgluth/rogii-quanzhong-pf-fleongg-blend-fork
.venv/bin/kaggle kernels status tejasgluth/rogii-zeyufeng-final-fork
```

Pull and validate any completed outputs:

```bash
.venv/bin/kaggle kernels output tejasgluth/rogii-shuaixu-exp098-top3-gpu0-retry-fork -p local_runs/rogii-shuaixu-exp098-top3-gpu0-retry-fork_output
.venv/bin/kaggle kernels output tejasgluth/rogii-opengs7-v25b-cb-ensemble-fork -p local_runs/rogii-opengs7-v25b-cb-ensemble-fork_output
.venv/bin/kaggle kernels output tejasgluth/rogii-quanzhong-pf-fleongg-blend-fork -p local_runs/rogii-quanzhong-pf-fleongg-blend-fork_output
.venv/bin/python scripts/analyze_submission_candidates.py
```

## Submit Order

Submit only if the kernel status is `COMPLETE` and its output audit is valid.

1. Shuaixu exp098 top-3 GPU system:

```bash
.venv/bin/kaggle competitions submit rogii-wellbore-geology-prediction \
  -k tejasgluth/rogii-shuaixu-exp098-top3-gpu0-retry-fork \
  -v 1 -f submission.csv \
  -m "shuaixu exp098 top3 GPU formation/NCC ensemble"
```

2. Opengs7 v25b CatBoost ensemble:

```bash
.venv/bin/kaggle competitions submit rogii-wellbore-geology-prediction \
  -k tejasgluth/rogii-opengs7-v25b-cb-ensemble-fork \
  -v 1 -f submission.csv \
  -m "opengs7 v25b CB ensemble GPU"
```

3. Quanzhong PF+Fleongg dynamic blend:

```bash
.venv/bin/kaggle competitions submit rogii-wellbore-geology-prediction \
  -k tejasgluth/rogii-quanzhong-pf-fleongg-blend-fork \
  -v 1 -f submission.csv \
  -m "quanzhong PF fleongg dynamic blend"
```

## Backups

If one of the top three fails or has invalid output:

```bash
.venv/bin/kaggle competitions submit rogii-wellbore-geology-prediction \
  -k tejasgluth/rogii-zeyufeng-final-fork \
  -v 1 -f submission.csv \
  -m "zeyufeng final fast PF backup"
```

Use V5F-family variants only after the structurally different systems above:

```bash
.venv/bin/kaggle competitions submit rogii-wellbore-geology-prediction \
  -k tejasgluth/rogii-fle3n-v5f-xfer-0p55 \
  -v 1 -f submission.csv \
  -m "v5f xfer 0.55 backup"
```

When a GPU slot frees, push and run V8 Gold as another backup:

```bash
.venv/bin/kaggle kernels push -p kaggle_z149_v8_gold_fork
```

## Do Not Use

- Static visible-output CSV kernels: accepted but blank/not scored.
- `tejasgluth/rogii-hongweiluan-v6-inference-fork`: private fork cannot access the original source parquet artifacts.
- Near-duplicate Lightning/Pixiux/Hamatz outputs unless all stronger systems fail.
