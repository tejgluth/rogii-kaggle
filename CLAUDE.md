# ROGII Wellbore Geology Prediction — Kaggle Competition System

You are the **orchestrator agent** for a Kaggle ML competition. Your job is to
plan experiments, spawn Codex worker agents to execute them in parallel, track
results, and iteratively build toward the best possible solution.

Read this entire file before doing anything else.

---

## Competition Context

- **URL**: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- **Task**: Regression — predict **True Vertical Thickness (TVT)** for every
  1-ft depth sample along horizontal oil/gas wellbores
- **Metric**: RMSE (lower is better)
- **Data**: `data/raw/train/` and `data/raw/test/`
  - Well trajectory files (measured depth, inclination, azimuth, TVD, XYZ)
  - Gamma Ray (GR) log files per well
  - Typewell reference logs (vertical reference well with known geology)
  - Target column: `tvt`

## Domain Knowledge (Read Before Proposing Features)

**Geosteering**: When drilling a horizontal well, the bit travels 2–5 km
laterally through rock. A geologist must keep the bit inside the productive
reservoir layer. TVT represents where the well sits within the geological layer
stack.

**Gamma Ray (GR) log**: Measures natural radioactivity. High GR = shale,
low GR = sandstone/reservoir. The GR log is the primary signal for
correlating with the Typewell to estimate TVT.

**Typewell**: A nearby vertical reference well. The Typewell GR log is the
"ground truth" geological template. To estimate TVT, you align the lateral
well's GR with the Typewell GR — the shift gives the TVT.

**Physical constraints on TVT**:
- TVT changes smoothly along a well (no sudden jumps)
- TVT is bounded by the known layer thicknesses in the Typewell
- Structural dip (tilt of rock layers) causes a systematic trend in TVT

**Key insight for features**: The most powerful signal is the cross-correlation
or DTW (Dynamic Time Warping) distance between the lateral GR window and the
Typewell GR at different lag offsets. The offset with the best match indicates
the TVT.

---

## Your Workflow — Follow This Exactly

### Phase 1: EDA (do this first, once)
1. Read `prompts/01_eda.md`
2. Spawn ONE Codex worker with the EDA task
3. Wait for completion, read the EDA report from `experiments/results/eda_report.md`
4. Summarize what you learned: data shapes, missing values, GR statistics,
   number of wells, depth ranges

### Phase 2: Baselines (get a CV score on the board)
1. Read `prompts/02_baseline.md`
2. Spawn 3 parallel Codex workers: LightGBM, XGBoost, simple NN
3. Each worker saves OOF and test predictions to `experiments/oof/` and
   `experiments/test_preds/` using the naming convention below
4. Log all CV scores to `experiments/experiment_log.csv`
5. Report the best baseline CV score before proceeding

### Phase 3: Feature Engineering (iterative, this is where you win)
1. Read `prompts/03_feature_engineering.md`
2. Generate 5–8 experiment ideas based on EDA findings and domain knowledge
3. Spawn Codex workers in parallel batches of 3–4
4. For each completed experiment:
   - If CV improved: keep it, add to `src/features.py`, plan next iteration
   - If CV did not improve: note why, try a different direction
5. Keep iterating until you have run at least 30 feature engineering experiments
   or CV stops improving across 10 consecutive experiments
6. Before each new batch, re-read `experiments/experiment_log.csv` to avoid
   repeating ideas

### Phase 4: Stacking and Combination
1. Read `prompts/04_stacking.md`
2. Collect all OOF files from top-performing experiments
3. Spawn a Codex worker to build hill-climbing ensemble
4. Spawn a Codex worker to build stacking meta-learner (Ridge, NN, LightGBM)
5. Choose the best combination for final submission

---

## How to Spawn a Codex Worker

Use `agents/codex_worker.py`. Always provide a JSON spec file first.

```python
# Step 1: Write spec
import json
spec = {
    "experiment_id": "exp_042",
    "phase": "feature_engineering",
    "description": "LightGBM with DTW features at lag offsets -20 to +20",
    "model": "lightgbm",
    "features_to_add": ["dtw_distance_lag_-20_to_20", "gr_corr_typewell"],
    "base_experiment": "exp_003",   # inherit features from this experiment
    "save_oof_path": "experiments/oof/oof_lgbm_exp042.npy",
    "save_test_path": "experiments/test_preds/test_lgbm_exp042.npy",
    "save_result_path": "experiments/results/exp042.json"
}
with open("experiments/specs/exp042.json", "w") as f:
    json.dump(spec, f, indent=2)

# Step 2: Spawn worker
from agents.codex_worker import spawn_codex_worker
result = spawn_codex_worker("experiments/specs/exp042.json")
```

To spawn multiple workers in parallel:
```python
from agents.codex_worker import spawn_parallel_workers
specs = ["experiments/specs/exp042.json",
         "experiments/specs/exp043.json",
         "experiments/specs/exp044.json"]
results = spawn_parallel_workers(specs, max_workers=3)
```

---

## File Naming Convention (Strict — Never Deviate)

| File type | Pattern | Example |
|-----------|---------|---------|
| Experiment spec | `experiments/specs/exp{NNN}.json` | `exp042.json` |
| OOF predictions | `experiments/oof/oof_{model}_exp{NNN}.npy` | `oof_lgbm_exp042.npy` |
| Test predictions | `experiments/test_preds/test_{model}_exp{NNN}.npy` | `test_lgbm_exp042.npy` |
| Result metadata | `experiments/results/exp{NNN}.json` | `exp042.json` |
| Feature code | `src/features.py` | add functions, never overwrite |
| Model code | `src/models.py` | add classes, never overwrite |

Result JSON format (Codex workers must write this):
```json
{
  "experiment_id": "exp_042",
  "model": "lightgbm",
  "cv_rmse": 10.84,
  "cv_rmse_std": 0.32,
  "features_used": ["gr_mean_30", "tvd", "dtw_lag_5"],
  "n_features": 24,
  "training_time_seconds": 142,
  "notes": "DTW at lag +5 was most important feature"
}
```

---

## Experiment Log

Always update `experiments/experiment_log.csv` after each experiment.
Use `agents/experiment_tracker.py`:

```python
from agents.experiment_tracker import log_experiment
log_experiment("exp042", cv_rmse=10.84, model="lgbm", phase="fe", notes="...")
```

---

## GPU Usage (Critical — Always Use GPU)

The DGX Spark has NVIDIA GPUs with CUDA. Always use:
- `import cudf` instead of `pandas` for data loading
- `import cuml` for GPU scikit-learn equivalents
- XGBoost: `params["device"] = "cuda"`
- LightGBM: `params["device"] = "gpu"`
- PyTorch: `device = torch.device("cuda")`

If cuDF is not available (Mac M4 fallback): use pandas but add a warning.

---

## Model Selection Priorities

1. **LightGBM** (GPU): fastest iteration, strong baseline
2. **XGBoost** (GPU): good diversity from LightGBM
3. **CatBoost** (GPU): robust to outliers
4. **LSTM / Temporal CNN** (PyTorch CUDA): treats depth as time series
5. **TabNet**: good for tabular with attention

Always run with 5-fold GroupKFold splitting by well_id. Never mix rows from
the same well across train/test folds.

---

## When Stuck

If an experiment produces an error:
1. Read the error output from `experiments/results/exp{NNN}_error.txt`
2. Fix the issue directly in `src/` if it is a data loading problem
3. Re-spawn the worker with the same spec but increment version: `exp042_v2`

If CV is not improving after 5 consecutive experiments:
1. Re-read EDA report
2. Ask: are there physical constraints being violated?
3. Try a completely different modeling approach (e.g., switch from tree to NN)
4. Read Kaggle discussion forum notes in `competition/leaderboard.md`

---

## Final Submission Checklist

Before running `scripts/create_submission.py`:
- [ ] At least 30 feature engineering experiments logged
- [ ] At least 3 diverse model types (tree, NN, linear)
- [ ] Stacking meta-learner tested
- [ ] CV and LB scores both checked for overfit (CV should be close to LB)
- [ ] Submission file has correct column names and no NaN values

---

## Important Notes

- **Never modify raw data** in `data/raw/`
- **Commit to git** after every 5 experiments: `git add -A && git commit -m "exp{N}-{N+4} results"`
- The `experiments/` folder is the source of truth. Protect it.

## Rate Limit Management (Claude Pro + ChatGPT Plus)

Both plans use **5-hour rolling windows**. Claude Pro has smaller limits than
Max 5x. Follow these rules to avoid hitting walls mid-session:

1. **Run experiments in batches of 2, not 3-4.** Two parallel Codex workers
   at a time is safer on Plus limits.
2. **Between waves, pause.** After each batch of 2 experiments completes,
   update the experiment log, write your next plan to `current_plan.md`,
   then start the next batch. This gives natural breathing room.
3. **Use Sonnet for code, Opus for strategy.** Reserve Opus calls for high-value
   decisions (which features to try next, stacking decisions). Use Sonnet for
   reading files and writing experiment specs.
4. **If you hit a limit**, write your full current state to
   `experiments/current_plan.md` immediately and stop. The next session
   resumes from that file.
5. **Codex uses ChatGPT auth** (not an API key). Its 5-hour window is
   separate from Claude's — they don't share limits. This is the key advantage
   of using two providers.
