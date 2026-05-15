# ROGII Wellbore Geology Prediction — Kaggle Competition

Multi-agent ML system using Claude Code (orchestrator) + Codex CLI (workers)
to compete in the ROGII Kaggle competition.

## Quick Start (DGX Spark)

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd rogii-kaggle

# 2. Run setup (creates conda env, installs deps, installs Codex CLI)
bash setup.sh

# 3. Activate environment
conda activate rogii

# 4. Add API keys
cp .env.example .env
# Edit .env with your keys

# 5. Authenticate Codex CLI
codex auth      # login with your ChatGPT Plus account

# 6. Download competition data
bash scripts/download_data.sh

# 7. Start Claude Code — it reads CLAUDE.md and begins the workflow
claude
```

## System Architecture

```
You (Human)
    │ steering decisions
    ▼
Claude Code (Orchestrator)          ← Claude Max 5x or Pro + API credits
    │ reads: CLAUDE.md, AGENTS.md
    │ spawns via: agents/codex_worker.py
    ├──────────────────────────────────────┐
    ▼                                      ▼
Codex Worker 1                    Codex Worker 2         (parallel)
  (train LightGBM)                  (train XGBoost)
    │                                      │
    ▼                                      ▼
experiments/oof/            experiments/oof/
experiments/test_preds/     experiments/test_preds/
experiments/results/        experiments/results/
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
         Claude Code reads results
         logs to experiment_log.csv
         decides next experiments
```

## Workflow (4 Phases)

| Phase | Goal | Agent |
|-------|------|-------|
| 1. EDA | Understand data | Codex (single) |
| 2. Baselines | Get CV score on board | Codex (3 parallel) |
| 3. Feature Engineering | Iterate to improve | Codex (3-4 parallel waves) |
| 4. Stacking | Combine best models | Codex (single) |

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Primary instructions for Claude Code — read this first |
| `agents/AGENTS.md` | Agent coordination spec |
| `agents/codex_worker.py` | Spawns Codex CLI subprocesses |
| `agents/experiment_tracker.py` | Logs all experiments to CSV |
| `prompts/` | Prompt templates for each phase |
| `src/features.py` | All feature engineering functions |
| `src/evaluate.py` | CV utilities and result saving |
| `competition/domain_knowledge.md` | Geosteering domain context |
| `experiments/experiment_log.csv` | Master experiment log |

## Budget

| Service | Plan | Cost |
|---------|------|------|
| Claude Code | Claude Pro ($20) — includes Claude Code CLI | $20 |
| Codex CLI | ChatGPT Plus ($20) — Codex included, auth via `codex auth` | $20 |
| Gemini CLI | Free tier | $0 |
| **Total** | | **$40/month** |

> **Rate limits**: Claude Pro and ChatGPT Plus both use 5-hour rolling windows.
> The orchestrator runs experiments in batches of 2-3 and pauses between waves
> to stay within limits. See `CLAUDE.md` for session management guidance.

## Competition

- **URL**: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- **Metric**: RMSE (lower = better)
- **Task**: Predict True Vertical Thickness (TVT) for horizontal wellbores
