# Agent System Specification

This file defines how orchestrator and worker agents coordinate.

## Agent Roles

### Orchestrator (Claude Code)
- Maintains full experiment history and context
- Decides which experiments to run next
- Spawns Codex workers with precise, narrow task specs
- Verifies worker outputs before moving on
- Never writes training code directly — delegates to workers

### Worker (Codex CLI)
- Receives a single JSON spec file path as its only input
- Reads the spec and writes + executes training code
- Saves OOF, test predictions, and result JSON to exact paths in spec
- Writes a brief notes field in the result JSON explaining what it found
- Never modifies the spec file or any other experiment's files

## Task Dependency Graph

```
Phase 1: EDA
  └── eda_task (no dependencies)

Phase 2: Baselines (all parallel, no dependencies)
  ├── baseline_lgbm
  ├── baseline_xgb
  └── baseline_nn

Phase 3: Feature Engineering (waves, each wave depends on previous wave results)
  Wave 1: [fe_dtw, fe_rolling_stats, fe_trajectory] (parallel)
  Wave 2: depends on Wave 1 results — orchestrator picks best and extends
  Wave 3: depends on Wave 2 results
  ...

Phase 4: Stacking (depends on all Phase 3 results)
  ├── hill_climbing (depends on all OOF files)
  └── meta_learner (depends on all OOF files)
```

## Communication Protocol

All communication is file-based on shared disk. No network calls between agents.

```
Orchestrator writes:  experiments/specs/exp{NNN}.json
Worker reads:         experiments/specs/exp{NNN}.json
Worker writes:        experiments/oof/oof_{model}_exp{NNN}.npy
                      experiments/test_preds/test_{model}_exp{NNN}.npy
                      experiments/results/exp{NNN}.json
Orchestrator reads:   experiments/results/exp{NNN}.json
Orchestrator updates: experiments/experiment_log.csv
```

## Worker Prompt Template

When Claude Code spawns a Codex worker, it must include ALL of the following
in the Codex prompt (codex_worker.py handles this automatically):

1. Full path to the spec file
2. Path to `src/` for shared utilities
3. Path to `data/processed/` for feature-engineered data
4. The exact output file paths it must create
5. The exact JSON schema the result file must follow
6. A reminder to use GPU (cuDF, cuML, CUDA)
7. A reminder to use GroupKFold by well_id

## Parallelism Rules

- Maximum **2 Codex workers in parallel** (ChatGPT Plus limit safety)
- Each worker gets an isolated task — no two workers write to the same file
- If a worker fails, orchestrator retries once with the same spec
- Workers should complete within 15 minutes; if longer, something is wrong
- After each 2-worker batch, pause for orchestrator to log results before
  starting the next batch — this manages the 5-hour rolling window

## State Recovery

If Claude Code session ends unexpectedly:
1. Read `experiments/experiment_log.csv` to see what has been completed
2. Read `experiments/current_plan.md` if it exists for the in-progress plan
3. Check `experiments/specs/` for any specs without corresponding results
4. Re-spawn workers for any incomplete experiments
5. Continue from where you left off
