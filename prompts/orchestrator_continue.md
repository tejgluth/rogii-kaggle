Act as the orchestrator agent for this ROGII Kaggle project.

Workspace: /home/tejas/Desktop/rogii-kaggle
Date context: this prompt may be used manually to continue the current project state.
Scheduling context:
- Do not install or enable recurring orchestrator timers.
- Keep work inside the active session unless the user explicitly asks for a new scheduled run.
- The launcher requests gpt-5.5 with extra-high reasoning for the orchestrator. Preserve that setting if you edit the launcher.
- You may use lower-reasoning/lower-cost Codex agents or subprocesses for bounded side tasks when that conserves usage, but keep strategic experiment selection in the high-reasoning orchestrator.

Objective:
- Win by lowering the Kaggle metric, Mean Squared Error, as much as possible.
- The immediate competitive target is public/private MSE < 9, which is RMSE < 3.0.
- Continue experimenting until there is a trustworthy submission candidate below that target.

Current known state:
- Use .venv/bin/python for repo experiments.
- Cached feature matrix: data/processed/exp043_exp026_features.npz.
- Strong clean base learners include:
  - exp063 XGBoost RMSE about 2.6928, MSE about 7.25.
  - exp064 XGBoost RMSE about 2.7103.
  - exp067 XGBoost RMSE about 2.7342.
- exp050 stack reports RMSE about 2.6626, MSE about 7.09, but it selected stack/smoothing settings on the full OOF target.
- exp056 residual stack reports RMSE about 2.4064, and exp070 reports about 2.3948, but both depend on the exp050 target-selected stack and need honest validation before trusting.
- exp071_honest_audit was created to audit stack/residual generalization using outer GroupKFold.

Rules:
- First inspect current running processes and latest result files/logs. Do not duplicate a long experiment already running.
- Preserve user changes. Do not reset or revert the worktree.
- Prefer high-leverage experiments: honest validation, better XGBoost/LightGBM tuning around the depth=5 slow-learning regime, fold-specific diagnostics, physically plausible postprocessing, and submission creation from the best trustworthy model.
- Track every experiment in experiments/results and save OOF/test arrays when possible.
- Build/update a submission CSV whenever there is a new best trustworthy test prediction.
- Before ending, write a concise continuation checkpoint to experiments/orchestrator_checkpoint.md with current best scores, running jobs, and the next recommended experiments.
- Do not claim live Kaggle leaderboard verification unless the Kaggle API or website access succeeds in this session.

Start by:
1. Read experiments/results/exp071.json if it exists; otherwise finish or run scripts/run_exp071_honest_audit.py.
2. If exp071 confirms residual stacking generalizes below RMSE 3.0, build a submission from the best current residual/stack test predictions.
3. If exp071 rejects exp056/exp070, continue from the clean XGB/exp050 base and run the next most promising experiment.
