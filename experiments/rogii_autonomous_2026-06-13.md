# ROGII Autonomous Run Notes - 2026-06-13

Goal: lower public RMSE for `rogii-wellbore-geology-prediction` by scoring the strongest public-kernel families, then branching cheap variants around blend/hedge/postprocess knobs.

## External Playbook Used

NVIDIA/Chris Deotte article, 2026-04-23: use agent-driven EDA, keep every OOF/test artifact, explore many feature/model variants quickly, then combine models with hill climbing/stacking/ensembles. Applied here as: summarize completed runs, score distinct public kernels, keep sidecar submissions/reports, sweep blend and exact-match hedge weights.

## Submitted To Competition

| ref | candidate | status at review | note |
| --- | --- | --- | --- |
| 53648487 | `tejasgluth/rogii-dual-pipeline-self-verifying-fixed` v1 | pending | Lightning guarded overlap override; fixed notebook audit bug. |
| 53649506 | `tejasgluth/rogii-target-free-tvt-geosteering-fork` v2 | pending | Pilkwang target-free fork; fixed `koolbox` dependency; selected ridge-PF/LGBM 0.55 blend. |
| 53649507 | `tejasgluth/rogii-lightning-preoverride-w55` v1 | pending | Plain SP45/Fleongg 0.55/0.45 from Lightning before guarded override. |
| 53649508 | `tejasgluth/rogii-ridge-artifact-projection-d4-fork` v1 | pending | Yaroslav D4 ridge artifact projection fork. |
| 53648444 | static Lightning override | complete, blank score | Static-output shortcut is accepted but not scored; do not use for leaderboard tests. |

Submit attempts blocked after the accepted refs above:

- `tejasgluth/rogii-target-free-tvt-w60-cpu-fork` v1: `403 Forbidden`.
- `tejasgluth/rogii-lightning-preoverride-w60` v1: `400 Bad Request`.
- `tejasgluth/rogii-lightning-sp45-component` v1: `400 Bad Request`.
- `tejasgluth/rogii-lightning-fleongg-component` v1: `400 Bad Request`.

Working assumption: competition/code-submission or pending-submission limit, not output contract failure. All four blocked artifacts validated locally.

## Completed Output Audit

All reviewed `submission.csv` files match sample row count/order and finite `tvt`.

| output | sha prefix | mean | std | note |
| --- | --- | ---: | ---: | --- |
| Pilkwang fixed v2 | `7ef55f6028a61db3` | 11904.4297 | 278.3148 | selected 0.55 ridge-PF/LGBM blend; guards disabled. |
| Pilkwang w60 CPU | `a36aa494963a8547` | 11904.2555 | 278.3725 | validated, not submitted due API limit. |
| Yaroslav D4 | `d77c14568cdc9302` | 11904.3903 | 278.4455 | submitted. |
| Lightning w55 | `088fab7149bb38eb` | 11904.6255 | 278.4894 | submitted. |
| Lightning w60 | `903cf0b35f1b772c` | 11904.0958 | 278.1788 | validated, not submitted due API limit. |
| Lightning SP45 only | `6014ee885c055e5d` | 11904.2765 | 278.8201 | validated, not submitted due API limit. |
| Lightning Fleongg only | `6e7a152d8f6b212f` | 11904.7666 | 278.0997 | validated, not submitted due API limit. |
| Lightning guarded override | `2b86386f19279e79` | 11903.6301 | 278.0248 | submitted; distinct from plain blends. |

Pairwise RMSE deltas among the validated outputs range from `0.25854` to `3.74858`, so the runs are not trivial duplicates.

## Newly Queued Kernel Runs

CPU queue accepted five public-kernel forks:

- `tejasgluth/rogii-fle3n-v5f-probe-fork` v1
- `tejasgluth/rogii-pixiux-dual-pipeline-blend-fork` v1
- `tejasgluth/rogii-iaztec-sp45-fleongg-eda-fork` v1
- `tejasgluth/rogii-rikuter-pilkwang-ridge-artifact-fork` v1
- `tejasgluth/rogii-hamatz-dual-pipeline-reproduced-fork` v1

GPU queue accepted two v5f hedge variants:

- `tejasgluth/rogii-fle3n-v5f-xfer-0p45` v1
- `tejasgluth/rogii-fle3n-v5f-xfer-0p55` v1

CPU batch limit prevented these from starting yet:

- `tejasgluth/rogii-fle3n-v5f-xfer-0p25`
- `tejasgluth/rogii-fle3n-v5f-xfer-0p40`
- `tejasgluth/rogii-fle3n-v5f-xfer-0p60`
- `tejasgluth/rogii-fle3n-v5f-xfer-0p75`

## Next Actions

1. Poll pending competition submissions and record public scores.
2. Pull outputs from the seven queued kernels as they complete.
3. Submit the best completed notebook outputs when code-submission slots free up.
4. If v5f outputs validate and submission slots remain blocked, keep their outputs local and submit later in order: v5f original, xfer 0.45, xfer 0.55, Pixiux, Rikuter/Pilkwang.
5. Once CPU slots free, queue remaining v5f hedge weights and rounding/offset variants around the best public-scored family.

## Prepared But Not Yet Pushed

EDA found train `TVT` values are effectively on a 0.01-ft grid. Prepared cheap v5f postprocess kernels to push when batch slots free:

- `tejasgluth/rogii-v5f-round-0p01`
- `tejasgluth/rogii-v5f-xfer045-round-0p01`
- `tejasgluth/rogii-v5f-round-0p05`
- `tejasgluth/rogii-v5f-offset-m0p10-round-0p01`
- `tejasgluth/rogii-v5f-offset-p0p10-round-0p01`
- `tejasgluth/rogii-v5f-xfer000-round-0p01`

Also pulled `pilkwang/rogii-eda-target-free-alignment-for-tvt`, which exposes heavier ridge/PF profile tests (`ridge_artifact_push`, `ridge_artifact_pf_boost`, `ridge_artifact_model_blend`). It is lower priority until the current faster public forks finish because the notebook metadata shows a long historical runtime.

Fresh notebooks inspected after the first queue:

- `pilkwang/eda-agent-security-trajectory-search`: different competition, ignored.
- `shuaixu123/rogii-exp-098-public-top3-gpu0-retry`: scoreable GPU notebook; staged as `tejasgluth/rogii-shuaixu-exp098-top3-gpu0-retry-fork`.
- `opengs7/v25b-rogii-cb-ensemble`: scoreable GPU CatBoost/GBM ensemble; staged as `tejasgluth/rogii-opengs7-v25b-cb-ensemble-fork`.
- `mvaishak/tvt-10run-submission`: scoreable GPU SP/Fleongg-style reproduction; staged as `tejasgluth/rogii-mvaishak-tvt-10run-fork`.
- `omadon/rogii-honest-cv-formation-surface-geometry` and `yujiyyy/rogii-cv-vs-lb-how-well-does-groupkfold-transfer`: methodology/diagnostic references; not queued.

## 2026-06-14 Continuation

### Public Submission Scores

Latest competition submission poll:

- `53649507` `tejasgluth/rogii-lightning-preoverride-w55` completed at public `7.580`.
- `53648487` `tejasgluth/rogii-dual-pipeline-self-verifying-fixed` completed at public `7.591`.
- `53649506` `tejasgluth/rogii-target-free-tvt-geosteering-fork` completed at public `7.633`.
- `53649508` Yaroslav D4 remained pending at the last poll.
- Static visible-output submission completed with blank score; do not use static CSV kernels for final competition slots.

### Research Notes

NVIDIA/Chris Deotte agentic workflow article takeaways applied here:

- Keep every OOF/test prediction artifact and use a repeatable audit script rather than hand-picked outputs.
- Search public code and discussions for feature ideas.
- Prefer diverse systems, then combine or submit them in priority order.
- Trust honest validation more than small public-LB perturbations.

Additional Kaggle discussion findings:

- Private test rescore excluded one outlier private well; public leaderboard is unchanged.
- Public split is fixed; repeated-score variation is from stochastic feature engineering, multiprocessing, GPU nondeterminism, and Numba RNG.
- PF/beam methods should improve by generating physically plausible trajectories and by ranking candidates by TVT RMSE correlation, not GR fit.
- Formation surfaces/geological planes, lateral self-correlation, neighboring-well correlation, and trajectory/geometry constraints are repeatedly cited as the path to sub-7 results.
- Forum reports suggest honest GroupKFold by well has tracked real improvements better than public-LB probing, but public LB can be noisy by 0.5+ ft for stochastic notebooks.

### New Output Audit

Added `scripts/analyze_submission_candidates.py`, producing:

- `experiments/results/submission_candidate_audit_2026-06-14.csv`
- `experiments/results/submission_candidate_audit_2026-06-14.md`

The audit validates sample order, finite predictions, stable hashes, near-duplicates, and visible-row diversity. It explicitly treats visible distances as diversity only, not score estimates.

Most useful distinct output families from the latest audit:

- `public_quanzhong_pf_fleongg`: very distinct from scored Lightning/Pilkwang/V5F families; nearest distinct RMSE `7.43`, but public score unknown and higher risk.
- `public_v8_gold`: distinct from Lightning/Pilkwang/V5F; nearest distinct RMSE `3.24`; GPU runtime around 1.5h on visible public output.
- `public_hongweiluan_v6_e2e` / `public_hongweiluan_v6_inference`: distinct but private fork depends on original author's kernel-source artifacts; only promote if our private fork completes.
- `public_zeyufeng_final`: fast PF-style output with distinct RMSE `2.27`; useful backup.
- V5F xfer/round sweep outputs are valid but close to the already scored public V5F/Lightning band; keep only as backups.

### New Queued Runs

Running at last update:

- GPU: `tejasgluth/rogii-shuaixu-exp098-top3-gpu0-retry-fork` v1.
- GPU: `tejasgluth/rogii-opengs7-v25b-cb-ensemble-fork` v1.
- CPU: `tejasgluth/rogii-quanzhong-pf-fleongg-blend-fork` v1.
- CPU: `tejasgluth/rogii-hongweiluan-v6-e2e-fork` v1, but risk: likely same kernel-source artifact access issue as the inference fork.
- CPU: `tejasgluth/rogii-v5f-xfer045-round-0p01` v1.
- CPU: `tejasgluth/rogii-v5f-round-0p05` v1.

Completed and pulled:

- `tejasgluth/rogii-zeyufeng-final-fork` v1; reproduced public output exactly, sha `bd68447223d20c61`.
- `tejasgluth/rogii-fle3n-v5f-xfer-0p25` v1.
- `tejasgluth/rogii-fle3n-v5f-xfer-0p4` v1.
- `tejasgluth/rogii-fle3n-v5f-xfer-0p6` v1.
- `tejasgluth/rogii-fle3n-v5f-xfer-0p75` v1.
- `tejasgluth/rogii-v5f-round-0p01` v1.

Failed:

- `tejasgluth/rogii-hongweiluan-v6-inference-fork` v1/v2. Public output completes, but our private fork cannot access `train_meta.parquet` / `test_meta.parquet` from the original author's kernel source, even with dynamic path search.

Still staged for GPU slots:

- `tejasgluth/rogii-z149-v8-gold-fork`
- `tejasgluth/rogii-mvaishak-tvt-10run-fork`

### Tomorrow Submission Priority

Use only completed private forks with valid `submission.csv`.

1. Submit `tejasgluth/rogii-shuaixu-exp098-top3-gpu0-retry-fork` v1 if it completes. Rationale: best public-code provenance ("LB Top 3"), feature-rich formation/NCC/GBM ensemble, and structurally stronger than weight tweaks.
2. Submit `tejasgluth/rogii-opengs7-v25b-cb-ensemble-fork` v1 if it completes. Rationale: separate GPU CatBoost/GBM ensemble family with artifact features; more useful than another V5F perturbation.
3. Submit `tejasgluth/rogii-quanzhong-pf-fleongg-blend-fork` v1 if it completes. Rationale: most distinct scoreable CPU trajectory-generator output; high risk but the only queued candidate with enough structural difference to plausibly break away from the mid-7 public band.

Backups, in order:

1. `tejasgluth/rogii-zeyufeng-final-fork` v1 if complete.
2. `tejasgluth/rogii-z149-v8-gold-fork` after it is pushed and completes when a GPU slot frees.
3. Best V5F-family completed output (`xfer0.55`, `xfer0.60`, `round0.01`, etc.) only if the structurally different systems fail.
