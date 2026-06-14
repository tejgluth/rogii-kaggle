# %% [code]
# v3
import warnings; warnings.filterwarnings('ignore')
import subprocess
subprocess.run(['pip', 'install', '-q', 'optuna'], check=False)

import time, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

t0 = time.time()

def find_comp_dir():
    candidates = [
        Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),
        Path('/kaggle/input/rogii-wellbore-geology-prediction'),
    ]
    for path in candidates:
        if (path / 'sample_submission.csv').exists():
            return path
    for filename in glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True):
        path = Path(filename).parent
        if (path / 'train').exists() and (path / 'test').exists():
            return path
    raise FileNotFoundError('Could not locate competition data directory.')


def find_train_dir():
    candidates = [
        Path('/kaggle/input/notebooks/hongweiluan/rogii-wellbore-v6-gpu-script'),
        Path('/kaggle/input/rogii-wellbore-v6-gpu-script'),
    ]
    for path in candidates:
        if (path / 'train_meta.parquet').exists() and (path / 'test_meta.parquet').exists():
            return path
    for filename in glob.glob('/kaggle/input/**/train_meta.parquet', recursive=True):
        path = Path(filename).parent
        if (path / 'test_meta.parquet').exists():
            return path
    raise FileNotFoundError('Could not locate Hongwei v6 train_meta/test_meta parquet files.')


COMP_DIR = find_comp_dir()
TRAIN_DIR = find_train_dir()
print(f'COMP_DIR={COMP_DIR}', flush=True)
print(f'TRAIN_DIR={TRAIN_DIR}', flush=True)

print('Loading predictions from training kernel...', flush=True)
train_meta = pd.read_parquet(TRAIN_DIR / 'train_meta.parquet')
test_meta  = pd.read_parquet(TRAIN_DIR / 'test_meta.parquet')
print(f'train_meta: {train_meta.shape}, test_meta: {test_meta.shape}', flush=True)

# Unpack
y_np     = train_meta['target'].values.astype(np.float32)
ens_oof  = train_meta['ens_oof'].values.astype(np.float32)
pf_oof   = train_meta['pf_ancc_d'].values.astype(np.float32)
form_oof = train_meta['form_d'].values.astype(np.float32)
last_tvt = train_meta['last_known_tvt'].values
md_since = train_meta['md_since'].values
ytrue_abs = y_np + last_tvt

ens_test  = test_meta['ens_test'].values.astype(np.float32)
pf_test   = test_meta['pf_ancc_d'].values.astype(np.float32)
form_test = test_meta['form_d'].values.astype(np.float32)
last_tvt_t = test_meta['last_known_tvt'].values
md_since_t = test_meta['md_since'].values

print(f'Loaded in {time.time()-t0:.1f}s', flush=True)
raw_oof_rmse = float(np.sqrt(np.mean((ens_oof - y_np)**2)))
print(f'Raw ensemble OOF RMSE: {raw_oof_rmse:.4f}', flush=True)

# ── Optuna post-processing (500 trials) ──────────────────────────────────────
def objective(trial):
    alpha  = trial.suggest_float('alpha', 0.3, 1.5)
    tau    = trial.suggest_float('tau', 50, 1000)
    w_pf   = trial.suggest_float('w_pf', 0.0, 0.4)
    w_form = trial.suggest_float('w_form', 0.0, 0.3)
    w_mdl  = 1.0 - w_pf - w_form
    d_blend = w_mdl * ens_oof + w_pf * pf_oof + w_form * form_oof
    tau_fac = 1.0 - np.exp(-np.maximum(md_since, 0.) / tau)
    d_pp = (1. - tau_fac) * pf_oof + tau_fac * alpha * d_blend
    return float(np.sqrt(np.mean((last_tvt + d_pp - ytrue_abs)**2)))

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50))
study.optimize(objective, n_trials=500, n_jobs=-1)
bp = study.best_params
print(f'Best PP params: {bp}  RMSE={study.best_value:.4f}', flush=True)

# ── Test predictions with best PP ─────────────────────────────────────────────
w_mdl_t = 1. - bp['w_pf'] - bp['w_form']
d_blend_t = w_mdl_t * ens_test + bp['w_pf'] * pf_test + bp['w_form'] * form_test
tau_fac_t = 1.0 - np.exp(-np.maximum(md_since_t, 0.) / bp['tau'])
test_pp = (1. - tau_fac_t) * pf_test + tau_fac_t * bp['alpha'] * d_blend_t
test_meta['pred'] = last_tvt_t + test_pp

for _, grp in test_meta.groupby('well', sort=False):
    v = grp['pred'].values
    wl = min(17, len(v))
    if wl % 2 == 0: wl -= 1
    if wl >= 5: v = savgol_filter(v, wl, 3)
    test_meta.loc[grp.index, 'pred'] = v

print(f'TVT range: [{test_meta["pred"].min():.2f}, {test_meta["pred"].max():.2f}]', flush=True)

# ── Save submission ──────────────────────────────────────────────────────────
sample = pd.read_csv(COMP_DIR / 'sample_submission.csv')
id2pred = dict(zip(test_meta['id'], test_meta['pred']))
sample['tvt'] = sample['id'].map(id2pred).fillna(test_meta['pred'].median())
sample[['id', 'tvt']].to_csv('/kaggle/working/submission.csv', index=False)
print(f'Rows: {len(sample)}', flush=True)
print(sample[["id","tvt"]].head(3).to_string(index=False))
print(f'Done: submission.csv written', flush=True)
print(f'Total time: {time.time()-t0:.1f}s', flush=True)
print(f'OOF RMSE (with PP): {study.best_value:.4f}', flush=True)
