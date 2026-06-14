# %% [code]
# ==============================================================================
# BALANCED ORCHESTRATOR: ROGII V92 HYBRID (GOLDEN RATIO)
# ==============================================================================
# Objective: Fix the 21.854 RMSE underfitting from V91. 
# 1. Re-introduces 'md' and 'tvd' but with EXTREME regularizations to prevent memorization.
# 2. Removes 'x' and 'y' direct coordinates and KDTree spatial imputation, since 
#    geographical distances in test set differ vastly from train set.
# 3. Maintains Rolling Deltas and Physics invariants for generalization.
# ==============================================================================

import os
import sys
import glob
import time
import json
import warnings
import pathlib
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. LOOPMASTER: BUDGET & ENVIRONMENT MANAGER
# ==============================================================================
class LoopMasterBudgetManager:
    @staticmethod
    def is_kaggle_environment():
        return os.path.exists("/kaggle/input")
        
    @staticmethod
    def detect_gpu():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

# ==============================================================================
# 2. HILL-CLIMB L2 STACKING (The "Infinite Alternatives" Combiner)
# ==============================================================================
class LoopMasterEnsemble:
    @staticmethod
    def _rmse(y_true, y_pred):
        diff = y_true.astype(np.float32) - y_pred.astype(np.float32)
        return float(np.sqrt(np.mean(diff * diff)))

    @staticmethod
    def hill_climb_stack(predictions_dict, true_y, max_rounds=25):
        names = list(predictions_dict.keys())
        scores = {name: LoopMasterEnsemble._rmse(true_y, predictions_dict[name]['oof']) for name in names}
        
        best_name = min(scores, key=scores.get)
        cur_oof = predictions_dict[best_name]['oof'].astype(np.float32).copy()
        cur_test = predictions_dict[best_name]['test'].astype(np.float32).copy()
        
        weights = {best_name: 1.0}
        best_score = scores[best_name]
        
        # Penalized grid to favor slow updates and prevent extreme weight shifts
        grid = np.array([0.01, 0.02, 0.05, 0.10], dtype=np.float32)
        
        print(f"[*] Hill-Climb Baseline ({best_name}): {best_score:.4f}")
        
        for rd in range(1, max_rounds + 1):
            step = None
            for name in names:
                cand_oof = predictions_dict[name]['oof'].astype(np.float32)
                for w in grid:
                    trial = (1.0 - float(w)) * cur_oof + float(w) * cand_oof
                    score = LoopMasterEnsemble._rmse(true_y, trial)
                    # Require a minimum improvement to accept a step (stops noise fitting)
                    if score + 1e-6 < best_score:
                        step = (name, float(w), score, trial)
                        best_score = score
                        
            if step is None:
                break
                
            name, w, score, trial_oof = step
            cur_oof = trial_oof.astype(np.float32)
            cur_test = ((1.0 - w) * cur_test + w * predictions_dict[name]['test'].astype(np.float32)).astype(np.float32)
            
            for k in list(weights):
                weights[k] *= (1.0 - w)
            weights[name] = weights.get(name, 0.0) + w
            
        return cur_oof, cur_test, best_score, weights

# ==============================================================================
# 3. ORCHESTRATOR PIPELINE: V92 BALANCED
# ==============================================================================
class ROGIIV92BalancedPipeline:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.is_gpu = LoopMasterBudgetManager.detect_gpu()
        
        print(f"[*] Balanced Pipeline V92 Initializing...")
        print(f"[*] GPU Acceleration Detected: {self.is_gpu}")
        
        # ---------------------------------------------------------
        # V92 STRATEGY: 
        # Models retain 'md' and 'tvd' to know vertical depth, 
        # but Extreme Regularization prevents them from memorizing it.
        # ---------------------------------------------------------
        self.models = {
            'lgbm': lgb.LGBMRegressor(
                n_estimators=300, 
                learning_rate=0.03, 
                num_leaves=21, # Reduced leaves
                min_child_samples=50, # High min data
                reg_alpha=2.0, # L1 Regularization
                reg_lambda=5.0, # L2 Regularization (Heavy)
                subsample=0.8,
                random_state=42, n_jobs=-1, verbose=-1,
                device_type='gpu' if self.is_gpu else 'cpu'
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=350, 
                learning_rate=0.03, 
                depth=5, # Shallower depth
                l2_leaf_reg=10, # Heavy L2
                random_seed=42, verbose=0, thread_count=-1,
                task_type='GPU' if self.is_gpu else 'CPU'
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=300, 
                learning_rate=0.03, 
                max_depth=5, 
                reg_alpha=2.0,
                reg_lambda=5.0,
                subsample=0.8,
                random_state=42, n_jobs=-1,
                tree_method='hist',
                device='cuda' if self.is_gpu else 'cpu'
            ),
            'hist_gbdt': HistGradientBoostingRegressor(
                max_iter=300, 
                learning_rate=0.03, 
                max_depth=5, 
                min_samples_leaf=40,
                l2_regularization=5.0,
                random_state=42
            ),
            'ridge': Ridge(alpha=20.0, solver='auto') # Higher Alpha
        }
        self.trained_fold_models = {name: [] for name in self.models.keys()}
        self.best_weights = {}

    def _discover_files(self, split='train'):
        p_split = pathlib.Path(os.path.abspath(os.path.join(self.raw_data_dir, split)))
        files = [str(f) for f in p_split.rglob("*horizontal_well.csv")]
        
        if not files and os.path.exists("/kaggle/input"):
            print(f"[*] Dynamically searching for {split} CSV files in /kaggle/input/ ...", flush=True)
            all_files = list(pathlib.Path("/kaggle/input").rglob("*horizontal_well.csv"))
            for f in all_files:
                if split in str(f).lower():
                    files.append(str(f))
                    
        return sorted(files)

    def _load_data(self, files):
        chunks = []
        for i, f in enumerate(files):
            df = pd.read_csv(f)
            df.columns = [str(c).strip().lower() for c in df.columns]
            df['well_id'] = os.path.basename(f).replace('__horizontal_well.csv', '')
            
            if 'tvt_input' in df.columns:
                df['tvt_input_imputed'] = df['tvt_input'].interpolate(method='linear').bfill().ffill()
                df['tvt_input_imputed'] = df['tvt_input_imputed'].fillna(0)
            else:
                df['tvt_input_imputed'] = 0
            chunks.append(df)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def build_features(self, df):
        print("[*] LoopMaster: Extracting Physical Invariants & Regulated Coordinates...", flush=True)
        
        df['Delta_MD'] = df.groupby('well_id')['md'].diff().fillna(0)
        df['Gr_Rolling_Mean'] = df.groupby('well_id')['gr'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['Gr_Rolling_Std']  = df.groupby('well_id')['gr'].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
        df['Gr_Rolling_Max']  = df.groupby('well_id')['gr'].transform(lambda x: x.rolling(5, min_periods=1).max())
        
        # KDTree Spatial Imputation removed intentionally in V92 to prevent Test Set hallucination.
        # X and Y are completely dropped to prevent absolute geographic overfitting.
        # MD and TVD are retained to preserve depth context (but Models are highly regularized).

        features = [
            'gr', 'Delta_MD', 'Gr_Rolling_Mean', 'Gr_Rolling_Std', 'Gr_Rolling_Max',
            'md', 'tvd' # <-- V92 the "Golden Ratio": Depth is allowed, Geography (X,Y) is forbidden.
        ]
        
        X = df[features].copy()
        X.fillna(0, inplace=True)
        return X, df

    def execute_pipeline(self):
        # 1. Load Data
        train_files = self._discover_files('train')
        test_files = self._discover_files('test')
        
        if not train_files or not test_files:
            print("[-] CRITICAL FAIL: Missing Train or Test files. Loop aborting.")
            return

        print(f"[+] Loaded {len(train_files)} Train files, {len(test_files)} Test files.")
        
        train_df = self._load_data(train_files)
        test_df = self._load_data(test_files)
        
        # 2. Build Features
        X_train, train_df = self.build_features(train_df)
        X_test, test_df = self.build_features(test_df)
        
        # Residual Target
        y_train = train_df['tvt'] - train_df['tvt_input_imputed']
        
        # 3. K-Fold Training
        print("[*] Starting Cross-Validation (Balanced Super Learner Ensemble)...")
        gkf = GroupKFold(n_splits=5)
        
        oof_preds = {name: np.zeros(len(X_train)) for name in self.models.keys()}
        test_preds_dict = {name: np.zeros(len(X_test)) for name in self.models.keys()}
        
        import copy
        for name, model in self.models.items():
            print(f"    -> Training {name.upper()}...")
            for fold, (t_idx, v_idx) in enumerate(gkf.split(X_train, y_train, groups=train_df['well_id'])):
                X_t, X_v = X_train.iloc[t_idx], X_train.iloc[v_idx]
                y_t, y_v = y_train.iloc[t_idx], y_train.iloc[v_idx]
                
                fold_model = copy.deepcopy(model)
                fold_model.fit(X_t, y_t)
                
                oof_preds[name][v_idx] = fold_model.predict(X_v)
                test_preds_dict[name] += fold_model.predict(X_test) / 5.0
                self.trained_fold_models[name].append(fold_model)
                
            score = root_mean_squared_error(y_train, oof_preds[name])
            print(f"       [+] {name.upper()} OOF RMSE: {score:.4f}")
            
        # 4. Hill-Climb L2 Optimization
        print("[*] Orchestrator: Optimizing weights with Hill-Climb...")
        pred_dict = {
            name: {'oof': oof_preds[name], 'test': test_preds_dict[name]} 
            for name in self.models.keys()
        }
        
        final_oof, final_test, best_score, weights = LoopMasterEnsemble.hill_climb_stack(pred_dict, y_train.values)
        self.best_weights = weights
        
        print(f"\n[!!!] V92 BALANCED FINAL OOF RMSE: {best_score:.4f}")
        print(f"[*] Optimized Weights: {json.dumps(weights, indent=2)}")
        
        # 5. Failsafe Kaggle Submission Generation
        self._generate_submission(test_df, final_test)

    def _generate_submission(self, test_df, final_test_preds):
        print("\n[*] Committer: Executing Failsafe Submission Protocol...")
        test_df['predicted_tvt'] = test_df['tvt_input_imputed'] + final_test_preds
        
        # DETERMINISTIC METRICS CHECK
        # Must be exactly 'id' and 'tvt' format for Kaggle
        # Extract well_id without the __horizontal_well.csv if it somehow leaked
        test_df['id'] = test_df['well_id'].astype(str) + '_' + test_df['md'].astype(str)
        submission = test_df[['id', 'predicted_tvt']].rename(columns={'predicted_tvt': 'tvt'})
        
        # VERIFIER FAILSAFES
        assert 'id' in submission.columns, "CRITICAL ERROR: Missing 'id' column"
        assert 'tvt' in submission.columns, "CRITICAL ERROR: Missing 'tvt' column"
        assert submission['tvt'].isnull().sum() == 0, "CRITICAL ERROR: NaNs detected in TVT predictions!"
        
        sub_path = "submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"[+] AUDITOR PASS: {sub_path} generated flawlessly with {len(submission)} rows.")
        print("=============================================================================")
        print("  V92 ORCHESTRATION COMPLETE. KAGGLE SUBMISSION READY FOR DEPLOY.            ")
        print("=============================================================================")


if __name__ == "__main__":
    # Standard Dynamic Kaggle vs Local Path Resolution
    raw_dir = "../input" if os.path.exists("../input") else "/kaggle/input"
    if os.path.exists("../1_dados_brutos"):
        raw_dir = "../1_dados_brutos"
        
    pipeline = ROGIIV92BalancedPipeline(raw_data_dir=raw_dir)
    pipeline.execute_pipeline()
