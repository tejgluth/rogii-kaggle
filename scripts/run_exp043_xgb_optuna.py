"""exp043: Optuna-tuned XGBoost on exp026 b_well/NCC features."""
import json
import sys
import time
from pathlib import Path

try:
    import cudf as cudf_pd  # noqa: F401
    import cuml  # noqa: F401
    GPU = True
except ImportError:
    GPU = False
    print("WARNING: cuDF not available, using CPU pandas")

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_exp026_bwell_ncc import build_well  # noqa: E402

TRAIN_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"
CACHE_PATH = ROOT / "data/processed/exp043_exp026_features.npz"
OOF_PATH = ROOT / "experiments/oof/oof_xgboost_exp043.npy"
TEST_PATH = ROOT / "experiments/test_preds/test_xgboost_exp043.npy"
RESULT_PATH = ROOT / "experiments/results/exp043.json"
STUDY_DB_PATH = ROOT / "experiments/results/exp043_optuna.db"


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _build_dataset():
    print("Building exp026 features...", flush=True)
    train_files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    train_parts = []
    for i, path in enumerate(train_files, start=1):
        tw_path = path.parent / (path.stem.replace("__horizontal_well", "__typewell") + ".csv")
        df = build_well(path, tw_path, is_train=True)
        if df is not None:
            train_parts.append(df)
        if i % 200 == 0:
            print(f"  train {i}/{len(train_files)}", flush=True)
    train = pd.concat(train_parts, ignore_index=True)

    test_files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    test_parts = []
    for i, path in enumerate(test_files, start=1):
        tw_path = path.parent / (path.stem.replace("__horizontal_well", "__typewell") + ".csv")
        df = build_well(path, tw_path, is_train=False)
        if df is not None:
            test_parts.append(df)
        if i % 200 == 0:
            print(f"  test {i}/{len(test_files)}", flush=True)
    test = pd.concat(test_parts, ignore_index=True)

    skip = {"well_id", "TVT", "delta_target", "is_lateral"}
    feature_cols = [c for c in train.columns if c not in skip]
    X = train[feature_cols].to_numpy(dtype=np.float32)
    y_delta = train["delta_target"].to_numpy(dtype=np.float32)
    y_abs = train["TVT"].to_numpy(dtype=np.float32)
    last_known = train["last_known_tvt"].to_numpy(dtype=np.float32)
    is_lateral = train["is_lateral"].to_numpy(dtype=bool)
    groups = train["well_id"].astype(str).to_numpy()
    Xt = test[feature_cols].to_numpy(dtype=np.float32)
    test_last_known = test["last_known_tvt"].to_numpy(dtype=np.float32)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        X=X,
        y_delta=y_delta,
        y_abs=y_abs,
        last_known=last_known,
        is_lateral=is_lateral,
        groups=groups,
        Xt=Xt,
        test_last_known=test_last_known,
        feature_cols=np.array(feature_cols, dtype=object),
    )
    return X, y_delta, y_abs, last_known, is_lateral, groups, Xt, test_last_known, feature_cols


def load_dataset():
    if CACHE_PATH.exists():
        print(f"Loading cached features from {CACHE_PATH}", flush=True)
        data = np.load(CACHE_PATH, allow_pickle=True)
        return (
            data["X"],
            data["y_delta"],
            data["y_abs"],
            data["last_known"],
            data["is_lateral"].astype(bool),
            data["groups"],
            data["Xt"],
            data["test_last_known"],
            data["feature_cols"].tolist(),
        )
    return _build_dataset()


def suggest_params(trial):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "max_bin": 255,
        "verbosity": 0,
        "max_depth": trial.suggest_categorical("max_depth", [6, 7, 8, 9, 10]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_weight": trial.suggest_categorical("min_child_weight", [5, 10, 20, 50, 100]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
    }


def train_one_fold(params, X, y_delta, last_known, y_abs, is_lateral, tr, va, dtest=None):
    dtrain = xgb.DMatrix(X[tr], label=y_delta[tr])
    dvalid = xgb.DMatrix(X[va], label=y_delta[va])
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=150,
        verbose_eval=False,
    )
    pred_delta = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
    pred_abs = last_known[va] + pred_delta
    lat_mask = is_lateral[va]
    fold_rmse = rmse(pred_abs[lat_mask], y_abs[va][lat_mask])
    test_delta = None
    if dtest is not None:
        test_delta = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    return model, pred_delta.astype(np.float32), test_delta, fold_rmse


def main():
    t0 = time.time()
    X, y_delta, y_abs, last_known, is_lateral, groups, Xt, test_last_known, feature_cols = load_dataset()
    print(f"Rows train={len(X)} test={len(Xt)} features={len(feature_cols)} GPU={GPU}", flush=True)

    folds = list(GroupKFold(n_splits=5).split(X, y_delta, groups))

    def objective(trial):
        params = suggest_params(trial)
        fold_rmses = []
        for fold, (tr, va) in enumerate(folds):
            _, _, _, fold_rmse = train_one_fold(
                params, X, y_delta, last_known, y_abs, is_lateral, tr, va
            )
            fold_rmses.append(fold_rmse)
            trial.report(float(np.mean(fold_rmses)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        score = float(np.mean(fold_rmses))
        trial.set_user_attr("fold_rmses_lateral", fold_rmses)
        print(f"trial {trial.number}: lateral={score:.4f} params={trial.params}", flush=True)
        return score

    sampler = optuna.samplers.TPESampler(seed=43, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="exp043_xgboost_optuna",
        storage=f"sqlite:///{STUDY_DB_PATH}",
        load_if_exists=True,
    )
    remaining = max(0, 30 - len([t for t in study.trials if t.state.is_finished()]))
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    best_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "max_bin": 255,
        "verbosity": 0,
        **study.best_trial.params,
    }
    print(f"Best trial {study.best_trial.number}: {study.best_value:.4f}", flush=True)
    print(f"Best params: {study.best_trial.params}", flush=True)

    oof_delta = np.zeros(len(y_delta), dtype=np.float32)
    dtest = xgb.DMatrix(Xt)
    test_folds = []
    fold_rmses_lateral = []
    fold_best_iters = []
    for fold, (tr, va) in enumerate(folds):
        model, pred_delta, test_delta, fold_rmse = train_one_fold(
            best_params, X, y_delta, last_known, y_abs, is_lateral, tr, va, dtest=dtest
        )
        oof_delta[va] = pred_delta
        test_folds.append(test_delta)
        fold_rmses_lateral.append(fold_rmse)
        fold_best_iters.append(int(model.best_iteration))
        print(f"final fold{fold}: best_iter={model.best_iteration} lateral_RMSE={fold_rmse:.4f}", flush=True)

    oof_abs = (last_known + oof_delta).astype(np.float32)
    test_delta = np.mean(np.column_stack(test_folds), axis=1).astype(np.float32)
    test_abs = (test_last_known + test_delta).astype(np.float32)
    cv_rmse_lateral = rmse(oof_abs[is_lateral], y_abs[is_lateral])
    cv_rmse_full = rmse(oof_abs, y_abs)
    cv_std = float(np.std(fold_rmses_lateral))

    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OOF_PATH, oof_abs)
    np.save(TEST_PATH, test_abs)

    base_rmse = 2.97733473777771
    notes = (
        "Optuna TPE tuned XGBoost over the exp038/exp026 feature set with no new "
        f"features. Lateral CV changed from exp038 {base_rmse:.4f} to "
        f"{cv_rmse_lateral:.4f}; tuning "
        f"{'helped' if cv_rmse_lateral < base_rmse else 'did not help'} versus the arbitrary exp038 params."
    )
    result = {
        "experiment_id": "exp043",
        "model": "xgboost_delta_bwell_ncc_optuna",
        "phase": "feature_engineering",
        "base_experiment": "exp038",
        "cv_rmse": cv_rmse_lateral,
        "cv_rmse_lateral": cv_rmse_lateral,
        "cv_rmse_full": cv_rmse_full,
        "cv_rmse_std": cv_std,
        "fold_rmses_lateral": fold_rmses_lateral,
        "fold_best_iterations": fold_best_iters,
        "base_cv_rmse_lateral": base_rmse,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_value": float(study.best_value),
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
        "features_to_add": [],
        "features_used": feature_cols,
        "n_features": len(feature_cols),
        "training_time_seconds": time.time() - t0,
        "notes": notes,
        "oof_path": "experiments/oof/oof_xgboost_exp043.npy",
        "test_path": "experiments/test_preds/test_xgboost_exp043.npy",
        "study_db_path": "experiments/results/exp043_optuna.db",
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    for path in [OOF_PATH, TEST_PATH, RESULT_PATH]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Required output missing or empty: {path}")

    print(f"OOF lateral RMSE: {cv_rmse_lateral:.4f}")
    print(f"OOF full RMSE: {cv_rmse_full:.4f}")
    print(f"EXPERIMENT COMPLETE: cv_rmse={cv_rmse_lateral:.4f}")


if __name__ == "__main__":
    main()
