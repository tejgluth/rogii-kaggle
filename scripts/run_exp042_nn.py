"""exp042: PyTorch MLP on exp026 features with delta target.

Train only on lateral rows, using GroupKFold by well_id. Predictions are saved
as absolute TVT in full original CSV row order.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_exp026_bwell_ncc import TEST_DIR, TRAIN_DIR, build_well, rmse

OOF_PATH = ROOT / "experiments/oof/oof_nn_exp042.npy"
TEST_PATH = ROOT / "experiments/test_preds/test_nn_exp042.npy"
RESULT_PATH = ROOT / "experiments/results/exp042.json"


class MLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        dims = [n_features, 256, 256, 128, 64]
        layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers.extend([
                nn.Linear(din, dout),
                nn.BatchNorm1d(dout),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            ])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def standardize_fit_transform(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
    means = np.nanmean(x_train, axis=0, dtype=np.float64).astype(np.float32)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    stds = np.nanstd(x_train, axis=0, dtype=np.float64).astype(np.float32)
    stds = np.where(np.isfinite(stds) & (stds > 1e-6), stds, 1.0).astype(np.float32)

    def transform(x):
        z = (x - means) / stds
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return transform(x_train), transform(x_val), transform(x_test)


def load_feature_frames():
    t0 = time.time()
    print("Loading train wells...", flush=True)
    train_files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    train_dfs = []
    for i, path in enumerate(train_files):
        tw_path = path.parent / (path.stem.replace("__horizontal_well", "__typewell") + ".csv")
        df = build_well(path, tw_path, is_train=True)
        if df is not None:
            train_dfs.append(df)
        if (i + 1) % 100 == 0:
            print(f"  loaded train {i + 1}/{len(train_files)} ({time.time() - t0:.0f}s)", flush=True)

    import pandas as pd
    train = pd.concat(train_dfs, ignore_index=True)
    print(f"train rows={len(train)} cols={train.shape[1]}", flush=True)

    print("Loading test wells...", flush=True)
    test_files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    test_dfs = []
    for i, path in enumerate(test_files):
        tw_path = path.parent / (path.stem.replace("__horizontal_well", "__typewell") + ".csv")
        df = build_well(path, tw_path, is_train=False)
        if df is not None:
            test_dfs.append(df)
        if (i + 1) % 100 == 0:
            print(f"  loaded test {i + 1}/{len(test_files)} ({time.time() - t0:.0f}s)", flush=True)
    test = pd.concat(test_dfs, ignore_index=True)
    print(f"test rows={len(test)} cols={test.shape[1]}", flush=True)
    return train, test


@torch.no_grad()
def predict_batches(model, x_np: np.ndarray, device: torch.device, batch_size: int = 65536):
    model.eval()
    preds = []
    for start in range(0, len(x_np), batch_size):
        xb = torch.from_numpy(x_np[start:start + batch_size]).to(device, non_blocking=True)
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def train_fold(x_tr, y_tr, x_va, x_te, device, fold):
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std() if y_tr.std() > 1e-6 else 1.0)
    y_tr_s = ((y_tr - y_mean) / y_std).astype(np.float32)

    model = MLP(x_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr_s))
    loader = DataLoader(
        ds,
        batch_size=4096,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    for epoch in range(30):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            total_n += len(xb)
        scheduler.step()
        if epoch in {0, 4, 9, 19, 29}:
            print(f"  fold{fold} epoch{epoch + 1:02d} train_mse={total_loss / total_n:.5f}", flush=True)

    val_delta = predict_batches(model, x_va, device) * y_std + y_mean
    test_delta = predict_batches(model, x_te, device) * y_std + y_mean
    return val_delta.astype(np.float32), test_delta.astype(np.float32)


def main():
    start_time = time.time()
    np.random.seed(42)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        print("EXPERIMENT FAILED: CUDA is not available")
        return
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}", flush=True)

    for path in [OOF_PATH.parent, TEST_PATH.parent, RESULT_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)

    train, test = load_feature_frames()
    skip = {"well_id", "TVT", "delta_target", "is_lateral"}
    feature_cols = [c for c in train.columns if c not in skip]
    print(f"#features: {len(feature_cols)}", flush=True)

    x_all = train[feature_cols].values.astype(np.float32)
    x_test_all = test[feature_cols].values.astype(np.float32)
    y_delta = train["delta_target"].values.astype(np.float32)
    y_abs = train["TVT"].values.astype(np.float32)
    last_known = train["last_known_tvt"].values.astype(np.float32)
    is_lateral = train["is_lateral"].values.astype(bool)
    groups = train["well_id"].values

    test_last_known = test["last_known_tvt"].values.astype(np.float32)
    test_is_lateral = test["is_lateral"].values.astype(bool)

    oof_abs = train["TVT_input"].fillna(0).values.astype(np.float32)
    oof_abs[is_lateral] = np.nan
    test_abs_folds = []
    fold_rmses_lateral = []
    fold_rmses_full = []

    gkf = GroupKFold(n_splits=5)
    lateral_idx = np.flatnonzero(is_lateral)
    test_lateral_idx = np.flatnonzero(test_is_lateral)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x_all, y_delta, groups)):
        tr_lat = tr_idx[is_lateral[tr_idx]]
        va_lat = va_idx[is_lateral[va_idx]]
        print(
            f"fold{fold}: train_lateral={len(tr_lat)} val_lateral={len(va_lat)}",
            flush=True,
        )
        x_tr, x_va, x_te = standardize_fit_transform(
            x_all[tr_lat], x_all[va_lat], x_test_all[test_lateral_idx]
        )
        val_delta, test_delta = train_fold(x_tr, y_delta[tr_lat], x_va, x_te, device, fold)
        oof_abs[va_lat] = last_known[va_lat] + val_delta

        fold_test_abs = test["TVT_input"].fillna(0).values.astype(np.float32)
        fold_test_abs[test_lateral_idx] = test_last_known[test_lateral_idx] + test_delta
        test_abs_folds.append(fold_test_abs)

        fold_lat_rmse = rmse(oof_abs[va_lat], y_abs[va_lat])
        fold_full_rmse = rmse(oof_abs[va_idx], y_abs[va_idx])
        fold_rmses_lateral.append(fold_lat_rmse)
        fold_rmses_full.append(fold_full_rmse)
        print(f" fold{fold}: lateral_RMSE={fold_lat_rmse:.4f} full_RMSE={fold_full_rmse:.4f}", flush=True)

    if np.isnan(oof_abs[lateral_idx]).any():
        raise RuntimeError("OOF contains missing lateral predictions")

    test_abs = np.mean(np.column_stack(test_abs_folds), axis=1).astype(np.float32)
    overall_lateral = rmse(oof_abs[is_lateral], y_abs[is_lateral])
    overall_full = rmse(oof_abs, y_abs)
    cv_std = float(np.std(fold_rmses_lateral))
    train_seconds = float(time.time() - start_time)

    np.save(OOF_PATH, oof_abs.astype(np.float32))
    np.save(TEST_PATH, test_abs.astype(np.float32))

    notes = (
        "PyTorch MLP trained on exp026 features and lateral delta target. "
        f"Lateral RMSE {overall_lateral:.4f} vs exp026 LightGBM lateral 4.1602; "
        "non-tree model intended mainly for stacking diversity."
    )
    result = {
        "experiment_id": "exp042",
        "model": "pytorch_nn",
        "phase": "feature_engineering",
        "base_experiment": "exp026",
        "description": "PyTorch neural net on exp026 features with delta target.",
        "cv_rmse": overall_full,
        "cv_rmse_lateral": overall_lateral,
        "cv_rmse_std": cv_std,
        "fold_rmses_lateral": fold_rmses_lateral,
        "fold_rmses_full": fold_rmses_full,
        "n_features": len(feature_cols),
        "features_used": feature_cols,
        "model_arch_str": "MLP(73 -> 256 -> 256 -> 128 -> 64 -> 1), BatchNorm+ReLU+Dropout(0.2)",
        "training_time_seconds": train_seconds,
        "notes": notes,
        "oof_path": "experiments/oof/oof_nn_exp042.npy",
        "test_path": "experiments/test_preds/test_nn_exp042.npy",
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    for path in [OOF_PATH, TEST_PATH, RESULT_PATH]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty output: {path}")
    print(f"saved {OOF_PATH} shape={oof_abs.shape} bytes={OOF_PATH.stat().st_size}", flush=True)
    print(f"saved {TEST_PATH} shape={test_abs.shape} bytes={TEST_PATH.stat().st_size}", flush=True)
    print(f"saved {RESULT_PATH} bytes={RESULT_PATH.stat().st_size}", flush=True)
    print(f"EXPERIMENT COMPLETE: cv_rmse={overall_full:.4f}", flush=True)


if __name__ == "__main__":
    main()
