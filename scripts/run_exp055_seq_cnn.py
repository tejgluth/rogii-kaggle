"""exp055: per-well 1D-CNN on delta target using cached exp026 features.
Different model family from trees — meant for stacking diversity.

Per-well sequences (variable length), 5-fold GroupKFold by well. Each sample sees a
context window via 1D conv. Predicts delta_target; final OOF = last_known + pred.
"""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda")

CACHE = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
X = CACHE["X"].astype(np.float32)
y_d = CACHE["y_delta"].astype(np.float32)
y_a = CACHE["y_abs"].astype(np.float32)
lk = CACHE["last_known"].astype(np.float32)
is_lat = CACHE["is_lateral"]
grp = np.asarray(CACHE["groups"])
Xt = CACHE["Xt"].astype(np.float32)
tlk = CACHE["test_last_known"].astype(np.float32)
print(f"features: {X.shape}")

# Normalise features
mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0) + 1e-6
X = np.nan_to_num((X - mu) / sd, nan=0.0).astype(np.float32)
Xt = np.nan_to_num((Xt - mu) / sd, nan=0.0).astype(np.float32)
test_grp = np.repeat(np.array(["t1", "t2", "t3"]), [Xt.shape[0]//3]*3)  # placeholder; only need per-well order
# rebuild test_grp from test files
import pandas as pd
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"
gs = []
for p in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
    n = len(pd.read_csv(p, usecols=["MD"]))
    gs.append(np.full(n, p.stem.replace("__horizontal_well", "")))
test_grp = np.concatenate(gs)
assert len(test_grp) == len(Xt)

# Pre-compute per-well row indices
def well_segments(groups):
    out = {}
    for w in np.unique(groups):
        idx = np.where(groups == w)[0]
        out[w] = idx
    return out

train_segs = well_segments(grp)
test_segs = well_segments(test_grp)


class WellSeqDataset(Dataset):
    """Yields one well at a time as (T, F) tensor + (T,) target + (T,) sample weight."""
    def __init__(self, wells, X, y=None, segs=None, is_lat=None):
        self.wells = list(wells)
        self.X, self.y, self.segs, self.is_lat = X, y, segs, is_lat
    def __len__(self): return len(self.wells)
    def __getitem__(self, i):
        w = self.wells[i]
        idx = self.segs[w]
        xb = torch.from_numpy(self.X[idx])
        yb = torch.from_numpy(self.y[idx]) if self.y is not None else torch.zeros(len(idx))
        lat = torch.from_numpy(self.is_lat[idx]).float() if self.is_lat is not None else torch.ones(len(idx))
        return xb, yb, lat, idx


def collate(batch):
    # variable length — process one at a time
    return batch


class SeqCNN(nn.Module):
    def __init__(self, fin, h=128, k=15):
        super().__init__()
        pad = k // 2
        self.proj = nn.Linear(fin, h)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(h, h, k, padding=pad), nn.GELU(), nn.BatchNorm1d(h)),
            nn.Sequential(nn.Conv1d(h, h, k, padding=pad, dilation=1), nn.GELU(), nn.BatchNorm1d(h)),
            nn.Sequential(nn.Conv1d(h, h, k, padding=2*pad, dilation=2), nn.GELU(), nn.BatchNorm1d(h)),
            nn.Sequential(nn.Conv1d(h, h, k, padding=4*pad, dilation=4), nn.GELU(), nn.BatchNorm1d(h)),
        ])
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, x):  # x: (T, F)
        h = self.proj(x).transpose(0, 1).unsqueeze(0)  # (1, h, T)
        for b in self.blocks:
            h = h + b(h)
        h = h.squeeze(0).transpose(0, 1)  # (T, h)
        return self.head(h).squeeze(-1)


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def train_fold(tr_wells, va_wells, te_wells, epochs=40, lr=2e-3):
    tr_ds = WellSeqDataset(tr_wells, X, y_d, train_segs, is_lat)
    va_ds = WellSeqDataset(va_wells, X, y_d, train_segs, is_lat)
    te_ds = WellSeqDataset(te_wells, Xt, None, test_segs, None)
    model = SeqCNN(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber = nn.SmoothL1Loss(beta=2.0, reduction="none")
    best_val = float("inf"); best_oof = None; best_test = None
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(tr_ds))
        total = 0.0; n = 0
        for j in perm:
            xb, yb, lat, _ = tr_ds[j]
            xb = xb.to(DEVICE); yb = yb.to(DEVICE); lat = lat.to(DEVICE)
            pred = model(xb)
            w = 1.0 + 2.0 * lat  # upweight lateral rows (they are scored)
            loss = (huber(pred, yb) * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.shape[0]; n += xb.shape[0]
        sched.step()
        # validate
        model.eval()
        oof_pred = np.zeros(len(y_d), np.float32)
        with torch.no_grad():
            for j in range(len(va_ds)):
                xb, _, _, idx = va_ds[j]
                xb = xb.to(DEVICE)
                oof_pred[idx] = model(xb).cpu().numpy()
        va_mask = np.isin(grp, va_wells) & is_lat
        r = rmse(lk[va_mask] + oof_pred[va_mask], y_a[va_mask])
        if r < best_val:
            best_val = r
            # extract va indices preds + test preds
            best_oof = oof_pred.copy()
            te_pred = np.zeros(len(Xt), np.float32)
            with torch.no_grad():
                for j in range(len(te_ds)):
                    xb, _, _, idx = te_ds[j]
                    xb = xb.to(DEVICE)
                    te_pred[idx] = model(xb).cpu().numpy()
            best_test = te_pred
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:02d}: train_loss={total/n:.4f}  val_lat_rmse={r:.4f}  best={best_val:.4f}", flush=True)
    return best_oof, best_test, best_val


# 5-fold GroupKFold
gkf = GroupKFold(5)
oof_d_all = np.zeros(len(y_d), np.float32)
test_d_folds = []
unique_wells = np.array(sorted(np.unique(grp)))
all_te_wells = list(np.unique(test_grp))

for fi, (tr_idx, va_idx) in enumerate(gkf.split(unique_wells, groups=unique_wells)):
    print(f"\n=== fold {fi} ===")
    tr_wells = unique_wells[tr_idx]
    va_wells = unique_wells[va_idx]
    oof_pred, te_pred, bv = train_fold(tr_wells, va_wells, all_te_wells)
    # Pick only the va rows from oof_pred (others are stale)
    va_mask_all = np.isin(grp, va_wells)
    oof_d_all[va_mask_all] = oof_pred[va_mask_all]
    test_d_folds.append(te_pred)
    print(f"  fold{fi} best_lat_rmse={bv:.4f}")

oof_abs = (lk + oof_d_all).astype(np.float32)
test_abs = (tlk + np.mean(np.column_stack(test_d_folds), axis=1)).astype(np.float32)
r_lat = rmse(oof_abs[is_lat], y_a[is_lat])
print(f"\n=== exp055 lateral RMSE: {r_lat:.4f} ===")
np.save(ROOT / "experiments/oof/oof_seqcnn_exp055.npy", oof_abs)
np.save(ROOT / "experiments/test_preds/test_seqcnn_exp055.npy", test_abs)
json.dump({"experiment_id": "exp055", "model": "seq_cnn", "phase": "feature_engineering",
           "cv_rmse_lateral": r_lat,
           "notes": "1D CNN with dilated convs on cached exp026 features, delta target, huber loss + lateral upweighting"},
          open(ROOT / "experiments/results/exp055.json", "w"), indent=2)
