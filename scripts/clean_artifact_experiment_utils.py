"""Shared helpers for leak-safe artifact experiments.

The sample-submission wells are audit-only.  Helpers in this module expose the
mask but never use heldout labels for model selection.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data/artifacts/wellbore-geology-prediction-artifacts"
COMP = ROOT / "data/raw/rogii-wellbore-geology-prediction"
HELDOUT_WELLS = np.array(["000d7d20", "00bbac68", "00e12e8b"])


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    err = pred.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt(np.mean(err * err)))


def well_equal_rmse(pred: np.ndarray, y: np.ndarray, wells: np.ndarray) -> float:
    frame = pd.DataFrame(
        {
            "well": wells.astype(str),
            "sqerr": (pred.astype(np.float64) - y.astype(np.float64)) ** 2,
        }
    )
    return float(np.sqrt(frame.groupby("well", sort=True)["sqerr"].mean().mean()))


def per_well_rmse(pred: np.ndarray, y: np.ndarray, wells: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "well": wells.astype(str),
            "sqerr": (pred.astype(np.float64) - y.astype(np.float64)) ** 2,
        }
    )
    return {
        str(well): float(np.sqrt(mse))
        for well, mse in frame.groupby("well", sort=True)["sqerr"].mean().items()
    }


def load_artifact_frame() -> tuple[pd.DataFrame, list[str]]:
    columns = pd.read_csv(ART / "train.csv", nrows=0).columns.tolist()
    feature_cols = [c for c in columns if c not in ("well", "id", "target")]
    dtypes = {c: "float32" for c in feature_cols + ["target"]}
    dtypes.update({"well": "string", "id": "string"})
    return pd.read_csv(ART / "train.csv", dtype=dtypes), feature_cols


def load_artifact_members(last_known_tvt: np.ndarray) -> dict[str, np.ndarray]:
    members: dict[str, np.ndarray] = {}
    for path in sorted(ART.glob("*_oof_preds.pkl")):
        name = path.name.replace("_oof_preds.pkl", "")
        members[name] = (last_known_tvt + joblib.load(path).astype(np.float32)).astype(np.float32)
    if not members:
        raise FileNotFoundError(f"No artifact OOF prediction files found under {ART}")
    return members


def sample_ids() -> pd.DataFrame:
    return pd.read_csv(COMP / "sample_submission.csv", usecols=["id"], dtype={"id": "string"})


def make_submission(ids: np.ndarray, pred: np.ndarray, heldout_mask: np.ndarray, output_path: Path) -> pd.DataFrame:
    clean_rows = pd.DataFrame(
        {
            "id": ids[heldout_mask].astype(str),
            "tvt": pred.astype(np.float32),
        }
    )
    sub = sample_ids().merge(clean_rows, on="id", how="left")
    if int(sub["tvt"].isna().sum()):
        raise ValueError("missing predictions for sample_submission ids")
    if not np.isfinite(sub["tvt"].to_numpy(dtype=np.float64)).all():
        raise ValueError("non-finite predictions in submission")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    return sub


def submission_report(sub: pd.DataFrame) -> dict[str, object]:
    sample = sample_ids()
    return {
        "submission_rows": int(len(sub)),
        "sample_rows": int(len(sample)),
        "ids_match_sample_order": bool(sub["id"].astype(str).equals(sample["id"].astype(str))),
        "missing_tvt": int(sub["tvt"].isna().sum()),
        "finite_tvt": bool(np.isfinite(sub["tvt"].to_numpy(dtype=np.float64)).all()),
        "tvt_min": float(sub["tvt"].min()),
        "tvt_max": float(sub["tvt"].max()),
        "tvt_mean": float(sub["tvt"].mean()),
    }


def audit_metrics(pred: np.ndarray, y: np.ndarray, wells: np.ndarray) -> dict[str, object]:
    return {
        "rmse": rmse(pred, y),
        "mse": rmse(pred, y) ** 2,
        "well_equal_rmse": well_equal_rmse(pred, y, wells),
        "bias": float(np.mean(pred.astype(np.float64) - y.astype(np.float64))),
        "per_well_rmse": per_well_rmse(pred, y, wells),
    }
