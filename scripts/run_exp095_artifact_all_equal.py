"""exp095: untuned equal average of all downloaded artifact OOFs plus exp028.

Unlike exp094, this does not select members or weights from validation labels.
It simply averages every downloaded artifact OOF prediction and local exp028.
The clean OOF submission is the only submission written, so the saved candidate
matches the strict overlap-well diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data/artifacts/wellbore-geology-prediction-artifacts"
RESULT = ROOT / "experiments/results/exp095.json"
SUBMISSION = ROOT / "submissions/exp095_artifact_all_equal_clean_oof.csv"


def rmse(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    err = pred[mask] - y[mask]
    return float(np.sqrt(np.mean(err * err)))


def main() -> None:
    heldout_wells = ["000d7d20", "00bbac68", "00e12e8b"]
    train_map = pd.read_csv(
        ART / "train.csv",
        usecols=["well", "id", "target", "last_known_tvt"],
        dtype={"well": "string", "id": "string", "target": "float32", "last_known_tvt": "float32"},
    )
    y = (
        train_map["last_known_tvt"].to_numpy(np.float32)
        + train_map["target"].to_numpy(np.float32)
    ).astype(np.float32)
    last = train_map["last_known_tvt"].to_numpy(np.float32)
    wells = train_map["well"].astype(str).to_numpy()
    heldout = np.isin(wells, heldout_wells)
    dev = ~heldout

    members: dict[str, np.ndarray] = {}
    for path in sorted(ART.glob("*_oof_preds.pkl")):
        name = path.name.replace("_oof_preds.pkl", "")
        members[name] = (last + joblib.load(path).astype(np.float32)).astype(np.float32)

    cache = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
    local_groups = np.asarray(cache["groups"]).astype(str)
    local_lateral = cache["is_lateral"].astype(bool)
    if not np.array_equal(local_groups[local_lateral], wells):
        raise ValueError("local exp028 lateral row order does not match artifact train.csv")
    members["local_exp028"] = np.load(ROOT / "experiments/oof/oof_combined_exp028.npy").astype(np.float32)[local_lateral]

    blend = np.mean(list(members.values()), axis=0).astype(np.float32)
    sample_ids = pd.read_csv(ROOT / "data/raw/rogii-wellbore-geology-prediction/sample_submission.csv")[["id"]]
    clean_rows = train_map.loc[heldout, ["id"]].copy()
    clean_rows["tvt"] = blend[heldout]
    sub = sample_ids.merge(clean_rows, on="id", how="left")
    if int(sub["tvt"].isna().sum()):
        raise ValueError("missing clean OOF submission predictions")
    sub.to_csv(SUBMISSION, index=False)

    payload = {
        "experiment_id": "exp095",
        "phase": "artifact_all_equal",
        "selection_rule": "untuned equal average of all downloaded artifact OOFs plus local exp028",
        "heldout_wells": heldout_wells,
        "members": sorted(members),
        "weights": {name: 1.0 / len(members) for name in sorted(members)},
        "dev_rmse": rmse(blend, y, dev),
        "dev_mse": rmse(blend, y, dev) ** 2,
        "clean_holdout_rmse": rmse(blend, y, heldout),
        "clean_holdout_mse": rmse(blend, y, heldout) ** 2,
        "clean_holdout_bias": float((blend[heldout] - y[heldout]).mean()),
        "submission_path": str(SUBMISSION.relative_to(ROOT)),
        "note": "This file is clean OOF-only for the overlapping test wells; no artifact test predictions are used.",
    }
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
