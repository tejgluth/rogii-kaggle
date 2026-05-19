"""exp094: dev-selected top-3 blend of exp028 and artifact models.

Selection rule is intentionally simple and based on non-test-well OOF RMSE:
take the three best single OOF candidates among local exp028 plus the downloaded
artifact OOFs, then average them equally.

This avoids tuning blend weights on the three overlapping test wells. The JSON
also reports the strict clean-holdout score on those wells for audit only.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data/artifacts/wellbore-geology-prediction-artifacts"
RESULT = ROOT / "experiments/results/exp094.json"
TESTPRED_SUBMISSION = ROOT / "submissions/exp094_artifact_top3_devblend.csv"
CLEAN_OOF_SUBMISSION = ROOT / "submissions/exp094_artifact_top3_clean_oof.csv"


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

    candidates: dict[str, np.ndarray] = {}
    for path in sorted(ART.glob("*_oof_preds.pkl")):
        name = path.name.replace("_oof_preds.pkl", "")
        candidates[name] = (last + joblib.load(path).astype(np.float32)).astype(np.float32)

    cache = np.load(ROOT / "data/processed/exp043_exp026_features.npz", allow_pickle=True)
    local_groups = np.asarray(cache["groups"]).astype(str)
    local_lateral = cache["is_lateral"].astype(bool)
    if not np.array_equal(local_groups[local_lateral], wells):
        raise ValueError("local exp028 lateral row order does not match artifact train.csv")
    candidates["local_exp028"] = np.load(ROOT / "experiments/oof/oof_combined_exp028.npy").astype(np.float32)[local_lateral]

    single_scores = [
        {
            "name": name,
            "dev_rmse": rmse(pred, y, dev),
            "dev_mse": rmse(pred, y, dev) ** 2,
            "clean_holdout_rmse": rmse(pred, y, heldout),
            "clean_holdout_mse": rmse(pred, y, heldout) ** 2,
            "clean_holdout_bias": float((pred[heldout] - y[heldout]).mean()),
        }
        for name, pred in candidates.items()
    ]
    top3 = [row["name"] for row in sorted(single_scores, key=lambda r: r["dev_rmse"])[:3]]
    blend_oof = np.mean([candidates[name] for name in top3], axis=0).astype(np.float32)

    submission_parts = []
    for path in [
        ROOT / "submissions/exp028_full_kernel.csv",
        ROOT / "submissions/artifact_catboost-3.csv",
        ROOT / "submissions/artifact_lightgbm-4.csv",
    ]:
        sub = pd.read_csv(path)
        submission_parts.append(sub.set_index("id")["tvt"].astype(np.float32))
    out = pd.concat(submission_parts, axis=1)
    out.columns = ["local_exp028", "catboost-3", "lightgbm-4"]
    blended = out.mean(axis=1).reset_index()
    blended.columns = ["id", "tvt"]
    sample_ids = pd.read_csv(ROOT / "data/raw/rogii-wellbore-geology-prediction/sample_submission.csv")[["id"]]
    blended = sample_ids.merge(blended, on="id", how="left")
    if int(blended["tvt"].isna().sum()):
        raise ValueError("missing blended submission predictions")
    blended.to_csv(TESTPRED_SUBMISSION, index=False)

    clean_rows = train_map.loc[heldout, ["id"]].copy()
    clean_rows["tvt"] = blend_oof[heldout]
    clean_oof_sub = sample_ids.merge(clean_rows, on="id", how="left")
    if int(clean_oof_sub["tvt"].isna().sum()):
        raise ValueError("missing clean OOF submission predictions")
    clean_oof_sub.to_csv(CLEAN_OOF_SUBMISSION, index=False)

    payload = {
        "experiment_id": "exp094",
        "phase": "artifact_blend",
        "selection_rule": "equal average of three lowest dev OOF RMSE single candidates",
        "heldout_wells": heldout_wells,
        "members": top3,
        "weights": {name: 1.0 / len(top3) for name in top3},
        "dev_rmse": rmse(blend_oof, y, dev),
        "dev_mse": rmse(blend_oof, y, dev) ** 2,
        "clean_holdout_rmse": rmse(blend_oof, y, heldout),
        "clean_holdout_mse": rmse(blend_oof, y, heldout) ** 2,
        "clean_holdout_bias": float((blend_oof[heldout] - y[heldout]).mean()),
        "single_scores": sorted(single_scores, key=lambda r: r["dev_rmse"]),
        "testpred_submission_path": str(TESTPRED_SUBMISSION.relative_to(ROOT)),
        "clean_oof_submission_path": str(CLEAN_OOF_SUBMISSION.relative_to(ROOT)),
        "submission_note": (
            "clean_oof_submission uses OOF predictions for the overlapping test wells and matches the clean-holdout metric; "
            "testpred_submission uses artifact/local test predictions and is optimistic if overlap-well labels were used in training."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
