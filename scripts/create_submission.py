"""
Create Final Submission
=======================
Converts final test predictions to Kaggle submission format.
Run after stacking phase is complete.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/raw/rogii-wellbore-geology-prediction"
TEST_DIR = DATA / "test"
SAMPLE = DATA / "sample_submission.csv"


def create_submission(
    test_preds_path: str = "experiments/test_preds/final_ensemble.npy",
    output_path: str = "submission.csv",
):
    """Create a Kaggle submission CSV from full test-row predictions.

    The model prediction array must have one value per row in the sorted test
    horizontal CSV files. Kaggle only expects rows where TVT_input is missing.
    """

    # Load test predictions
    preds_path = ROOT / test_preds_path
    if not preds_path.exists():
        print(f"ERROR: Test predictions not found: {preds_path}")
        print("Run stacking phase first (Phase 4)")
        sys.exit(1)

    test_preds = np.load(str(preds_path))
    print(f"Loaded test predictions: shape={test_preds.shape}")

    sample_sub = pd.read_csv(SAMPLE)
    rows = []
    cursor = 0
    for path in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
        well_id = path.stem.replace("__horizontal_well", "")
        df = pd.read_csv(path, usecols=["TVT_input"])
        n_rows = len(df)
        well_preds = test_preds[cursor:cursor + n_rows]
        cursor += n_rows
        mask = df["TVT_input"].isna().to_numpy()
        rows.append(pd.DataFrame({
            "id": [f"{well_id}_{i}" for i in np.flatnonzero(mask)],
            "tvt": well_preds[mask],
        }))
    if cursor != len(test_preds):
        raise ValueError(f"Prediction length mismatch: consumed {cursor}, got {len(test_preds)}")

    predicted = pd.concat(rows, ignore_index=True)
    submission = sample_sub[["id"]].merge(predicted, on="id", how="left")
    missing = int(submission["tvt"].isna().sum())
    if missing:
        raise ValueError(f"Missing {missing} sample ids after lateral-row filtering")

    output_path = ROOT / output_path
    submission.to_csv(str(output_path), index=False)

    print(f"\nSubmission saved: {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"TVT range: {submission['tvt'].min():.2f} - {submission['tvt'].max():.2f}")
    print(f"\nReady to submit:")
    print(f"  kaggle competitions submit -c rogii-wellbore-geology-prediction "
          f"-f {output_path} -m 'final ensemble'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default="experiments/test_preds/final_ensemble.npy")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    create_submission(args.preds, args.output)
