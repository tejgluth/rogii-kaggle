"""
Create Final Submission
=======================
Converts final test predictions to Kaggle submission format.
Run after stacking phase is complete.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import cudf as pd
    GPU = True
except ImportError:
    import pandas as pd
    GPU = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def create_submission(
    test_preds_path: str = "experiments/test_preds/final_ensemble.npy",
    output_path: str = "submission.csv",
):
    """Create a Kaggle submission CSV from test predictions."""

    # Load test predictions
    preds_path = ROOT / test_preds_path
    if not preds_path.exists():
        print(f"ERROR: Test predictions not found: {preds_path}")
        print("Run stacking phase first (Phase 4)")
        sys.exit(1)

    test_preds = np.load(str(preds_path))
    print(f"Loaded test predictions: shape={test_preds.shape}")

    # Load sample submission to get correct format
    sample_sub_candidates = [
        ROOT / "data" / "raw" / "sample_submission.csv",
        ROOT / "data" / "raw" / "test" / "sample_submission.csv",
    ]

    sample_sub = None
    for candidate in sample_sub_candidates:
        if candidate.exists():
            if GPU:
                import cudf
                sample_sub = cudf.read_csv(str(candidate))
            else:
                import pandas
                sample_sub = pandas.read_csv(str(candidate))
            print(f"Loaded sample submission: {candidate}")
            break

    if sample_sub is None:
        print("WARNING: sample_submission.csv not found")
        print("Creating submission with row_id index")
        if GPU:
            import cudf
            submission = cudf.DataFrame({
                "row_id": range(len(test_preds)),
                "tvt": test_preds,
            })
        else:
            import pandas
            submission = pandas.DataFrame({
                "row_id": range(len(test_preds)),
                "tvt": test_preds,
            })
    else:
        # Find target column name
        target_col = next(
            (c for c in sample_sub.columns if c.lower() in ("tvt", "target")),
            sample_sub.columns[-1]
        )
        sample_sub[target_col] = test_preds
        submission = sample_sub

    # Apply post-processing
    from src.postprocess import smooth_predictions
    submission_np = test_preds.copy()

    output_path = ROOT / output_path
    if GPU:
        submission.to_csv(str(output_path), index=False)
    else:
        submission.to_csv(str(output_path), index=False)

    print(f"\nSubmission saved: {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"TVT range: {test_preds.min():.2f} — {test_preds.max():.2f}")
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
