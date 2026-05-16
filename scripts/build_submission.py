"""Build submission CSV from a test-prediction npy file (full test-CSV length, 19221).
Filters to rows where TVT_input is NaN and matches sample_submission ids.
Usage: python scripts/build_submission.py <test_npy_path> [output_csv]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction/test"
SAMPLE = ROOT / "data/raw/rogii-wellbore-geology-prediction/sample_submission.csv"


def main(test_npy: str, output: str = "submission.csv"):
    preds = np.load(test_npy)
    sample = pd.read_csv(SAMPLE)
    test_files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))

    rows = []
    cursor = 0
    for p in test_files:
        well = p.stem.replace("__horizontal_well", "")
        df = pd.read_csv(p)
        n = len(df)
        well_preds = preds[cursor:cursor + n]
        cursor += n
        mask = df["TVT_input"].isna()
        ids = [f"{well}_{i}" for i in df.index[mask]]
        vals = well_preds[mask.values]
        rows.append(pd.DataFrame({"id": ids, "tvt": vals}))
    assert cursor == len(preds), f"Mismatch: cursor={cursor}, preds_len={len(preds)}"
    sub = pd.concat(rows, ignore_index=True)
    sub = sample[["id"]].merge(sub, on="id", how="left")
    missing = sub["tvt"].isna().sum()
    print(f"Built submission: {len(sub)} rows, {missing} missing.")
    if missing:
        print("WARNING: missing predictions — id alignment may be wrong.")
        sub["tvt"] = sub["tvt"].fillna(sub["tvt"].median())
    sub.to_csv(output, index=False)
    print(f"Wrote {output}")
    print(sub.head())
    print(f"TVT stats: min={sub['tvt'].min():.2f} max={sub['tvt'].max():.2f} mean={sub['tvt'].mean():.2f}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "experiments/test_preds/test_stack_exp024.npy"
    out = sys.argv[2] if len(sys.argv) > 2 else "submission.csv"
    main(src, out)
