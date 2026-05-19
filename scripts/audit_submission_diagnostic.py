"""Audit submission diagnostics for the ROGII workspace.

This is not a Kaggle scorer. It checks local train/test mismatches that can make
OOF validation look excellent while a submission is poor, and it optionally uses
the local train labels for overlapping test well IDs as a diagnostic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
SAMPLE = DATA / "sample_submission.csv"


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err * err)))


def load_truth_by_id() -> dict[str, float]:
    truth: dict[str, float] = {}
    test_wells = sorted(p.stem.replace("__horizontal_well", "") for p in TEST_DIR.glob("*__horizontal_well.csv"))
    for well_id in test_wells:
        train_path = TRAIN_DIR / f"{well_id}__horizontal_well.csv"
        if not train_path.exists():
            continue
        df = pd.read_csv(train_path, usecols=["TVT"])
        truth.update({f"{well_id}_{i}": float(v) for i, v in enumerate(df["TVT"].to_numpy())})
    return truth


def audit_columns() -> None:
    train_h = next(iter(sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))))
    test_h = next(iter(sorted(TEST_DIR.glob("*__horizontal_well.csv"))))
    train_cols = set(pd.read_csv(train_h, nrows=0).columns)
    test_cols = set(pd.read_csv(test_h, nrows=0).columns)
    train_only_features = sorted(train_cols - test_cols - {"TVT"})
    print(f"train horizontal columns: {sorted(train_cols)}")
    print(f"test horizontal columns:  {sorted(test_cols)}")
    print(f"train-only non-target horizontal columns: {train_only_features}")

    train_wells = {p.stem.replace("__horizontal_well", "") for p in TRAIN_DIR.glob("*__horizontal_well.csv")}
    test_wells = {p.stem.replace("__horizontal_well", "") for p in TEST_DIR.glob("*__horizontal_well.csv")}
    overlap = sorted(train_wells & test_wells)
    print(f"train wells={len(train_wells)} test wells={len(test_wells)} overlap={len(overlap)} {overlap}")


def score_submission(path: Path, truth: dict[str, float], sample_ids: pd.Series) -> None:
    sub = pd.read_csv(path)
    if "tvt" in sub.columns:
        pred_col = "tvt"
    elif "TVT" in sub.columns:
        pred_col = "TVT"
    else:
        print(f"{path.name}: skipped, no tvt column")
        return
    if len(sub) != len(sample_ids):
        print(f"{path.name}: skipped, row count {len(sub)} != sample {len(sample_ids)}")
        return
    ids_match = sub["id"].equals(sample_ids)
    y = sub["id"].map(truth).to_numpy(dtype=float)
    if np.isnan(y).any():
        print(f"{path.name}: skipped, missing local overlap truth for {int(np.isnan(y).sum())} rows")
        return
    pred = sub[pred_col].to_numpy(dtype=float)
    err = pred - y
    print(
        f"{path.name:36s} ids_match={str(ids_match):5s} "
        f"rmse={rmse(err):8.4f} mse={float(np.mean(err * err)):10.4f} "
        f"mae={float(np.mean(np.abs(err))):8.4f} bias={float(np.mean(err)):8.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", nargs="*", help="Submission CSVs to score. Defaults to submissions/*.csv")
    args = parser.parse_args()

    audit_columns()
    truth = load_truth_by_id()
    sample = pd.read_csv(SAMPLE)
    paths = [Path(p) for p in args.submissions]
    if not paths:
        paths = sorted((ROOT / "submissions").glob("*.csv"))
    print("\nDiagnostic scores against local overlapping train labels:")
    for path in paths:
        score_submission(path, truth, sample["id"])


if __name__ == "__main__":
    main()
