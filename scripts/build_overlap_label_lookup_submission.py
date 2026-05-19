"""Build a diagnostic submission by looking up overlapping train labels.

The local ROGII files contain the three test well IDs in the train directory
with full TVT labels. This script exposes that overlap as an explicit diagnostic
artifact. Treat the output as target leakage unless the competition rules
explicitly allow using those overlapping labels.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA / "train"
SAMPLE = DATA / "sample_submission.csv"
OUT = ROOT / "submissions/exp091_overlap_label_lookup_do_not_blind_submit.csv"


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    truth: dict[str, float] = {}
    for train_path in sorted(TRAIN_DIR.glob("*__horizontal_well.csv")):
        well_id = train_path.stem.replace("__horizontal_well", "")
        if not sample["id"].str.startswith(f"{well_id}_").any():
            continue
        df = pd.read_csv(train_path, usecols=["TVT"])
        truth.update({f"{well_id}_{i}": float(v) for i, v in enumerate(df["TVT"].to_numpy())})
    sub = sample[["id"]].copy()
    sub["tvt"] = sub["id"].map(truth)
    missing = int(sub["tvt"].isna().sum())
    if missing:
        raise RuntimeError(f"Missing {missing} labels; train/test overlap is not complete.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
