"""
Data Loader
===========
Loads and merges well data from competition files.
Handles both CSV and LAS formats.
"""

from pathlib import Path

try:
    import cudf as pd
    GPU = True
except ImportError:
    import pandas as pd
    GPU = False

import numpy as np

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw"


def load_las_file(path: str) -> "pd.DataFrame":
    """Load a LAS file into a DataFrame."""
    try:
        import lasio
        las = lasio.read(path)
        df = las.df().reset_index()
        df.columns = [c.lower() for c in df.columns]
        return df
    except ImportError:
        raise ImportError("Install lasio: pip install lasio")


def load_well_file(path: str) -> "pd.DataFrame":
    """Load a single well file (CSV or LAS)."""
    path = Path(path)
    if path.suffix.lower() in (".las", ".LAS"):
        return load_las_file(str(path))
    else:
        if GPU:
            return pd.read_csv(str(path))
        else:
            import pandas
            df = pandas.read_csv(str(path))
            return df


def load_wells(split: str = "train") -> "pd.DataFrame":
    """
    Load all wells from train or test split.
    Returns a single DataFrame with a well_id column.

    Args:
        split: 'train' or 'test'

    Returns:
        DataFrame with all wells concatenated, well_id column added
    """
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {split_dir}\n"
            f"Run: bash scripts/download_data.sh"
        )

    dfs = []
    well_files = sorted(split_dir.glob("*"))

    for well_path in well_files:
        if well_path.suffix.lower() in (".csv", ".las", ".LAS"):
            try:
                df = load_well_file(str(well_path))
                df["well_id"] = well_path.stem
                df["well_file"] = well_path.name
                dfs.append(df)
            except Exception as e:
                print(f"WARNING: Could not load {well_path.name}: {e}")

    if not dfs:
        raise ValueError(f"No well files found in {split_dir}")

    if GPU:
        import cudf
        combined = cudf.concat(dfs, ignore_index=True)
    else:
        import pandas
        combined = pandas.concat(
            [df if hasattr(df, "_repr_html_") else df for df in dfs],
            ignore_index=True
        )

    # Standardize column names
    combined.columns = [c.lower().strip() for c in combined.columns]

    print(f"Loaded {split}: {len(combined):,} rows, "
          f"{combined['well_id'].nunique()} wells")
    return combined


def load_typewell() -> "pd.DataFrame":
    """Load Typewell reference data."""
    typewell_candidates = [
        DATA_ROOT / "typewell.csv",
        DATA_ROOT / "train" / "typewell.csv",
        DATA_ROOT / "typewell.las",
    ]
    for path in typewell_candidates:
        if path.exists():
            df = load_well_file(str(path))
            df.columns = [c.lower().strip() for c in df.columns]
            print(f"Loaded typewell: {len(df):,} rows from {path.name}")
            return df

    raise FileNotFoundError(
        "Typewell file not found. Check competition data structure."
    )


def get_target_and_features(df: "pd.DataFrame"):
    """Split DataFrame into X, y, and well_id groups."""
    target_col = "tvt"
    exclude_cols = {target_col, "well_id", "well_file", "index"}

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. "
                         f"Available: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if GPU:
        import cudf
        X = df[feature_cols].to_pandas().values.astype(np.float32)
        y = df[target_col].to_pandas().values.astype(np.float32)
        groups = df["well_id"].to_pandas().values
    else:
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)
        groups = df["well_id"].values

    return X, y, groups, feature_cols
