#!/usr/bin/env python3
"""Exploratory data analysis for ROGII exp001."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
OOF_DIR = REPO_ROOT / "experiments" / "oof"
TEST_PREDS_DIR = REPO_ROOT / "experiments" / "test_preds"

REPORT_PATH = RESULTS_DIR / "eda_report.md"
STATS_PATH = RESULTS_DIR / "eda_stats.json"
EXP_PATH = RESULTS_DIR / "exp001.json"
OOF_PATH = OOF_DIR / "oof_none_exp001.npy"
TEST_PREDS_PATH = TEST_PREDS_DIR / "test_none_exp001.npy"


def well_id(path: Path) -> str:
    return path.name.split("__", maxsplit=1)[0]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "None"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):,.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def series_stats(values: list[float] | pd.Series) -> dict[str, float | int | None]:
    s = pd.Series(values, dtype="float64").dropna()
    if s.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def distribution(values: pd.Series) -> dict[str, float | int | None]:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
        }
    q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "count": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
        "p5": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
    }


def collect_horizontal_stats(paths: list[Path], include_tvt: bool) -> tuple[pd.DataFrame, dict[str, int], pd.Series, pd.Series]:
    rows: list[dict[str, Any]] = []
    missing: dict[str, int] = {}
    gr_values: list[pd.Series] = []
    tvt_values: list[pd.Series] = []

    for path in paths:
        df = pd.read_csv(path)
        for column, count in df.isna().sum().items():
            missing[column] = missing.get(column, 0) + int(count)

        gr = pd.to_numeric(df["GR"], errors="coerce") if "GR" in df else pd.Series(dtype="float64")
        depth = pd.to_numeric(df["MD"], errors="coerce") if "MD" in df else pd.Series(dtype="float64")
        row: dict[str, Any] = {
            "well_id": well_id(path),
            "n_rows": int(len(df)),
            "depth_min": float(depth.min()) if not depth.dropna().empty else None,
            "depth_max": float(depth.max()) if not depth.dropna().empty else None,
            "gr_mean": float(gr.mean()) if not gr.dropna().empty else None,
            "gr_std": float(gr.std(ddof=1)) if gr.dropna().size > 1 else 0.0,
            "gr_min": float(gr.min()) if not gr.dropna().empty else None,
            "gr_max": float(gr.max()) if not gr.dropna().empty else None,
        }
        if not gr.dropna().empty:
            gr_values.append(gr)

        if include_tvt:
            tvt = pd.to_numeric(df["TVT"], errors="coerce") if "TVT" in df else pd.Series(dtype="float64")
            row.update(
                {
                    "tvt_mean": float(tvt.mean()) if not tvt.dropna().empty else None,
                    "tvt_std": float(tvt.std(ddof=1)) if tvt.dropna().size > 1 else 0.0,
                    "tvt_min": float(tvt.min()) if not tvt.dropna().empty else None,
                    "tvt_max": float(tvt.max()) if not tvt.dropna().empty else None,
                    "tvt_range": float(tvt.max() - tvt.min()) if not tvt.dropna().empty else None,
                }
            )
            if {"GR", "TVT"}.issubset(df.columns):
                pair = df[["GR", "TVT"]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(pair) > 1 and pair["GR"].nunique() > 1 and pair["TVT"].nunique() > 1:
                    row["gr_tvt_corr"] = float(pair["GR"].corr(pair["TVT"]))
                else:
                    row["gr_tvt_corr"] = None
            if not tvt.dropna().empty:
                tvt_values.append(tvt)

        rows.append(row)

    gr_all = pd.concat(gr_values, ignore_index=True) if gr_values else pd.Series(dtype="float64")
    tvt_all = pd.concat(tvt_values, ignore_index=True) if tvt_values else pd.Series(dtype="float64")
    return pd.DataFrame(rows), missing, gr_all, tvt_all


def collect_typewell_stats(paths: list[Path], horizontal_depth_min: float, horizontal_depth_max: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        df = pd.read_csv(path)
        depth = pd.to_numeric(df["TVT"], errors="coerce") if "TVT" in df else pd.Series(dtype="float64")
        gr = pd.to_numeric(df["GR"], errors="coerce") if "GR" in df else pd.Series(dtype="float64")
        depth_min = float(depth.min()) if not depth.dropna().empty else None
        depth_max = float(depth.max()) if not depth.dropna().empty else None
        overlaps = (
            depth_min is not None
            and depth_max is not None
            and depth_max >= horizontal_depth_min
            and depth_min <= horizontal_depth_max
        )
        rows.append(
            {
                "well_id": well_id(path),
                "n_rows": int(len(df)),
                "depth_column": "TVT",
                "depth_min": depth_min,
                "depth_max": depth_max,
                "gr_mean": float(gr.mean()) if not gr.dropna().empty else None,
                "gr_std": float(gr.std(ddof=1)) if gr.dropna().size > 1 else 0.0,
                "overlaps_horizontal_md_range": bool(overlaps),
            }
        )
    return pd.DataFrame(rows)


def print_sample(label: str, path: Path) -> tuple[list[str], list[str], tuple[int, int]]:
    df = pd.read_csv(path)
    print(f"\n## {label}: {path.name}")
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("Dtypes:")
    print(df.dtypes.to_string())
    print("Head(3):")
    print(df.head(3).to_string(index=False))
    return list(df.columns), [str(dtype) for dtype in df.dtypes], tuple(df.shape)


def write_report(
    train_stats: pd.DataFrame,
    test_stats: pd.DataFrame,
    type_stats: pd.DataFrame,
    tvt_dist: dict[str, Any],
    gr_train_dist: dict[str, Any],
    gr_test_dist: dict[str, Any],
    corr_stats: dict[str, Any],
    missing_train: dict[str, int],
    missing_test: dict[str, int],
) -> str:
    n_train = len(train_stats)
    n_test = len(test_stats)
    n_type = len(type_stats)
    train_rows = int(train_stats["n_rows"].sum())
    test_rows = int(test_stats["n_rows"].sum())
    type_rows = int(type_stats["n_rows"].sum())
    tvt_range_stats = series_stats(train_stats["tvt_range"])
    overlap_rate = float(type_stats["overlaps_horizontal_md_range"].mean()) if n_type else 0.0

    key_findings = [
        f"The dataset has {n_train} train horizontal wells, {n_test} test horizontal wells, and {n_type} train typewell files.",
        f"Train horizontals contain {train_rows:,} rows; test horizontals contain {test_rows:,} rows.",
        f"Test horizontal wells are few ({n_test}) relative to train, so validation should be grouped by well rather than row-random.",
        f"TVT ranges from {fmt(tvt_dist['min'])} to {fmt(tvt_dist['max'])}, with mean {fmt(tvt_dist['mean'])} and std {fmt(tvt_dist['std'])}.",
        f"Per-well TVT range varies from {fmt(tvt_range_stats['min'])} to {fmt(tvt_range_stats['max'])}, indicating different target span difficulty by well.",
        f"Train GR mean/std are {fmt(gr_train_dist['mean'])}/{fmt(gr_train_dist['std'])}; test GR mean/std are {fmt(gr_test_dist['mean'])}/{fmt(gr_test_dist['std'])}.",
        f"Mean per-well GR/TVT correlation is {fmt(corr_stats['mean'])} with std {fmt(corr_stats['std'])}, so raw GR alone is not a stable linear target proxy.",
        f"{overlap_rate:.1%} of train typewell TVT ranges overlap the overall train horizontal MD range.",
        "Typewells use `TVT` as their depth-like column and include optional `Geology`; horizontals use `MD` plus trajectory columns.",
        "No missing values were observed in core train/test horizontal columns." if not any(missing_train.values()) and not any(missing_test.values()) else "Some columns contain missing values and should be guarded before feature generation.",
    ]

    feature_ideas = [
        "Use grouped validation by `well_id` to estimate generalization to unseen wells.",
        "Create local GR window features: rolling mean, std, min, max, slope, and percentile rank along measured depth.",
        "Use `TVT_input` cautiously as a strong baseline feature; verify whether its semantics are available at inference without leakage.",
        "Add trajectory features from `MD`, `X`, `Y`, and `Z`: deltas, local inclination proxies, curvature proxies, and depth-normalized movement.",
        "Align horizontal wells to typewell GR signatures with depth offsets or dynamic time warping style features.",
        "Normalize GR per well and possibly per nearby typewell to reduce well-specific scale shifts.",
    ]

    quality_issues = [
        "Typewell and horizontal files have different schemas; feature code must branch by file type.",
        "`Geology` exists in typewells only and is sparse, so it is not a direct horizontal-well feature.",
        "The test set has only three horizontal wells, making leaderboard sensitivity likely.",
        "Depth meaning differs by file family: horizontals expose `MD`, while typewells expose `TVT`.",
    ]
    missing_cols = sorted({k for k, v in {**missing_train, **missing_test}.items() if v > 0})
    if missing_cols:
        quality_issues.append(f"Missing values are present in: {', '.join(missing_cols)}.")

    shape_table = md_table(
        ["Dataset", "Files", "Rows"],
        [
            ["train horizontal wells", n_train, f"{train_rows:,}"],
            ["test horizontal wells", n_test, f"{test_rows:,}"],
            ["train typewell files", n_type, f"{type_rows:,}"],
        ],
    )
    report = f"""# exp001 EDA Report

## Data Shape Summary

{shape_table}

## Key Findings

""" + "\n".join(f"- {item}" for item in key_findings) + f"""

## Feature Engineering Ideas

{chr(10).join(f"- {item}" for item in feature_ideas)}

## Data Quality Issues

{chr(10).join(f"- {item}" for item in quality_issues)}

## TVT Distribution

- Mean: {fmt(tvt_dist['mean'])}
- Std: {fmt(tvt_dist['std'])}
- Min / p5 / p25 / p50 / p75 / p95 / max: {fmt(tvt_dist['min'])} / {fmt(tvt_dist['p5'])} / {fmt(tvt_dist['p25'])} / {fmt(tvt_dist['p50'])} / {fmt(tvt_dist['p75'])} / {fmt(tvt_dist['p95'])} / {fmt(tvt_dist['max'])}
- Per-well TVT range mean/std/min/max: {fmt(tvt_range_stats['mean'])} / {fmt(tvt_range_stats['std'])} / {fmt(tvt_range_stats['min'])} / {fmt(tvt_range_stats['max'])}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    return key_findings[3]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)

    train_horizontal = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    test_horizontal = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    train_typewells = sorted(TRAIN_DIR.glob("*__typewell.csv"))

    print("# 1. File inventory")
    print(f"Train horizontal_well files: {len(train_horizontal)}")
    print(f"Test horizontal_well files: {len(test_horizontal)}")
    print(f"Train typewell files: {len(train_typewells)}")
    print(f"Sample train horizontal_well filename: {train_horizontal[0].name if train_horizontal else 'None'}")
    print(f"Sample test horizontal_well filename: {test_horizontal[0].name if test_horizontal else 'None'}")
    print(f"Sample train typewell filename: {train_typewells[0].name if train_typewells else 'None'}")

    print("\n# 2. Column structure")
    train_columns, _, _ = print_sample("Train horizontal well sample", train_horizontal[0])
    type_columns, _, _ = print_sample("Train typewell sample", train_typewells[0])
    test_columns, _, _ = print_sample("Test horizontal well sample", test_horizontal[0])

    print("\n# 3. Train well statistics")
    train_stats, train_missing, train_gr, train_tvt = collect_horizontal_stats(train_horizontal, include_tvt=True)
    print("Per-well stats summary:")
    print(train_stats.describe(include="all").to_string())
    print("\nPer-well TVT range sorted descending:")
    print(train_stats[["well_id", "tvt_range"]].sort_values("tvt_range", ascending=False).to_string(index=False))

    print("\n# 4. Test well statistics")
    test_stats, test_missing, test_gr, _ = collect_horizontal_stats(test_horizontal, include_tvt=False)
    print(f"Test stats shape: {test_stats.shape}")
    print("Test GR stats:")
    print(test_stats[["well_id", "n_rows", "gr_mean", "gr_std", "gr_min", "gr_max"]].to_string(index=False))
    print(test_stats[["n_rows", "gr_mean", "gr_std", "gr_min", "gr_max"]].describe().to_string())

    print("\n# 5. Typewell statistics")
    horizontal_md_min = float(train_stats["depth_min"].min())
    horizontal_md_max = float(train_stats["depth_max"].max())
    type_stats = collect_typewell_stats(train_typewells, horizontal_md_min, horizontal_md_max)
    print(type_stats.to_string(index=False))
    print(
        "Typewell ranges overlapping overall train horizontal MD range: "
        f"{int(type_stats['overlaps_horizontal_md_range'].sum())}/{len(type_stats)}"
    )

    print("\n# 6. Missing values")
    print("Train horizontal NaNs by column:")
    print(pd.Series(train_missing).sort_index().to_string())
    print("Test horizontal NaNs by column:")
    print(pd.Series(test_missing).sort_index().to_string())

    print("\n# 7. TVT distribution")
    tvt_dist = distribution(train_tvt)
    print(pd.Series(tvt_dist).to_string())
    tvt_range_stats = series_stats(train_stats["tvt_range"])
    print("Per-well TVT range statistics:")
    print(pd.Series(tvt_range_stats).to_string())

    print("\n# 8. GR correlation with TVT")
    corr_stats = series_stats(train_stats["gr_tvt_corr"])
    print(f"Mean per-well correlation(GR, TVT): {fmt(corr_stats['mean'])}")
    print(f"Std per-well correlation(GR, TVT): {fmt(corr_stats['std'])}")

    gr_train_dist = distribution(train_gr)
    gr_test_dist = distribution(test_gr)
    note = write_report(
        train_stats,
        test_stats,
        type_stats,
        tvt_dist,
        gr_train_dist,
        gr_test_dist,
        corr_stats,
        train_missing,
        test_missing,
    )

    missing_value_columns = sorted({k for k, v in {**train_missing, **test_missing}.items() if v > 0})
    stats_json = {
        "n_train_wells": len(train_horizontal),
        "n_test_wells": len(test_horizontal),
        "n_typewell_files": len(train_typewells),
        "train_total_rows": int(train_stats["n_rows"].sum()),
        "test_total_rows": int(test_stats["n_rows"].sum()),
        "train_columns": train_columns,
        "test_columns": test_columns,
        "typewell_columns": type_columns,
        "tvt_mean": tvt_dist["mean"],
        "tvt_std": tvt_dist["std"],
        "tvt_min": tvt_dist["min"],
        "tvt_max": tvt_dist["max"],
        "gr_mean": gr_train_dist["mean"],
        "gr_std": gr_train_dist["std"],
        "gr_tvt_corr_mean": corr_stats["mean"],
        "gr_tvt_corr_std": corr_stats["std"],
        "has_missing_values": bool(missing_value_columns),
        "missing_value_columns": missing_value_columns,
        "train_missing_values": train_missing,
        "test_missing_values": test_missing,
        "tvt_distribution": tvt_dist,
        "per_well_tvt_range": tvt_range_stats,
    }
    STATS_PATH.write_text(json.dumps(jsonable(stats_json), indent=2), encoding="utf-8")

    exp_json = {
        "experiment_id": "exp001",
        "model": "none",
        "cv_rmse": None,
        "cv_rmse_std": None,
        "features_used": [],
        "n_features": 0,
        "training_time_seconds": 0,
        "notes": note,
    }
    EXP_PATH.write_text(json.dumps(exp_json, indent=2), encoding="utf-8")

    np.save(OOF_PATH, np.array([]))
    np.save(TEST_PREDS_PATH, np.array([]))

    print("\nOutput files:")
    print(REPORT_PATH.relative_to(REPO_ROOT))
    print(STATS_PATH.relative_to(REPO_ROOT))
    print(EXP_PATH.relative_to(REPO_ROOT))
    print(OOF_PATH.relative_to(REPO_ROOT))
    print(TEST_PREDS_PATH.relative_to(REPO_ROOT))
    print("EXPERIMENT COMPLETE: cv_rmse=None")


if __name__ == "__main__":
    main()
