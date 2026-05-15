#!/usr/bin/env python3
"""Phase 1 EDA for the ROGII wellbore geology prediction data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "rogii-wellbore-geology-prediction"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
PLOTS_DIR = RESULTS_DIR / "eda_plots"
REPORT_PATH = RESULTS_DIR / "eda_report.md"
STATS_PATH = RESULTS_DIR / "eda_stats.json"

PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
SUMMARY_COLUMNS = [
    "MD",
    "X",
    "Y",
    "Z",
    "TVD",
    "TVT",
    "TVT_input",
    "GR",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
]


def split_kind(path: Path) -> str:
    if path.name.endswith("__horizontal_well.csv"):
        return "horizontal_well"
    if path.name.endswith("__typewell.csv"):
        return "typewell"
    if path.suffix.lower() == ".png":
        return "image"
    return path.suffix.lower().lstrip(".") or "unknown"


def well_id_from_path(path: Path) -> str:
    name = path.stem
    return name.replace("__horizontal_well", "").replace("__typewell", "")


def read_csv(path: Path, split: str, kind: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["well_id"] = well_id_from_path(path)
    df["file_name"] = path.name
    df["split"] = split
    df["file_kind"] = kind
    return df


def load_split(split: str, split_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(split_dir.glob("*.csv")):
        frames.append(read_csv(path, split, split_kind(path)))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        non_null = values.dropna()
        if non_null.empty:
            result[col] = {"count": 0, "missing": int(values.isna().sum())}
            continue
        stats: dict[str, float | int | None] = {
            "count": int(non_null.size),
            "missing": int(values.isna().sum()),
            "mean": float(non_null.mean()),
            "std": float(non_null.std()),
            "min": float(non_null.min()),
            "max": float(non_null.max()),
        }
        for pct, value in non_null.quantile(PERCENTILES).items():
            stats[f"p{int(round(pct * 100)):02d}"] = float(value)
        result[col] = stats
    return result


def missing_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int]]:
    result = {}
    row_count = len(df)
    for col in columns:
        missing = row_count if col not in df.columns else int(df[col].isna().sum())
        result[col] = {
            "missing": missing,
            "missing_pct": float(missing / row_count) if row_count else 0.0,
        }
    return result


def safe_corr(df: pd.DataFrame, left: str, right: str) -> float | None:
    if left not in df.columns or right not in df.columns:
        return None
    pair = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 2 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return None
    return float(pair[left].corr(pair[right]))


def per_well_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, kind, well_id), group in df.groupby(["split", "file_kind", "well_id"], dropna=False):
        row: dict[str, Any] = {
            "split": split,
            "file_kind": kind,
            "well_id": well_id,
            "rows": int(len(group)),
            "columns": sorted([c for c in group.columns if group[c].notna().any()]),
        }
        for col in ["MD", "TVT", "TVT_input", "GR", "X", "Y", "Z"]:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                if not values.empty:
                    row[f"{col}_min"] = float(values.min())
                    row[f"{col}_max"] = float(values.max())
                    row[f"{col}_mean"] = float(values.mean())
                    row[f"{col}_std"] = float(values.std())
                    row[f"{col}_range"] = float(values.max() - values.min())
        rows.append(row)
    return pd.DataFrame(rows)


def sample_for_plot(df: pd.DataFrame, n: int = 150_000) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


def select_representative_wells(well_stats: pd.DataFrame) -> list[str]:
    train_horizontal = well_stats[
        (well_stats["split"] == "train")
        & (well_stats["file_kind"] == "horizontal_well")
        & well_stats["TVT_range"].notna()
    ].sort_values("rows")
    if train_horizontal.empty:
        return []
    positions = [0, len(train_horizontal) // 2, len(train_horizontal) - 1]
    return train_horizontal.iloc[positions]["well_id"].drop_duplicates().tolist()


def save_line_plot(df: pd.DataFrame, wells: list[str], x: str, y: str, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(len(wells), 1, figsize=(10, 3.2 * max(len(wells), 1)), sharex=False)
    if len(wells) == 1:
        axes = [axes]
    for ax, well_id in zip(axes, wells):
        group = df[df["well_id"] == well_id].sort_values(x)
        ax.plot(group[x], group[y], linewidth=1.0)
        ax.set_title(well_id)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def top_outlier_wells(well_stats: pd.DataFrame) -> list[dict[str, Any]]:
    train = well_stats[
        (well_stats["split"] == "train")
        & (well_stats["file_kind"] == "horizontal_well")
        & well_stats["GR_mean"].notna()
        & well_stats["TVT_range"].notna()
    ].copy()
    if train.empty:
        return []
    metrics = ["rows", "GR_mean", "GR_std", "TVT_range", "MD_range"]
    for col in metrics:
        std = train[col].std()
        train[f"{col}_z"] = 0.0 if pd.isna(std) or std == 0 else (train[col] - train[col].mean()) / std
    z_cols = [f"{col}_z" for col in metrics]
    train["outlier_score"] = train[z_cols].abs().max(axis=1)
    cols = ["well_id", "rows", "GR_mean", "GR_std", "TVT_range", "MD_range", "outlier_score"]
    return train.sort_values("outlier_score", ascending=False)[cols].head(10).to_dict("records")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    train = load_split("train", TRAIN_DIR)
    test = load_split("test", TEST_DIR)
    all_data = pd.concat([train, test], ignore_index=True, sort=False)
    horizontal = all_data[all_data["file_kind"] == "horizontal_well"].copy()
    train_horizontal = horizontal[horizontal["split"] == "train"].copy()
    test_horizontal = horizontal[horizontal["split"] == "test"].copy()
    typewells = all_data[all_data["file_kind"] == "typewell"].copy()

    all_files = sorted(DATA_ROOT.rglob("*"))
    file_inventory = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "split": path.parent.name if path.parent.name in {"train", "test"} else "root",
            "name": path.name,
            "suffix": path.suffix.lower(),
            "kind": split_kind(path),
            "size_bytes": path.stat().st_size,
        }
        for path in all_files
        if path.is_file()
    ]
    file_format_counts = pd.Series([item["suffix"] or "<none>" for item in file_inventory]).value_counts().to_dict()
    naming_pattern_counts = pd.Series([item["kind"] for item in file_inventory]).value_counts().to_dict()

    well_stats = per_well_summary(all_data)
    representative_wells = select_representative_wells(well_stats)
    plot_paths: dict[str, str] = {}

    if representative_wells:
        tvt_plot = PLOTS_DIR / "tvt_vs_depth_representative_wells.png"
        gr_plot = PLOTS_DIR / "gr_vs_depth_representative_wells.png"
        save_line_plot(train_horizontal, representative_wells, "MD", "TVT", "TVT versus measured depth", tvt_plot)
        save_line_plot(train_horizontal, representative_wells, "MD", "GR", "GR versus measured depth", gr_plot)
        plot_paths["tvt_vs_depth_representative_wells"] = str(tvt_plot.relative_to(REPO_ROOT))
        plot_paths["gr_vs_depth_representative_wells"] = str(gr_plot.relative_to(REPO_ROOT))

    if {"GR", "TVT"}.issubset(train_horizontal.columns):
        scatter = PLOTS_DIR / "gr_vs_tvt_train_sample.png"
        sample = sample_for_plot(train_horizontal[["GR", "TVT", "well_id"]].dropna())
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sample["GR"], sample["TVT"], s=3, alpha=0.15)
        ax.set_xlabel("GR")
        ax.set_ylabel("TVT")
        ax.set_title("Training horizontal wells: GR versus TVT")
        fig.tight_layout()
        fig.savefig(scatter, dpi=160)
        plt.close(fig)
        plot_paths["gr_vs_tvt_train_sample"] = str(scatter.relative_to(REPO_ROOT))

    if "GR" in typewells.columns and "GR" in horizontal.columns:
        dist_plot = PLOTS_DIR / "gr_distribution_typewell_vs_lateral.png"
        dist_df = pd.concat(
            [
                sample_for_plot(horizontal[["GR"]].dropna()).assign(source="horizontal"),
                sample_for_plot(typewells[["GR"]].dropna()).assign(source="typewell"),
            ],
            ignore_index=True,
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.kdeplot(data=dist_df, x="GR", hue="source", common_norm=False, ax=ax)
        ax.set_title("GR distribution: typewell references versus horizontal wells")
        fig.tight_layout()
        fig.savefig(dist_plot, dpi=160)
        plt.close(fig)
        plot_paths["gr_distribution_typewell_vs_lateral"] = str(dist_plot.relative_to(REPO_ROOT))

    if "GR" in train_horizontal.columns and "GR" in test_horizontal.columns:
        train_test_plot = PLOTS_DIR / "train_test_gr_distribution.png"
        split_df = pd.concat(
            [
                sample_for_plot(train_horizontal[["GR"]].dropna()).assign(split="train"),
                test_horizontal[["GR"]].dropna().assign(split="test"),
            ],
            ignore_index=True,
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.kdeplot(data=split_df, x="GR", hue="split", common_norm=False, ax=ax)
        ax.set_title("GR distribution: train versus test horizontal wells")
        fig.tight_layout()
        fig.savefig(train_test_plot, dpi=160)
        plt.close(fig)
        plot_paths["train_test_gr_distribution"] = str(train_test_plot.relative_to(REPO_ROOT))

    if {"GR_range", "TVT_range"}.issubset(well_stats.columns):
        ranges_plot = PLOTS_DIR / "well_level_gr_tvt_ranges.png"
        plot_df = well_stats[
            (well_stats["split"] == "train")
            & (well_stats["file_kind"] == "horizontal_well")
            & well_stats["GR_range"].notna()
            & well_stats["TVT_range"].notna()
        ]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(plot_df["GR_range"], plot_df["TVT_range"], s=12, alpha=0.55)
        ax.set_xlabel("Well GR range")
        ax.set_ylabel("Well TVT range")
        ax.set_title("Well-level GR and TVT variation")
        fig.tight_layout()
        fig.savefig(ranges_plot, dpi=160)
        plt.close(fig)
        plot_paths["well_level_gr_tvt_ranges"] = str(ranges_plot.relative_to(REPO_ROOT))

    key_columns = sorted(set(SUMMARY_COLUMNS) & set(all_data.columns))
    trajectory_columns = [c for c in ["MD", "TVD", "inclination", "Inclination", "azimuth", "Azimuth", "X", "Y", "Z"] if c in all_data.columns]
    dogleg_stats = {}
    if {"X", "Y", "Z", "MD"}.issubset(horizontal.columns):
        tmp = horizontal.sort_values(["split", "well_id", "MD"]).copy()
        for col in ["X", "Y", "Z", "MD"]:
            tmp[f"d_{col}"] = tmp.groupby(["split", "well_id"])[col].diff()
        tmp["three_d_step"] = np.sqrt(tmp["d_X"] ** 2 + tmp["d_Y"] ** 2 + tmp["d_Z"] ** 2)
        tmp["tortuosity_ratio"] = tmp["three_d_step"] / tmp["d_MD"].replace(0, np.nan)
        dogleg_stats = numeric_summary(tmp, ["three_d_step", "tortuosity_ratio"])

    train_test_overlap = sorted(
        (set(train_horizontal.columns) & set(test_horizontal.columns))
        - {"TVT", "file_name", "split", "file_kind"}
    )
    overlap_distribution = {}
    for col in train_test_overlap:
        if pd.api.types.is_numeric_dtype(train_horizontal[col]) and pd.api.types.is_numeric_dtype(test_horizontal[col]):
            overlap_distribution[col] = {
                "train_mean": float(train_horizontal[col].mean()),
                "test_mean": float(test_horizontal[col].mean()),
                "train_std": float(train_horizontal[col].std()),
                "test_std": float(test_horizontal[col].std()),
                "mean_delta_test_minus_train": float(test_horizontal[col].mean() - train_horizontal[col].mean()),
            }

    typewell_depth = well_stats[well_stats["file_kind"] == "typewell"][
        ["split", "well_id", "rows", "TVT_min", "TVT_max", "TVT_range", "GR_mean", "GR_std", "GR_min", "GR_max"]
    ].to_dict("records")

    train_typewells = typewells[typewells["split"] == "train"]
    test_typewells = typewells[typewells["split"] == "test"]

    stats = {
        "data_root": str(DATA_ROOT.relative_to(REPO_ROOT)),
        "file_inventory": file_inventory,
        "file_format_counts": file_format_counts,
        "naming_pattern_counts": naming_pattern_counts,
        "well_counts": {
            "train_horizontal": int(train_horizontal["well_id"].nunique()),
            "train_typewell": int(typewells[typewells["split"] == "train"]["well_id"].nunique()),
            "test_horizontal": int(test_horizontal["well_id"].nunique()),
            "test_typewell": int(typewells[typewells["split"] == "test"]["well_id"].nunique()),
            "all_unique_wells": int(all_data["well_id"].nunique()),
        },
        "shape_summary": {
            "train_horizontal": {"rows": int(len(train_horizontal)), "columns": int(train_horizontal.shape[1])},
            "train_typewell": {"rows": int(len(train_typewells)), "columns": int(train_typewells.shape[1])},
            "test_horizontal": {"rows": int(len(test_horizontal)), "columns": int(test_horizontal.shape[1])},
            "test_typewell": {"rows": int(len(test_typewells)), "columns": int(test_typewells.shape[1])},
        },
        "per_column_summary_statistics": {
            "all_data": numeric_summary(all_data, key_columns),
            "train_horizontal": numeric_summary(train_horizontal, key_columns),
            "test_horizontal": numeric_summary(test_horizontal, key_columns),
            "typewells": numeric_summary(typewells, key_columns),
        },
        "missing_value_counts": {
            "all_data": missing_summary(all_data, ["well_id", "MD", "X", "Y", "Z", "TVT", "TVT_input", "GR", "Geology"]),
            "train_horizontal": missing_summary(train_horizontal, ["well_id", "MD", "X", "Y", "Z", "TVT", "TVT_input", "GR"]),
            "test_horizontal": missing_summary(test_horizontal, ["well_id", "MD", "X", "Y", "Z", "TVT_input", "GR"]),
            "typewells": missing_summary(typewells, ["well_id", "TVT", "GR", "Geology"]),
        },
        "per_well_summary_statistics": well_stats.to_dict("records"),
        "typewell_depth_coverage": typewell_depth,
        "trajectory_columns": trajectory_columns,
        "relationships": {
            "train_horizontal_gr_tvt_corr": safe_corr(train_horizontal, "GR", "TVT"),
            "train_horizontal_tvd_tvt_corr": safe_corr(train_horizontal, "TVD", "TVT"),
            "train_horizontal_z_tvt_corr": safe_corr(train_horizontal, "Z", "TVT"),
            "train_horizontal_tvt_input_tvt_corr": safe_corr(train_horizontal, "TVT_input", "TVT"),
            "trajectory_step_proxy": dogleg_stats,
        },
        "train_test_overlap_distribution": overlap_distribution,
        "outlier_wells": top_outlier_wells(well_stats),
        "representative_wells": representative_wells,
        "plot_paths": plot_paths,
    }

    write_report(stats, well_stats)
    STATS_PATH.write_text(json.dumps(json_ready(stats), indent=2, sort_keys=True), encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def md_table(rows: list[list[Any]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def write_report(stats: dict[str, Any], well_stats: pd.DataFrame) -> None:
    shape = stats["shape_summary"]
    shape_rows = [
        [name, fmt(values["rows"], 0), fmt(values["columns"], 0)]
        for name, values in shape.items()
    ]
    train_gr = stats["per_column_summary_statistics"]["train_horizontal"]["GR"]
    train_tvt = stats["per_column_summary_statistics"]["train_horizontal"]["TVT"]
    type_gr = stats["per_column_summary_statistics"]["typewells"]["GR"]
    train_test_gr = stats["train_test_overlap_distribution"].get("GR", {})
    missing_train = stats["missing_value_counts"]["train_horizontal"]
    outliers = stats["outlier_wells"][:5]

    typewell_rows = well_stats[well_stats["file_kind"] == "typewell"]["rows"]
    horizontal_rows = well_stats[well_stats["file_kind"] == "horizontal_well"]["rows"]
    tvt_range = well_stats[
        (well_stats["split"] == "train")
        & (well_stats["file_kind"] == "horizontal_well")
        & well_stats["TVT_range"].notna()
    ]["TVT_range"]
    gr_range = well_stats[
        (well_stats["file_kind"] == "horizontal_well")
        & well_stats["GR_range"].notna()
    ]["GR_range"]

    findings = [
        f"Observed layout is nested under `{stats['data_root']}` with train/test folders, not directly under `data/raw/train`.",
        f"There are {stats['well_counts']['train_horizontal']} train horizontal wells and {stats['well_counts']['test_horizontal']} test horizontal wells; train also has {stats['well_counts']['train_typewell']} typewell files and test has {stats['well_counts']['test_typewell']}.",
        f"Train horizontal TVT spans {fmt(train_tvt['min'])} to {fmt(train_tvt['max'])}, with mean {fmt(train_tvt['mean'])} and std {fmt(train_tvt['std'])}.",
        f"Train horizontal GR spans {fmt(train_gr['min'])} to {fmt(train_gr['max'])}, with mean {fmt(train_gr['mean'])}; typewell GR mean is {fmt(type_gr['mean'])}.",
        f"GR and TVT have weak global linear correlation in train horizontals: r={fmt(stats['relationships']['train_horizontal_gr_tvt_corr'])}.",
        f"`TVT_input` is almost a direct target proxy where present in train horizontals: r={fmt(stats['relationships']['train_horizontal_tvt_input_tvt_corr'])}. Treat it carefully to avoid leakage assumptions.",
        f"Train/test horizontal GR means differ by {fmt(train_test_gr.get('mean_delta_test_minus_train'))}, so distribution shift appears modest but should be validated by well-grouped splits.",
        f"Typewells are shorter reference logs: median rows {fmt(typewell_rows.median())} versus horizontal median rows {fmt(horizontal_rows.median())}.",
        f"Well-level ranges vary materially: train horizontal median TVT range is {fmt(tvt_range.median())}; horizontal median GR range is {fmt(gr_range.median())}.",
    ]

    outlier_lines = [
        f"- `{row['well_id']}`: score {fmt(row['outlier_score'])}, rows {fmt(row['rows'], 0)}, GR mean {fmt(row['GR_mean'])}, TVT range {fmt(row['TVT_range'])}"
        for row in outliers
    ]

    report = f"""# Phase 1 EDA Report

Generated by `experiments/results/eda_script.py`.

## Data Shape Summary

{md_table(shape_rows, ["Dataset", "Rows", "Columns"])}

## File Inventory Summary

- Data root: `{stats['data_root']}`
- File formats: {json.dumps(stats['file_format_counts'], sort_keys=True)}
- Naming patterns: {json.dumps(stats['naming_pattern_counts'], sort_keys=True)}
- Trajectory columns observed: {", ".join(stats['trajectory_columns'])}
- Representative plotted wells: {", ".join(stats['representative_wells'])}

## High-Signal Findings

""" + "\n".join(f"{idx}. {finding}" for idx, finding in enumerate(findings, start=1)) + f"""

## Target Variable

- Train horizontal `TVT`: mean {fmt(train_tvt['mean'])}, std {fmt(train_tvt['std'])}, min {fmt(train_tvt['min'])}, p05 {fmt(train_tvt['p05'])}, median {fmt(train_tvt['p50'])}, p95 {fmt(train_tvt['p95'])}, max {fmt(train_tvt['max'])}.
- Median per-well train horizontal TVT range: {fmt(tvt_range.median())}; max per-well TVT range: {fmt(tvt_range.max())}.
- TVT versus depth plot: `{stats['plot_paths'].get('tvt_vs_depth_representative_wells', 'n/a')}`.

## Gamma Ray Log

- Train horizontal `GR`: mean {fmt(train_gr['mean'])}, std {fmt(train_gr['std'])}, min {fmt(train_gr['min'])}, p05 {fmt(train_gr['p05'])}, median {fmt(train_gr['p50'])}, p95 {fmt(train_gr['p95'])}, max {fmt(train_gr['max'])}.
- Median per-horizontal-well GR range: {fmt(gr_range.median())}; max per-horizontal-well GR range: {fmt(gr_range.max())}.
- Global train horizontal GR/TVT correlation: {fmt(stats['relationships']['train_horizontal_gr_tvt_corr'])}.
- GR versus depth plot: `{stats['plot_paths'].get('gr_vs_depth_representative_wells', 'n/a')}`.

## Typewell Reference

- Typewell files: {fmt(stats['well_counts']['train_typewell'] + stats['well_counts']['test_typewell'], 0)}.
- Typewell row coverage: min {fmt(typewell_rows.min(), 0)}, median {fmt(typewell_rows.median(), 0)}, max {fmt(typewell_rows.max(), 0)}.
- Typewell GR distribution plot: `{stats['plot_paths'].get('gr_distribution_typewell_vs_lateral', 'n/a')}`.

## Trajectory Data

- Horizontal wells provide `MD`, `X`, `Y`, and `Z`; no explicit `TVD`, inclination, or azimuth columns were observed.
- `Z` and `TVT` correlation in train horizontals: {fmt(stats['relationships']['train_horizontal_z_tvt_corr'])}.
- A trajectory step proxy was computed from consecutive `X/Y/Z` deltas and normalized by `MD` delta; see `trajectory_step_proxy` in `eda_stats.json`.

## Missing Data

- Train horizontal missing `GR`: {fmt(missing_train['GR']['missing'], 0)} rows; missing `TVT`: {fmt(missing_train['TVT']['missing'], 0)} rows; missing `TVT_input`: {fmt(missing_train['TVT_input']['missing'], 0)} rows.
- Combined-data missingness is inflated for columns that only exist in one file family, for example typewells do not carry trajectory columns and test horizontals do not carry `TVT`.
- `Geology` appears only in some train typewell files and is materially sparse; treat it as optional metadata unless a later phase explicitly uses it.

## Outlier Wells

{chr(10).join(outlier_lines)}

## Feature Engineering Ideas

- Use well-group normalized depth features from `MD`, including depth fraction and distance from heel.
- Add within-well GR rolling means, gradients, lags/leads, percentile rank, and z-score features.
- Use trajectory geometry from `X/Y/Z/MD`: local displacement, vertical change, lateral step ratio, and smoothed curvature proxies.
- Align horizontal wells to nearby/typewell GR patterns with lag/correlation features, but validate by grouped wells.
- Include formation marker offsets where available (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) and their distances to `Z` or `TVT_input`.

## Data Quality Issues

- File families have different schemas; loaders must separate horizontal and typewell records before modeling.
- Test horizontal wells have no `TVT`, so target checks must use train-only data.
- `TVT_input` is present in both train and test and is extremely correlated with train `TVT`; verify competition semantics before relying on it too heavily.
- Some typewell `Geology` values are missing and not available in test typewells.
- Train/test distribution checks should be repeated with grouped validation, because row-level random splits would leak well identity and depth continuity.

## Baseline Recommendation

Start with a grouped-by-well baseline using horizontal wells only. Predict `TVT` from `MD`, `GR`, `TVT_input`, `X/Y/Z`, formation marker offsets, and simple within-well rolling GR/depth features; evaluate with GroupKFold by `well_id` before adding typewell alignment features.

## Plot Outputs

""" + "\n".join(f"- `{path}`" for path in stats["plot_paths"].values()) + "\n"

    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
